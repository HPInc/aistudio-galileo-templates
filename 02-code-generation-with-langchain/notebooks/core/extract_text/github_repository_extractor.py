import os
import re
import requests
import shutil
import uuid
import logging
from typing import List, Dict, Any, Optional
import nbformat
from urllib.parse import urlparse

class GitHubRepositoryExtractor:
    """
    Extracts code and documentation from GitHub repositories.
    Works with multiple file types including Python files and Jupyter notebooks.
    
    Attributes:
        repo_url (str): GitHub repository URL
        repo_owner (str): GitHub repository owner
        repo_name (str): GitHub repository name
        save_dir (str): Local directory to save downloaded files
        verbose (bool): If True, enables logging output
        api_base_url (str): Base URL for GitHub API requests
        supported_extensions (tuple): File extensions this class will process
    """
    
    def __init__(self, repo_url: str, save_dir: str = './repo_files', verbose: bool = False,
                 supported_extensions: tuple = ('.py', '.ipynb', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.md')):
        """
        Initializes the GitHub repository extractor.
        
        Args:
            repo_url (str): URL to the GitHub repository
            save_dir (str): Directory to save downloaded files
            verbose (bool): Whether to enable verbose logging
            supported_extensions (tuple): File extensions to process
        """
        # Parse repository URL to extract owner and name
        parsed_url = self._parse_github_url(repo_url)
        self.repo_url = repo_url
        self.repo_owner = parsed_url["owner"]
        self.repo_name = parsed_url["repo"]
        self.save_dir = save_dir
        self.verbose = verbose
        self.supported_extensions = supported_extensions
        self.api_base_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/contents"
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.propagate = False
        self.logger.handlers.clear()
        
        if verbose:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.addHandler(logging.NullHandler())
    
    def _parse_github_url(self, url: str) -> Dict[str, str]:
        """
        Parses a GitHub URL to extract owner and repository name.
        
        Args:
            url (str): GitHub repository URL
            
        Returns:
            Dict with owner and repo names
            
        Raises:
            ValueError: If the URL is not a valid GitHub repository URL
        """
        # Parse the URL
        parsed = urlparse(url)
        
        # Validate that it's a GitHub URL
        if not parsed.netloc.endswith('github.com'):
            raise ValueError(f"Not a GitHub URL: {url}")
        
        # Extract path components
        path_parts = [p for p in parsed.path.split('/') if p]
        
        if len(path_parts) < 2:
            raise ValueError(f"Invalid GitHub repository URL: {url}")
            
        return {
            "owner": path_parts[0],
            "repo": path_parts[1]
        }
    
    def run(self) -> List[Dict]:
        """
        Main entry point - processes the repository and extracts code with context.
        
        Returns:
            List of dictionaries with extracted code and metadata
        """
        self.logger.info(f"Processing repository: {self.repo_owner}/{self.repo_name}")
        
        # Clean up save directory if it exists
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
            self.logger.info(f"Removed existing directory: {self.save_dir}")
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger.info(f"Created directory: {self.save_dir}")
        
        # Process the repository contents
        extracted_data = self._process_directory(self.api_base_url)
        self.logger.info(f"Extracted {len(extracted_data)} code snippets from repository")
        
        return extracted_data
    
    def _process_directory(self, api_url: str) -> List[Dict]:
        """
        Recursively processes a directory in the repository.
        
        Args:
            api_url (str): GitHub API URL for the directory
            
        Returns:
            List of dictionaries with extracted code and metadata
        """
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            contents = response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch contents from {api_url}: {str(e)}")
            return []
        
        all_data = []
        
        for item in contents:
            try:
                if item['type'] == 'file':
                    file_extension = os.path.splitext(item['name'])[1].lower()
                    
                    if file_extension in self.supported_extensions:
                        file_path = os.path.join(self.save_dir, item['path'])
                        
                        # Create directory structure if needed
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        
                        # Download and extract from the file
                        self._download_file(item['download_url'], file_path)
                        extracted = self._extract_from_file(file_path)
                        all_data.extend(extracted)
                
                elif item['type'] == 'dir':
                    # Process subdirectory recursively
                    subdir_api_url = item['url']
                    all_data.extend(self._process_directory(subdir_api_url))
            
            except Exception as e:
                self.logger.error(f"Error processing {item.get('path', 'unknown')}: {str(e)}")
        
        return all_data
    
    def _download_file(self, file_url: str, save_path: str) -> None:
        """
        Downloads a file from GitHub.
        
        Args:
            file_url (str): URL to download the file from
            save_path (str): Path to save the file to
        """
        try:
            response = requests.get(file_url)
            response.raise_for_status()
            
            # Create directory if needed
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"Downloaded: {save_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to download {file_url}: {str(e)}")
    
    def _extract_from_file(self, file_path: str) -> List[Dict]:
        """
        Extracts code and context from a file based on its type.
        
        Args:
            file_path (str): Path to the file to extract from
            
        Returns:
            List of dictionaries with extracted code and metadata
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.ipynb':
            return self._extract_from_notebook(file_path)
        else:
            return self._extract_from_code_file(file_path)
    
    def _extract_from_notebook(self, notebook_path: str) -> List[Dict]:
        """
        Extracts code cells and associated context from Jupyter notebooks.
        
        Args:
            notebook_path (str): Path to the notebook file
            
        Returns:
            List of dictionaries with extracted code and metadata
        """
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            extracted_data = []
            context = ''
            
            for cell in notebook['cells']:
                if cell['cell_type'] == 'markdown':
                    # Use markdown cells as context for subsequent code cells
                    context = ''.join(cell['source'])
                
                elif cell['cell_type'] == 'code' and cell['source'].strip():
                    cell_data = {
                        "id": str(uuid.uuid4()),
                        "embedding": None,
                        "code": ''.join(cell['source']),
                        "filename": os.path.relpath(notebook_path, self.save_dir),
                        "context": context
                    }
                    extracted_data.append(cell_data)
            
            return extracted_data
        
        except Exception as e:
            self.logger.error(f"Error extracting from notebook {notebook_path}: {str(e)}")
            return []
    
    def _extract_from_code_file(self, file_path: str) -> List[Dict]:
        """
        Extracts code and documentation from general code files.
        
        Args:
            file_path (str): Path to the code file
            
        Returns:
            List of dictionaries with extracted code and context
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Split the file into chunks with reasonable size
            chunks = self._split_code_into_chunks(content, file_path)
            
            # Convert chunks to the expected output format
            extracted_data = []
            for i, (code, doc_context) in enumerate(chunks):
                if code.strip():  # Only include non-empty code
                    chunk_data = {
                        "id": str(uuid.uuid4()),
                        "embedding": None,
                        "code": code,
                        "filename": os.path.relpath(file_path, self.save_dir),
                        "context": doc_context
                    }
                    extracted_data.append(chunk_data)
            
            return extracted_data
        
        except Exception as e:
            self.logger.error(f"Error extracting from file {file_path}: {str(e)}")
            return []
    
    def _split_code_into_chunks(self, content: str, file_path: str) -> List[tuple]:
        """
        Splits code files into logical chunks based on code structure and documentation.
        
        Args:
            content (str): File content
            file_path (str): Path to the file (for language detection)
            
        Returns:
            List of (code, documentation_context) tuples
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Set comment patterns based on file extension
        if file_extension in ('.py', '.ipynb'):
            # For Python files
            doc_pattern = r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|#.*$)'
            function_pattern = r'(def\s+\w+\(.*?\):|class\s+\w+.*?:)'
        elif file_extension in ('.js', '.ts'):
            # For JavaScript/TypeScript files
            doc_pattern = r'(\/\*[\s\S]*?\*\/|\/\/.*$)'
            function_pattern = r'(function\s+\w+\s*\(.*?\)|class\s+\w+|const\s+\w+\s*=\s*function)'
        elif file_extension in ('.java'):
            # For Java files
            doc_pattern = r'(\/\*[\s\S]*?\*\/|\/\/.*$)'
            function_pattern = r'(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(.*?\)'
        elif file_extension in ('.c', '.cpp', '.h'):
            # For C/C++ files
            doc_pattern = r'(\/\*[\s\S]*?\*\/|\/\/.*$)'
            function_pattern = r'(\w+\s+\w+\s*\(.*?\))'
        else:
            # Generic pattern
            doc_pattern = r'(\/\*[\s\S]*?\*\/|#.*$|\/\/.*$)'
            function_pattern = r'(\w+\s+\w+\s*\(.*?\))'
        
        lines = content.split('\n')
        chunks = []
        
        current_code = []
        current_context = ''
        last_doc = ''
        
        for i, line in enumerate(lines):
            # Check if line contains documentation
            doc_match = re.search(doc_pattern, line, re.MULTILINE)
            if doc_match:
                last_doc = doc_match.group(0)
                if not current_context:
                    current_context = last_doc
            
            # Check if line starts a new function or class
            func_match = re.search(function_pattern, line)
            if func_match and i > 0:
                # If we were collecting code, save the chunk
                if current_code:
                    chunks.append(('\n'.join(current_code), current_context))
                
                # Start a new chunk
                current_code = [line]
                current_context = last_doc
            else:
                current_code.append(line)
        
        # Don't forget the last chunk
        if current_code:
            chunks.append(('\n'.join(current_code), current_context))
        
        # If we didn't find natural breakpoints, split the file into reasonable chunks
        if len(chunks) <= 1 and len(lines) > 50:
            chunks = []
            chunk_size = 50
            for i in range(0, len(lines), chunk_size):
                chunk_lines = lines[i:min(i+chunk_size, len(lines))]
                chunks.append(('\n'.join(chunk_lines), f"Code chunk from lines {i+1}-{min(i+chunk_size, len(lines))}"))
        
        return chunks
