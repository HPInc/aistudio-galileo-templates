import os
import requests
import shutil
import nbformat
import uuid
import logging
from typing import List, Dict
 
class GitHubNotebookExtractor:
    """
    Extracts code and context from Jupyter notebooks in a GitHub repository.
    Downloads `.ipynb` files and extracts markdown + code cells.
 
    Attributes:
        repo_owner (str): GitHub repository owner.
        repo_name (str): GitHub repository name.
        save_dir (str): Local directory to save downloaded notebooks.
        verbose (bool): If True, enables logging output (INFO+ only).
    """
 
    def __init__(self, repo_owner: str, repo_name: str, save_dir: str = './notebooks', verbose: bool = False):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.save_dir = save_dir
        self.verbose = verbose
        self.api_base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents"
 
        # Setup logger
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
 
    def run(self) -> List[Dict]:
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
            self.logger.info(f"Removed existing directory: {self.save_dir}")
 
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger.info(f"Created directory: {self.save_dir}")
 
        return self._process_directory(self.api_base_url)
 
    def _process_directory(self, api_url: str) -> List[Dict]:
        response = requests.get(api_url)
        if response.status_code != 200:
            self.logger.error(f"Failed to fetch contents from {api_url} (status: {response.status_code})")
            return []
 
        contents = response.json()
        all_data = []
 
        for item in contents:
            if item['type'] == 'file' and item['name'].endswith('.ipynb'):
                notebook_path = os.path.join(self.save_dir, item['name'])
                self._download_file(item['download_url'], notebook_path)
                all_data.extend(self._extract_from_notebook(notebook_path))
            elif item['type'] == 'dir':
                subdir_url = f"{self.api_base_url}/{item['path']}"
                all_data.extend(self._process_directory(subdir_url))
 
        return all_data
 
    def _download_file(self, file_url: str, save_path: str) -> None:
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            self.logger.info(f"Downloaded: {save_path}")
        else:
            self.logger.warning(f"Failed to download {file_url} (status: {response.status_code})")
 
    def _extract_from_notebook(self, notebook_path: str) -> List[Dict]:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
 
        extracted_data = []
        context = ''
 
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                context = ''.join(cell['source'])
            elif cell['cell_type'] == 'code':
                cell_data = {
                    "id": str(uuid.uuid4()),
                    "embedding": None,
                    "code": ''.join(cell['source']),
                    "filename": os.path.basename(notebook_path),
                    "context": context
                }
                extracted_data.append(cell_data)
 
        return extracted_data