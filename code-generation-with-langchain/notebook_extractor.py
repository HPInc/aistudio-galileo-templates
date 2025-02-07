import os
import requests
import shutil
import nbformat
import uuid
from typing import Any, Dict, List, Optional

try:
    import git
except ImportError:
    git = None


class NotebookExtractor:
    """
    Class for extracting information (code and context) from notebooks (.ipynb)
    in a GitHub repository. It offers two extraction methods:
    - Via GitHub API (direct file download);
    - Via repository cloning (using GitPython).

    The result is a list of dictionaries, where each dictionary contains:
    - id: Unique identifier for the cell;
    - embedding: Placeholder for embedding (None);
    - code: Cell code (if the cell type is 'code');
    - filename: Notebook file name;
    - context: Content of the previous markdown cell(s).
    """

    
    def __init__(self, save_dir: str = "./notebooks") -> None:
        """
        :param save_dir: Directory where the notebooks will be saved/cloned.

        """
        self.save_dir = save_dir

    def download_file(self, file_url: str, save_path: str) -> None:
        """ 
        Downloads a file from a given URL and saves it to the specified path.
        :param file_url: URL of the file.
        :param save_path: Local path to save the file.

        """
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(response.content)
            print(f"Downloaded: {save_path}")
        else:
            print(f"Failed to download {file_url}, Status Code: {response.status_code}")

    def extract_code_and_context(self, notebook_path: str) -> List[Dict[str, Any]]:
        """   
        Reads a notebook and extracts, for each code cell, its content and context (previous markdown).

        :param notebook_path: Path to the notebook (.ipynb).
        :return: List of dictionaries containing the extracted data.

        """
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)
    
        extracted_data: List[Dict[str, Any]] = []
        context: str = ""
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "markdown":
                context = "".join(cell.get("source", ""))
            elif cell.get("cell_type") == "code":
                cell_data = {
                    "id": str(uuid.uuid4()),
                    "embedding": None,
                    "code": "".join(cell.get("source", "")),
                    "filename": os.path.basename(notebook_path),
                    "context": context,
                }
                extracted_data.append(cell_data)
        return extracted_data

    def download_notebooks_from_repo_dir(
        self,
        repo_owner: str,
        repo_name: str,
        dir_path: str,
        all_extracted_data: List[Dict[str, Any]]
    ) -> None:
        """
        Recursive function to traverse a repository directory via API and extract notebooks.

        :param repo_owner: Repository owner.
        :param repo_name: Repository name.
        :param dir_path: Relative path of the directory in the repository.
        :param all_extracted_data: List to accumulate the extracted data.

        """
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{dir_path}"
        response = requests.get(api_url)
        if response.status_code != 200:
            print(f"Error fetching directory contents: {response.status_code}")
            return

        repo_contents = response.json()
        for item in repo_contents:
            if item.get("type") == "file" and item.get("name", "").endswith(".ipynb"):
                notebook_path = os.path.join(self.save_dir, os.path.basename(item["path"]))
                self.download_file(item["download_url"], notebook_path)
                extracted = self.extract_code_and_context(notebook_path)
                all_extracted_data.extend(extracted)
            elif item.get("type") == "dir":
                self.download_notebooks_from_repo_dir(repo_owner, repo_name, item["path"], all_extracted_data)

    def extract_from_api(self, repo_owner: str, repo_name: str) -> List[Dict[str, Any]]:
        """
        Extracts notebooks using the public GitHub API.

        :param repo_owner: Repository owner.
        :param repo_name: Repository name.
        :return: List containing the extracted data.

        """
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
            print(f"Existing directory {self.save_dir} removed.")
        os.makedirs(self.save_dir, exist_ok=True)

        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents"
        response = requests.get(api_url)
        if response.status_code != 200:
            print(f"Error fetching repository contents: {response.status_code}")
            return []

        repo_contents = response.json()
        all_extracted_data: List[Dict[str, Any]] = []
        for item in repo_contents:
            if item.get("type") == "file" and item.get("name", "").endswith(".ipynb"):
                notebook_path = os.path.join(self.save_dir, item["name"])
                self.download_file(item["download_url"], notebook_path)
                extracted = self.extract_code_and_context(notebook_path)
                all_extracted_data.extend(extracted)
            elif item.get("type") == "dir":
                self.download_notebooks_from_repo_dir(repo_owner, repo_name, item["path"], all_extracted_data)
        return all_extracted_data

    def extract_from_clone(self, repo_url: str) -> List[Dict[str, Any]]:
        """
        Clones the repository and extracts notebooks locally.

        Requires the GitPython library.

        :param repo_url: Repository URL.
        :return: List containing the extracted data.

        """
        if git is None:
            raise ImportError("GitPython não está instalada.")

        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
            print(f"Existing directory {self.save_dir} removed.")
        git.Repo.clone_from(repo_url, self.save_dir)
        print(f"Repository cloned into: {self.save_dir}")

        all_extracted_data: List[Dict[str, Any]] = []
        for root, _, files in os.walk(self.save_dir):
            for file in files:
                if file.endswith(".ipynb"):
                    notebook_path = os.path.join(root, file)
                    print(f"Extracting data from: {notebook_path}")
                    extracted = self.extract_code_and_context(notebook_path)
                    all_extracted_data.extend(extracted)
        print(f"Extraction completed. Total cells processed: {len(all_extracted_data)}")
        return all_extracted_data

    def extract(
        self,
        repo_owner: Optional[str] = None,
        repo_name: Optional[str] = None,
        repo_url: Optional[str] = None,
        method: str = "api"
    ) -> List[Dict[str, Any]]:
        """
        Processes the repository to extract notebooks using the chosen method.

        :param repo_owner: Repository owner (required for the "api" method).
        :param repo_name: Repository name (required for the "api" method).
        :param repo_url: Repository URL (required for the "clone" method).
        :param method: "api" to use the API or "clone" to clone the repository.
        :return: List containing the extracted data.

        """
        if method.lower() == "api":
            if not repo_owner or not repo_name:
                raise ValueError("For the 'api' method, please provide repo_owner and repo_name.")
            return self.extract_from_api(repo_owner, repo_name)
        elif method.lower() == "clone":
            if not repo_url:
                raise ValueError("For the 'clone' method, please provide repo_url.")
            return self.extract_from_clone(repo_url)
        else:
            raise ValueError("Invalid method. Use 'api' or 'clone'.")


