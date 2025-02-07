import os
import re
import requests
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from langchain.document_loaders import PyMuPDFLoader

class ArxivPaperFetcher:
    """
    Class to search arXiv for papers based on a query, download the corresponding PDF,
    extract text from the PDF, and return the paper's title and text.

    Methods:
    - fetch_papers(query: str, max_results: int = 1) -> List[Dict[str, str]]:
        Searches arXiv using the query and returns a list of dictionaries with the paper's title and text.
    """

    ARXIV_API_URL = "http://export.arxiv.org/api/query"

    def __init__(self) -> None:
        """Initialize the ArxivPaperFetcher."""
        pass  

    def _download_pdf(self, pdf_url: str, output_path: str) -> bool:
        """
        Downloads a PDF from the provided URL and saves it to the specified output path.

        :param pdf_url: URL of the PDF to download.
        :param output_path: Local file path where the PDF will be saved.
        :return: True if the download was successful; otherwise, False.
        """
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"Error downloading PDF: {response.status_code}")
            return False

    def fetch_papers(self, query: str, max_results: int = 1) -> List[Dict[str, str]]:
        """
        Searches arXiv for papers based on the query and extracts text from the PDF of each article.

        :param query: The search term for arXiv.
        :param max_results: Maximum number of results to return.
        :return: A list of dictionaries, each containing 'title' and 'text' keys.
        """
        url = f"{self.ARXIV_API_URL}?search_query=all:{query}&start=0&max_results={max_results}"
        response = requests.get(url)
        papers: List[Dict[str, str]] = []

        if response.status_code == 200:
            root = ET.fromstring(response.content)
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title_element = entry.find('{http://www.w3.org/2005/Atom}title')
                if title_element is None:
                    continue
                title = title_element.text or ""
                title = title.strip()
                pdf_url: Optional[str] = None

                for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
                    if link.attrib.get('type') == 'application/pdf':
                        pdf_url = link.attrib.get('href')
                        break

                if pdf_url:
                    safe_title = re.sub(r'\W+', '_', title)[:50]
                    pdf_path = f"temp_{safe_title}.pdf"
                    pdf_downloaded = self._download_pdf(pdf_url, pdf_path)

                    if pdf_downloaded:
                        try:
                            loader = PyMuPDFLoader(pdf_path)
                            docs = loader.load()
                            text = "\n".join([doc.page_content for doc in docs])
                            
                            papers.append({
                                'title': title,
                                'text': text
                            })
                            print(f"Text extracted from the article '{title}':\n{text[:500]}...")
                        except Exception as e:
                            print(f"Error extracting text from PDF '{title}': {str(e)}")
                    else:
                        print(f"Error downloading article PDF '{title}'.")
                else:
                    arxiv_url = entry.find('{http://www.w3.org/2005/Atom}id').text or ""
                    print(f"No PDF link found for the article '{title}'. You can view it online here: {arxiv_url}")
        else:
            print("Error accessing arXiv.")

        return papers

# -----------------------------------------------------------------------------
# Example Usage:
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Instantiate the ArxivPaperFetcher.
    fetcher = ArxivPaperFetcher()
    
    # Define the search query and maximum results.
    search_query = "machine learning"
    max_results = 2
    
    # Fetch papers from arXiv.
    papers = fetcher.fetch_papers(search_query, max_results)
    
    # Display the retrieved papers.
    for paper in papers:
        print(f"\nTitle: {paper['title']}")
        print(f"Extracted Text (first 500 chars):\n{paper['text'][:500]}...\n")
