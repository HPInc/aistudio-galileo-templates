import pandas as pd
from typing import Any, Dict, List
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingsProcessor:
    """
    Class to load an embeddings model and update a data structure with generated embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the embeddings model.

        :param model_name: The name of the HuggingFace embeddings model.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def update_embeddings(self, data_structure: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update the given data structure with embeddings generated for the 'context' field.

        For each item in the data structure, the method generates an embedding for the context
        and updates the item with a new key 'embedding'.

        :param data_structure: List of dictionaries, each expected to contain a 'context' key.
        :return: Updated data structure with embeddings.
        """
        updated_structure: List[Dict[str, Any]] = []
        for item in data_structure:
            context: str = item.get('context', '')
            embedding_vector = self.embeddings.embed_query(context)
            item['embedding'] = embedding_vector
            updated_structure.append(item)
        return updated_structure


class DataFrameConverter:
    """
    Class to convert embedded snippets into a format suitable for DataFrame creation.
    """

    @staticmethod
    def to_dataframe_rows(embedded_snippets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert a list of embedded snippets into a list of dictionaries suitable for DataFrame rows.

        Each dictionary will contain:
          - "ids": The unique identifier of the snippet.
          - "embeddings": The generated embedding vector.
          - "code": The code snippet.
          - "metadatas": A dictionary with additional metadata (e.g., filename and context).

        :param embedded_snippets: List of dictionaries containing the embedded snippets.
        :return: List of dictionaries formatted for DataFrame creation.
        """
        outputs: List[Dict[str, Any]] = []
        for snippet in embedded_snippets:
            output = {
                "ids": snippet.get('id'),
                "embeddings": snippet.get('embedding'),
                "code": snippet.get('code'),
                "metadatas": {
                    "filenames": snippet.get('filename'),
                    "context": snippet.get('context')
                }
            }
            outputs.append(output)
        return outputs


