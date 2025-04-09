import pandas as pd
from tqdm import tqdm

class EmbeddingUpdater:
    def __init__(self, embedding_model, verbose=False):
        """
        Initialize with an external embedding model instance.

        :param embedding_model: Object with method embed_query(text)
        :param verbose: If True, will print logs or show progress
        """
        self.embedding_model = embedding_model
        self.verbose = verbose

    def update(self, data_structure):
        """
        Update each item's 'embedding' field using the given embedding model.

        :param data_structure: List of dictionaries with a 'context' field
        :return: List with embeddings included
        """
        updated_structure = []
        iterator = tqdm(data_structure, desc="Generating Embeddings") if self.verbose else data_structure

        for item in iterator:
            context = item.get("context", "")
            embedding_vector = self.embedding_model.embed_query(context)
            item["embedding"] = embedding_vector
            updated_structure.append(item)

            if self.verbose and not isinstance(iterator, tqdm):
                print(f"[LOG] Embedding generated for ID {item.get('id', 'unknown')}")

        return updated_structure


class DataFrameConverter:
    def __init__(self, verbose=False):
        """
        Initialize the DataFrame converter.

        :param verbose: If True, will print logs during DataFrame conversion
        """
        self.verbose = verbose

    def to_dataframe(self, embedded_snippets):
        """
        Convert embedded snippets into a DataFrame format.

        :param embedded_snippets: List of dicts with embedding and metadata
        :return: Pandas DataFrame
        """
        outputs = []
        for snippet in embedded_snippets:
            row = {
                "ids": snippet.get("id"),
                "embeddings": snippet.get("embedding"),
                "code": snippet.get("code"),
                "metadatas": {
                    "filenames": snippet.get("filename"),
                    "context": snippet.get("context"),
                },
            }
            outputs.append(row)

        df = pd.DataFrame(outputs)

        if self.verbose:
            print("[LOG] DataFrame created with", len(df), "rows")

        return df

    def print_contexts(self, df):
        """
        Display the 'context' field from the 'metadatas' column.

        :param df: DataFrame with 'metadatas' column
        """
        contexts = df["metadatas"].apply(lambda x: x.get("context", None))
        print("[LOG] Contexts in DataFrame:")
        print(contexts)
