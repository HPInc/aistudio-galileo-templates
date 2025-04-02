import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingUpdater:
    def __init__(self, model_name="all-MiniLM-L6-v2", verbose=False):
        """
        Initialize the embedding generator using a HuggingFace embedding model.

        :param model_name: Name of the Hugging Face model to use for embeddings
        :param verbose: If True, will print logs during embedding updates
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.verbose = verbose

    def update(self, data_structure):
        """
        Update each item's 'embedding' field using the LLM embeddings for context.

        :param data_structure: List of dictionaries with a 'context' field
        :return: Updated structure with embeddings
        """
        updated_structure = []

        for item in data_structure:
            context = item.get('context', "")
            embedding_vector = self.embeddings.embed_query(context)
            item['embedding'] = embedding_vector
            updated_structure.append(item)

            if self.verbose:
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
                "ids": snippet['id'],
                "embeddings": snippet['embedding'],
                "code": snippet['code'],
                "metadatas": {
                    "filenames": snippet['filename'],
                    "context": snippet['context'],
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
        contexts = df['metadatas'].apply(lambda x: x.get('context', None))
        print("[LOG] Contexts in DataFrame:")
        print(contexts)


