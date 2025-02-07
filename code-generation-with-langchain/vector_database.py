import chromadb
import pandas as pd
from langchain.schema import Document
from typing import Any, Dict, List

class VectorDatabase:
    """
    Class to create and manage a vector database collection using ChromaDB.
    It provides functions to upsert documents (with embeddings) and to query the collection
    using a retriever that returns similar documents.
    """

    def __init__(self, collection_name: str = "my_collection") -> None:
        """
        Initializes the ChromaDB client and retrieves (or creates) a collection.

        :param collection_name: Name of the collection to be used.
        """
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    def upsert_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> None:
        """
        Upserts documents into the collection. Each document should have a corresponding id,
        metadata, and embedding vector.

        :param ids: List of document IDs.
        :param documents: List of document contents (e.g., code).
        :param metadatas: List of metadata dictionaries for each document.
        :param embeddings: List of embeddings corresponding to each document.
        """
        # Print out each document's details before upserting
        for i in range(len(ids)):
            print(f"ID: {ids[i]}")
            print(f"Document: {documents[i]}")
            print(f"Metadata: {metadatas[i]}")
            print(f"Embedding: {embeddings[i]}\n")

        self.collection.upsert(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings
        )

        print("Documents added successfully!!")
        document_count = self.collection.count()
        print(f"Total documents in the collection: {document_count}")

    def upsert_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Extracts the required columns from a Pandas DataFrame and upserts the documents.

        The DataFrame is expected to have the following columns:
          - "ids": Document IDs.
          - "code": Document contents.
          - "metadatas": Metadata associated with each document.
          - "embeddings": Embedding vectors.

        :param df: Pandas DataFrame containing the documents and their embeddings.
        """
        # Extracting the required parts from the DataFrame
        ids = df["ids"].tolist()
        documents = df["code"].tolist()
        metadatas = df["metadatas"].tolist()
        embeddings_list = df["embeddings"].tolist()

        data_to_insert = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "embeddings": embeddings_list
        }
        print("Data extracted from DataFrame:")
        print(data_to_insert)

        # Upsert the extracted data into the collection
        self.upsert_documents(ids, documents, metadatas, embeddings_list)

    def query_retriever(self, query_text: str, top_n: int = 10) -> List[Document]:
        """
        Queries the collection for the top_n documents most similar to the given query text.
        Converts the query result into a list of langchain Document objects.

        :param query_text: The query text.
        :param top_n: Number of top results to return.
        :return: List of Documents from the query result.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_n
        )

        # Convert each result into a Document object
        documents = [
            Document(
                page_content=str(results['documents'][i]),
                metadata=results['metadatas'][i]
                if isinstance(results['metadatas'][i], dict)
                else results['metadatas'][i][0]
            )
            for i in range(len(results['documents']))
        ]
        return documents

