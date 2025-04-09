import logging
import chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreWriter:
    def __init__(self, collection_name="my_collection", verbose=False):
        """
        Initializes a ChromaDB collection.

        :param collection_name: Name of the collection to upsert data into
        :param verbose: If True, log detailed info for each record
        """
        self.verbose = verbose
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(f"ChromaDB collection '{collection_name}' initialized.")

    def upsert_dataframe(self, df):
        """
        Extracts data from a DataFrame and upserts into the vector store.

        :param df: DataFrame with 'ids', 'code', 'metadatas', 'embeddings' columns
        """
        ids = df["ids"].tolist()
        documents = df["code"].tolist()
        metadatas = df["metadatas"].tolist()
        embeddings = df["embeddings"].tolist()

        if self.verbose:
            for i in range(len(ids)):
                logger.info(f"[UPDATING] ID: {ids[i]}")
                logger.info(f"[UPDATING] Document: {documents[i]}")
                logger.info(f"[UPDATING] Metadata: {metadatas[i]}")
                logger.info(f"[UPDATING] Embedding: {embeddings[i]}\n")

        self.collection.upsert(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings
        )

        logger.info("✅ Documents upserted successfully into ChromaDB.")
