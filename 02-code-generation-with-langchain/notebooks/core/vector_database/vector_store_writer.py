import logging
import chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreWriter:
    def __init__(self, collection_name="my_collection", verbose=False, embedding_model=None):
        """
        Initializes a ChromaDB collection.

        :param collection_name: Name of the collection to upsert data into
        :param verbose: If True, log detailed info for each record
        :param embedding_model: Optional embedding model to use with ChromaDB (if supported)
        """
        self.verbose = verbose
        self.embedding_model = embedding_model
        
        # Use PersistentClient for consistency with other code
        persist_dir = "./chroma_db"
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Try to use embedding_model if provided and supported by ChromaDB
        collection_kwargs = {"name": collection_name}
        # Don't pass the embedding_function to avoid ChromaDB interface issues
        
        try:
            self.collection = self.client.get_or_create_collection(**collection_kwargs)
            logger.info(f"ChromaDB collection '{collection_name}' initialized with persistent storage.")
        except TypeError as e:
            logger.warning(f"Error creating collection: {str(e)}")
            # Fallback for older ChromaDB versions
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logger.info(f"ChromaDB collection '{collection_name}' initialized with fallback method.")

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
