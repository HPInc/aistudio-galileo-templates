import logging
from typing import List, Any

# Try different import paths based on what's available in the project
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
    except ImportError:
        pass  # The import will be used for type hints only, not required at runtime

logger = logging.getLogger(__name__)

class ChromaEmbeddingAdapter:
    """
    Adapter class to make HuggingFaceEmbeddings compatible with ChromaDB's expected interface.
    
    ChromaDB expects an embedding function with this signature:
    __call__(self, input: List[str]) -> List[List[float]]
    
    While HuggingFaceEmbeddings provides:
    embed_documents(texts: List[str]) -> List[List[float]]
    
    This adapter bridges the gap between these interfaces.
    """
    
    def __init__(self, embedding_model):
        """
        Initialize with a HuggingFaceEmbeddings instance or similar model
        that provides the embed_documents method
        
        Args:
            embedding_model: A model instance with embed_documents method
        """
        self.embedding_model = embedding_model
        logger.info(f"Initialized ChromaEmbeddingAdapter with model: {type(embedding_model).__name__}")
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Make the adapter callable with the signature expected by ChromaDB
        
        Args:
            input: List of strings to embed
            
        Returns:
            List of embedding vectors (as lists of floats)
        """
        try:
            # Call the underlying embedding model's embed_documents method
            embeddings = self.embedding_model.embed_documents(input)
            return embeddings
        except Exception as e:
            logger.error(f"Error in ChromaEmbeddingAdapter.__call__: {str(e)}")
            raise