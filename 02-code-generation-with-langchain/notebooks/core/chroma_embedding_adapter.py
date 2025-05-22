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
        if not input:
            logger.warning("Empty input provided to ChromaEmbeddingAdapter. Returning empty list.")
            return []
            
        # Create a fallback embedding of correct dimension (all zeros)
        default_embedding_dim = 384  # Default dimension for all-MiniLM-L6-v2
        default_embedding = [0.0] * default_embedding_dim
        
        try:
            # Call the underlying embedding model's embed_documents method
            embeddings = self.embedding_model.embed_documents(input)
            
            # Validate the returned embeddings
            if embeddings is None or len(embeddings) == 0:
                logger.warning("Embedding model returned None or empty list. Using zeros.")
                return [default_embedding] * len(input)
                
            # Check for None values in any of the embeddings
            valid_embeddings = []
            for i, emb in enumerate(embeddings):
                if emb is None or len(emb) == 0 or any(val is None for val in emb):
                    logger.warning(f"Invalid embedding at index {i}. Using zeros.")
                    valid_embeddings.append(default_embedding)
                else:
                    valid_embeddings.append(emb)
                    
            return valid_embeddings
        except Exception as e:
            logger.error(f"Error in ChromaEmbeddingAdapter.__call__: {str(e)}")
            # Return default embeddings instead of raising
            logger.warning(f"Returning default embeddings for {len(input)} inputs")
            return [default_embedding] * len(input)