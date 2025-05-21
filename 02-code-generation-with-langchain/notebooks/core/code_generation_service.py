"""
Code Generation Service implementation that extends the BaseGenerativeService.

This service provides code generation capabilities using LLM models with vector retrieval
and integrates with Galileo for protection, observation, and evaluation.
It can extract context from GitHub repositories to enhance code generation responses.
"""

import os
import sys
import logging
import traceback
import time
import json
import datetime
import numpy as np
from typing import Dict, Any, List, Optional
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser, Document
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from galileo_protect import ProtectParser
import chromadb

# Add the src directory to the path to import base_service
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.service.base_service import BaseGenerativeService
from src.utils import get_context_window, dynamic_retriever, format_docs_with_adaptive_context, clean_code

# Add core directory to path for local imports
core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if core_path not in sys.path:
    sys.path.append(core_path)

# Import GitHub repository extraction tools
from core.extract_text.github_repository_extractor import GitHubRepositoryExtractor
from core.generate_metadata.llm_context_updater import LLMContextUpdater
from core.dataflow.dataflow import EmbeddingUpdater, DataFrameConverter
from core.vector_database.vector_store_writer import VectorStoreWriter

# Set up logger
logger = logging.getLogger(__name__)

class CodeGenerationService(BaseGenerativeService):
    """
    Code Generation Service that extends the BaseGenerativeService.
    Supports both direct code generation questions and
    context retrieval from specified GitHub repositories.
    """

    def __init__(self):
        """Initialize the code generation service.
        
        IMPORTANT: The embedding initialization order is critical - embeddings must be
        initialized before any LLM model to prevent CUDA library loading issues
        that may occur when initializing LlamaCpp models.
        
        To avoid downloading the default embedding model unnecessarily, the actual
        embedding initialization is deferred until load_context is called, which
        will check for an artifact model first. If rapid initialization is needed
        before load_context is called, initialize_embedding_function can be called manually.
        """
        super().__init__()
        self.vector_store = None
        self.retriever = None
        self.collection = None
        self.collection_name = "my_collection"
        self.embedding_path = None
        self.context_window = None
        
        # Repository cache to avoid re-processing the same repositories
        self.repository_cache = {}
        
        # The embedding_function will be initialized in load_context
        # or can be manually initialized by calling initialize_embedding_function
        self.embedding_function = None
        
        # Set default processing parameters
        self.default_batch_size = 20
        self.default_timeout = 300  # 5 minutes
        
        # Configure logging to reduce noise
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        
        # Set logging format for better readability
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                formatter = logging.Formatter('[%(levelname)s] %(message)s')
                handler.setFormatter(formatter)
    
    def initialize_embedding_function(self, embedding_model_path=None):
        """Initialize the embedding function.
        
        Args:
            embedding_model_path: Path to a locally saved embedding model (optional)
            
        Returns:
            An initialized HuggingFaceEmbeddings object
        """
        logger.info("Initializing embedding function")
        
        # Import HuggingFaceEmbeddings
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            # Fall back to older import path
            from langchain.embeddings import HuggingFaceEmbeddings
        
        # Determine which model path to use
        model_name = embedding_model_path if embedding_model_path else "all-MiniLM-L6-v2"
        if embedding_model_path:
            logger.info(f"Using provided embedding model path: {embedding_model_path}")
        else:
            logger.info("Using default embedding model: all-MiniLM-L6-v2")
        
        # Initialize the embedding function
        self.embedding_function = HuggingFaceEmbeddings(model_name=model_name)
        logger.info(f"Successfully initialized embedding function with model: {model_name}")
        
        return self.embedding_function
    
    def extract_repository(self, repository_url: str, metadata_only: bool = False, 
                          batch_size: int = 20, timeout: int = 300) -> List[Dict[str, Any]]:
        """
        Extract code and metadata from a GitHub repository.
        Uses a cache mechanism to avoid re-processing the same repository.
        
        Args:
            repository_url: URL of the GitHub repository
            metadata_only: If True, only perform fast metadata extraction without LLM processing
            batch_size: Number of files to process in each batch
            timeout: Maximum time in seconds for the entire operation
            
        Returns:
            List of dictionaries containing extracted code and metadata
        """
        import concurrent.futures
        from functools import partial
        
        start_time = time.time()
        
        try:
            self.reset_repository_state(repository_url)
            
            # Check if repository is already in cache
            if repository_url in self.repository_cache:
                logger.info(f"Using cached data for repository: {repository_url}")
                
                # Update the collection reference from the cache
                self.collection = self.repository_cache[repository_url]["collection"]
                metadata_status = self.repository_cache[repository_url].get("metadata_only", False)
                
                # If we want full processing but the cache only has metadata, we need to process further
                if metadata_only or not metadata_status:
                    try:
                        count = self.collection.count()
                        logger.info(f"Cache hit: Collection has {count} documents")
                        return self.repository_cache[repository_url]["data"]
                    except Exception as e:
                        logger.warning(f"Cached collection error: {str(e)}. Re-processing repository.")
                        # Continue with fresh processing if we can't access the cached collection
            
            logger.info(f"Extracting code from repository: {repository_url} (metadata_only={metadata_only})")
            
            # Step 1: Clone repository and extract files
            extractor = GitHubRepositoryExtractor(
                repo_url=repository_url,
                save_dir="./repo_files",
                verbose=False,
                max_file_size_kb=500,
                max_chunk_size=100,
                supported_extensions=('.py', '.ipynb', '.md', '.txt', '.json', '.js', '.ts')
            )
            extracted_data = extractor.run()
            file_count = len(extracted_data)
            logger.info(f"Extracted {file_count} code snippets from repository")
            
            # Check timeout after extraction
            if (time.time() - start_time) > timeout:
                logger.warning("Repository extraction timeout reached. Returning partial results.")
                return self._store_partial_results(repository_url, extracted_data, metadata_only=True)
            
            # For metadata_only mode, skip LLM processing and just add basic metadata
            if metadata_only:
                logger.info("Metadata-only mode: Skipping LLM context generation")
                # Add basic metadata to each file
                for item in extracted_data:
                    filename = item.get('filename', 'unknown_file')
                    # Generate basic metadata based on file extension and size
                    extension = os.path.splitext(filename)[1] if '.' in filename else ''
                    code_size = len(item.get('code', ''))
                    item['context'] = f"File: {filename} ({extension} file, {code_size} characters)"
                    
                # Skip to embeddings generation
                updated_data = extracted_data
            else:
                # Step 2: Use LLM to generate metadata for each code snippet with batching
                from langchain_core.prompts import PromptTemplate
                template = """
                You will receive three pieces of information: a code snippet, a file name, and an optional context. Based on this information, explain in a clear, summarized and concise way what the code snippet is doing.

                Code:
                {code}

                File name:
                {filename}

                Context:
                {context}

                Describe what the code above does.
                """
                
                prompt = PromptTemplate.from_template(template)
                
                # Only use the LLM if it's been initialized
                if self.llm:
                    logger.info("Using existing LLM model for metadata generation")
                    llm_chain = prompt | self.llm
                else:
                    logger.warning("LLM not initialized, skipping metadata generation")
                    return self._store_partial_results(repository_url, extracted_data, metadata_only=True)
                
                # Process files in batches to avoid memory issues
                updated_data = []
                total_batches = (file_count + batch_size - 1) // batch_size  # ceil division
                
                for batch_idx in range(total_batches):
                    # Check timeout before each batch
                    if (time.time() - start_time) > timeout:
                        logger.warning(f"Timeout reached after processing {batch_idx}/{total_batches} batches")
                        # Store what we have so far
                        if updated_data:
                            updated_data.extend(extracted_data[len(updated_data):])
                        else:
                            updated_data = extracted_data
                        return self._store_partial_results(repository_url, updated_data, metadata_only=True)
                    
                    batch_start = batch_idx * batch_size
                    batch_end = min((batch_idx + 1) * batch_size, file_count)
                    current_batch = extracted_data[batch_start:batch_end]
                    
                    logger.info(f"Processing batch {batch_idx+1}/{total_batches} ({len(current_batch)} files)")
                    
                    try:
                        # Set a batch timeout that's a fraction of the total timeout
                        batch_timeout = min(30, timeout // total_batches)
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            batch_future = executor.submit(self._process_metadata_batch, 
                                                          current_batch, llm_chain, template)
                            batch_result = batch_future.result(timeout=batch_timeout)
                            updated_data.extend(batch_result)
                    except concurrent.futures.TimeoutError:
                        logger.warning(f"Batch {batch_idx+1} processing timed out after {batch_timeout}s")
                        # Add the unprocessed items from this batch with basic metadata
                        for item in current_batch:
                            filename = item.get('filename', 'unknown_file')
                            item['context'] = f"File: {filename} (metadata generation timed out)"
                            updated_data.append(item)
                    except Exception as batch_error:
                        logger.error(f"Error processing batch {batch_idx+1}: {str(batch_error)}")
                        # Add the unprocessed items from this batch with basic metadata
                        for item in current_batch:
                            filename = item.get('filename', 'unknown_file')
                            item['context'] = f"File: {filename} (metadata generation failed)"
                            updated_data.append(item)
            
            # Check timeout again before embedding generation
            if (time.time() - start_time) > timeout * 0.9:  # 90% of timeout
                logger.warning("Approaching timeout limit. Skipping embedding generation.")
                return self._store_partial_results(repository_url, updated_data, metadata_only=True)
            
            # Step 3: Generate embeddings for each code snippet
            try:
                embedding_updater = EmbeddingUpdater(embedding_model=self.embedding_function, verbose=False)
                updated_data = embedding_updater.update(updated_data)
                logger.info("Embeddings generated successfully")
            except Exception as e:
                logger.error(f"Error during embedding generation: {str(e)}")
                # Continue with the process, just log the error
            
            # Step 4: Convert to DataFrame for easier processing
            try:
                converter = DataFrameConverter(verbose=False)
                df = converter.to_dataframe(updated_data)
                logger.info("Data conversion to DataFrame complete")
            except Exception as e:
                logger.error(f"Error during DataFrame conversion: {str(e)}")
                # Create a basic DataFrame with just the essential fields
                import pandas as pd
                basic_data = []
                for item in updated_data:
                    basic_data.append({
                        'id': item.get('id', f"id_{len(basic_data)}"),
                        'code': item.get('code', ''),
                        'filename': item.get('filename', 'unknown'),
                        'context': item.get('context', ''),
                        'embedding': item.get('embedding', [])
                    })
                df = pd.DataFrame(basic_data)
                logger.info("Created basic DataFrame with essential fields")
            
            # Step 5: Store in vector database
            try:
                writer = VectorStoreWriter(collection_name=self.collection_name, verbose=False)
                writer.upsert_dataframe(df)
                logger.info(f"Repository data stored in collection: {self.collection_name}")
                
                # Update the collection reference
                self.collection = writer.collection
            except Exception as e:
                logger.error(f"Error storing data in vector database: {str(e)}")
                # If we can't store in the database, return what we have without caching
                return updated_data
            
            # Store in cache for future use
            self.repository_cache[repository_url] = {
                "data": updated_data,
                "collection": self.collection,
                "timestamp": datetime.datetime.now().isoformat(),
                "metadata_only": metadata_only
            }
            logger.info(f"Repository {repository_url} added to cache (metadata_only={metadata_only})")
            
            return updated_data
        except Exception as e:
            logger.error(f"Error extracting repository data: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty list instead of raising, to allow graceful degradation
            return []
    
    def _process_metadata_batch(self, batch, llm_chain, template):
        """Process a batch of files to generate metadata."""
        try:
            updater = LLMContextUpdater(
                llm_chain=llm_chain,
                prompt_template=template,
                verbose=False,
                print_prompt=False
            )
            return updater.update(batch)
        except Exception as e:
            logger.error(f"Error in batch metadata processing: {str(e)}")
            # Return batch with basic context information for resilience
            for item in batch:
                if 'context' not in item or not item['context']:
                    filename = item.get('filename', 'unknown_file')
                    item['context'] = f"File: {filename} (metadata generation failed)"
            return batch
    
    def _store_partial_results(self, repository_url, data, metadata_only=True):
        """Store partial results in cache and vector database."""
        try:
            # Add basic embeddings if missing
            for item in data:
                if 'embedding' not in item or not item['embedding']:
                    # Create a small random vector as placeholder
                    item['embedding'] = np.random.rand(384).tolist()  # 384 is the dimension for all-MiniLM-L6-v2
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'id': item.get('id', f"id_{i}"),
                'code': item.get('code', ''),
                'filename': item.get('filename', 'unknown'),
                'context': item.get('context', ''),
                'embedding': item.get('embedding', [])
            } for i, item in enumerate(data)])
            
            # Store in vector database
            writer = VectorStoreWriter(collection_name=self.collection_name, verbose=False)
            writer.upsert_dataframe(df)
            
            # Update the collection reference and cache
            self.collection = writer.collection
            
            # Store in cache for future use
            self.repository_cache[repository_url] = {
                "data": data,
                "collection": self.collection,
                "timestamp": datetime.datetime.now().isoformat(),
                "metadata_only": metadata_only
            }
            logger.info(f"Stored partial results for {repository_url} (metadata_only={metadata_only})")
            
            return data
        except Exception as e:
            logger.error(f"Error storing partial results: {str(e)}")
            return data
    
    def custom_retriever(self, query: str, top_n: int = None) -> List[Document]:
        """
        Custom retriever function 
        
        Args:
            query: The query string for retrieval
            top_n: Number of documents to retrieve (if None, determined by context window)
            
        Returns:
            List of Document objects with content and metadata
        """
        
        # Determine whether to use the vector_store or collection
        retrieval_source = None
        if self.vector_store:
            logger.info("Using vector_store for retrieval")
            retrieval_source = self.vector_store
        elif self.collection:
            logger.info("Using direct collection for retrieval")
            retrieval_source = self.collection
        else:
            logger.error("No retrieval source available (neither vector_store nor collection)")
            return []
        
        try:
            # Use class-level context window if available, or get from model
            context_window = None
            if hasattr(self, 'context_window') and self.context_window:
                context_window = self.context_window
                logger.info(f"Using stored context window: {context_window} tokens")
            elif hasattr(self, 'llm'):
                context_window = get_context_window(self.llm)
                logger.info(f"Retrieved model context window: {context_window} tokens")
            
            documents = dynamic_retriever(
                query=query, 
                collection=retrieval_source, 
                top_n=top_n, 
                context_window=context_window
            )
            
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def load_vector_store(self, persist_directory="./chroma_db"):
        """
        Load or create a vector store for code retrieval.
        
        Args:
            persist_directory: Directory to store vector database
        """
        try:
            logger.info(f"Loading vector store from {persist_directory}")
            
            # Initialize chromadb client
            client = chromadb.PersistentClient(path=persist_directory)
            
            # Try to get existing collection or create a new one
            try:
                self.collection = client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Collection '{self.collection_name}' loaded/created successfully")
            except Exception as col_err:
                logger.error(f"Error getting/creating collection: {str(col_err)}")
                logger.error(f"Exception type: {type(col_err).__name__}")
            
            # Initialize LangChain vector store with the embedding function
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embedding_function,
                collection_name=self.collection_name
            )
            self.retriever = self.vector_store.as_retriever()
            logger.info(f"Vector store successfully loaded from {persist_directory}")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.info("Creating new empty vector store")
            # Use the embedding function that was already initialized in __init__
            self.vector_store = Chroma(embedding_function=self.embedding_function)
            self.retriever = self.vector_store.as_retriever()
            logger.info("Created new empty vector store")
    
    def load_model(self, context) -> None:
        """
        Load the appropriate model based on configuration.
        
        Args:
            context: MLflow model context containing artifacts
        """
        try:
            # Ensure embedding function is initialized before loading LLM (critical for CUDA library loading)
            logger.info("Ensuring embedding model is initialized before loading LLM")
            if hasattr(self, 'embedding_function') and self.embedding_function is not None:
                logger.info("Using existing embedding function instance")
            else:
                logger.warning("Embedding function not initialized yet, initializing now")
                # Try to use embedding model path from artifacts if available
                embedding_model_path = None
                if "embedding_model" in context.artifacts:
                    embedding_model_path = context.artifacts["embedding_model"]
                    if os.path.exists(embedding_model_path):
                        logger.info(f"Using embedding model from artifacts: {embedding_model_path}")
                
                # Initialize embedding function
                self.initialize_embedding_function(embedding_model_path)
            
            # Now proceed with loading the LLM model
            model_source = self.model_config.get("model_source", "local")
            logger.info(f"Attempting to load model from source: {model_source}")
            
            if model_source == "local":
                self.load_local_model(context)
            else:
                logger.info(f"Using model source: {model_source}")
                # Import utility function for initializing LLM
                from src.utils import initialize_llm
                
                # Extract secrets from config
                secrets = {}
                if "secrets" in self.model_config:
                    secrets = self.model_config["secrets"]
                
                # Get local model path from artifacts
                local_model_path = None
                if "models" in context.artifacts:
                    local_model_path = context.artifacts["models"]
                
                # Initialize LLM using the utility function
                self.llm = initialize_llm(model_source, secrets, local_model_path)
                
            if self.llm is None:
                logger.error("Failed to initialize model from any source")
                raise ValueError("No model could be initialized")
                
            logger.info(f"Model of type {type(self.llm).__name__} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def load_local_model(self, context):
        """
        Load a local LlamaCpp model.
        
        Args:
            context: MLflow model context containing artifacts
        """
        try:
            logger.info("Initializing local LlamaCpp model.")
            model_path = context.artifacts.get("models", None)
            
            logger.info(f"Model path: {model_path}")
            
            if not model_path or not os.path.exists(model_path):
                logger.error(f"Model file not found at path: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            logger.info(f"Model file exists. Size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
            
            logger.info("Setting up callback manager")
            self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            # Determine optimal context window for this model using the utility function
            from src.utils import get_model_context_window
            
            # Create a temporary model object with model_path attribute
            temp_model = type('TempModel', (), {'model_path': model_path})
            
            # Get context window using the utility function (handles lookup in MODEL_CONTEXT_WINDOWS)
            context_window = get_model_context_window(temp_model)
            logger.info(f"Determined context window: {context_window} tokens")
            
            logger.info("Initializing LlamaCpp with the following parameters:")
            logger.info(f"  - Model Path: {model_path}")
            
            # Check CUDA availability
            cuda_available = False
            try:
                # Try to check if CUDA is available
                import subprocess
                try:
                    subprocess.check_output(['ldconfig', '-p'], stderr=subprocess.STDOUT)
                    libcuda_check = subprocess.check_output(['ldconfig', '-p', '|', 'grep', 'libcuda.so.1'], stderr=subprocess.STDOUT, shell=True)
                    if libcuda_check:
                        cuda_available = True
                except (subprocess.SubprocessError, FileNotFoundError):
                    cuda_available = False
            except Exception:
                cuda_available = False
                
            logger.info(f"CUDA availability check: {'Available' if cuda_available else 'Not available'}")
            
            # Configure GPU layers based on CUDA availability
            n_gpu_layers = 30 if cuda_available else 0
            logger.info(f"  - n_gpu_layers: {n_gpu_layers}, n_batch: 512, n_ctx: {context_window}")
            logger.info(f"  - max_tokens: 1024, f16_kv: True, temperature: 0.2")
            
            try:
                self.llm = LlamaCpp(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_batch=512,
                    n_ctx=context_window,
                    f16_kv=True,
                    callback_manager=self.callback_manager,
                    verbose=True,
                    max_tokens=1024,
                    temperature=0.2
                )
                
                self.llm.__dict__['_context_window'] = context_window
                self.context_window = context_window
                logger.info(f"Using local LlamaCpp model for code generation with {'GPU' if cuda_available else 'CPU'} mode.")
            except Exception as model_error:
                logger.error(f"Failed to initialize LlamaCpp model: {str(model_error)}")
                logger.error(f"Exception type: {type(model_error).__name__}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        except Exception as e:
            logger.error(f"Error in load_local_model: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def load_prompt(self) -> None:
        """Load the prompt templates for code generation."""
        # Template for code generation with repository context
        self.code_description_template = """You will receive three pieces of information: a code snippet, a file name, and an optional context. Based on this information, explain in a clear, summarized and concise way what the code snippet is doing.

Code:
{code}

File name:
{filename}

Context:
{context}

Describe what the code above does.
"""

        # Template for direct code generation without repository context
        self.code_generation_template = """You are a code generator AI that ONLY outputs working Python code.
NEVER ask questions or request clarification.
ALWAYS respond with complete, executable Python code.
DO NOT include any explanations, comments, or non-code text.
If you're uncertain about implementation details, make reasonable assumptions and provide working code.

Context:
{context}

Task: {question}
"""

        # Default prompt for backward compatibility with existing chain structure
        self.prompt_str = """You are a Python wizard tasked with generating code for a Jupyter Notebook (.ipynb) based on the given context.
Your answer should consist of just the Python code, without any additional text or explanation.

Context:
{context}

Question: {question}
"""
        self.prompt = ChatPromptTemplate.from_template(self.prompt_str)
        
        # Create additional prompt objects
        self.code_description_prompt = ChatPromptTemplate.from_template(self.code_description_template)
        self.code_generation_prompt = ChatPromptTemplate.from_template(self.code_generation_template)
    
    def format_docs(self, docs: List[Document], context_window: int = None) -> str:
        """
        Format a list of documents into a single string
        
        Args:
            docs: List of Document objects
            context_window: Size of the model's context window in tokens (optional)
            
        Returns:
            Formatted string of document contents optimized for the context window
        """
        # Use the utility function if context window is provided
        if context_window:
            return format_docs_with_adaptive_context(docs, context_window=context_window)
        # Fall back to simple concatenation if no context window info available
        return "\n\n".join([doc.page_content for doc in docs])
    
    def load_chain(self) -> None:
        """Create the code generation chains using the loaded model, prompts, and retriever."""
        try:
            # Load the vector store first
            logger.info("Loading vector store for retrieval")
            self.load_vector_store()
            
            # Verify retriever readiness using either direct collection or fallback to LangChain retriever
            if not self.vector_store and not self.collection and not self.retriever:
                logger.error("No retrieval mechanism available")
                raise ValueError("A retrieval mechanism must be initialized before creating the chain")
                
            logger.info("Creating code generation chains")
            
            # Use class-level context window if available, or retrieve from model
            context_window = None
            if hasattr(self, 'context_window') and self.context_window:
                context_window = self.context_window
                logger.info(f"Using stored context window: {context_window} tokens")
            elif hasattr(self, 'llm'):
                context_window = get_context_window(self.llm)
                logger.info(f"Retrieved model context window: {context_window} tokens")
            
            # Create the context formatter function with adaptive formatting
            def get_formatted_context(inputs):
                # Get retrieval query (could be "query" or "question" depending on input)
                query = inputs.get("query", inputs.get("question", ""))
                
                # Get documents using shared retriever
                docs = self.custom_retriever(query)
                
                if not docs:
                    logger.warning("No documents retrieved for query")
                    return ""
                    
                # Format documents with adaptive context optimization
                return format_docs_with_adaptive_context(docs, context_window=context_window)
            
            # Create the standard chain for general use
            logger.info("Creating standard chain")
            self.chain = {
                "context": get_formatted_context,
                "question": RunnablePassthrough()
            } | self.prompt | self.llm | StrOutputParser()
                
            # Create the specialized chain for repository-based code generation
            logger.info("Creating repository-based code generation chain")
            self.repository_chain = {
                "context": get_formatted_context,
                "question": RunnablePassthrough()
            } | self.code_generation_prompt | self.llm | StrOutputParser()
            
            # Create a direct code generation chain without repository context
            logger.info("Creating direct code generation chain")
            self.direct_chain = {
                "context": lambda _: "",  # Empty context for direct questions
                "question": RunnablePassthrough()
            } | self.code_generation_prompt | self.llm | StrOutputParser()
            
            logger.info("Code generation chains created successfully")
        except Exception as e:
            logger.error(f"Error creating code generation chain: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def predict(self, context, model_input: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate code based on the input parameters.
        
        Args:
            context: MLflow model context
            model_input: Input data for code generation, expecting:
                         - A dict with "inputs" containing any of:
                           - "question": User's code generation request (required)
                           - "repository_url": GitHub repository URL (optional)
                           - "metadata_only": Process only metadata without full LLM analysis (optional, default: False)
                           - "process_timeout": Maximum time for repository processing in seconds (optional, default: 300)
                           - "batch_size": Number of files to process in each batch (optional, default: 20)
            
        Returns:
            DataFrame with the generated code in a "result" column
        """
        # Set reasonable logging levels to reduce clutter
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        
        logger.info(f"Received model_input: {str(model_input)[:200]}...")
        
        # Extract input data from the MLFlow API format
        if "inputs" in model_input:
            input_data = model_input["inputs"]
        else:
            input_data = model_input
        
        # Extract main fields from input data
        question = ""
        repository_url = None
        metadata_only = False
        process_timeout = 300  # Default 5 minutes
        batch_size = 20  # Default batch size for repository processing
        
        # Extract question field (required)
        if "question" in input_data:
            if hasattr(input_data["question"], "iloc"):
                question = input_data["question"].iloc[0] if not input_data["question"].empty else ""
            else:
                question = input_data["question"]
        
        # Extract repository_url field (optional)
        if "repository_url" in input_data:
            if hasattr(input_data["repository_url"], "iloc"):
                repository_url = input_data["repository_url"].iloc[0] if not input_data["repository_url"].empty else None
            else:
                repository_url = input_data["repository_url"]
        else:
            logger.info("No repository_url provided, resetting repository state")
            self.reset_repository_state()
        
        # Extract metadata_only parameter (optional)
        if "metadata_only" in input_data:
            if hasattr(input_data["metadata_only"], "iloc"):
                metadata_only = input_data["metadata_only"].iloc[0] if not input_data["metadata_only"].empty else False
            else:
                metadata_only = bool(input_data["metadata_only"])
        
        # Extract process_timeout parameter (optional)
        if "process_timeout" in input_data:
            try:
                if hasattr(input_data["process_timeout"], "iloc"):
                    process_timeout = int(input_data["process_timeout"].iloc[0]) if not input_data["process_timeout"].empty else 300
                else:
                    process_timeout = int(input_data["process_timeout"])
            except (ValueError, TypeError):
                logger.warning("Invalid process_timeout value, using default (300 seconds)")
                process_timeout = 300
        
        # Extract batch_size parameter (optional)
        if "batch_size" in input_data:
            try:
                if hasattr(input_data["batch_size"], "iloc"):
                    batch_size = int(input_data["batch_size"].iloc[0]) if not input_data["batch_size"].empty else 20
                else:
                    batch_size = int(input_data["batch_size"])
            except (ValueError, TypeError):
                logger.warning("Invalid batch_size value, using default (20)")
                batch_size = 20
        
        # Check if question field is provided
        if not question:
            logger.warning("No question provided for code generation")
            return pd.DataFrame([{"result": "Error: No question provided for code generation."}])
        
        try:
            logger.info(f"Processing code generation request for question: {str(question)[:50]}...")
            logger.info(f"Parameters: metadata_only={metadata_only}, timeout={process_timeout}s, batch_size={batch_size}")
            
            # If repository_url is provided, process it first
            if repository_url:
                logger.info(f"Repository URL provided: {repository_url}")
                try:
                    # Extract repository data with the specified parameters
                    start_time = time.time()
                    self.extract_repository(
                        repository_url, 
                        metadata_only=metadata_only,
                        timeout=process_timeout,
                        batch_size=batch_size
                    )
                    processing_time = time.time() - start_time
                    logger.info(f"Repository processing completed in {processing_time:.2f} seconds")
                    
                    # If we have data in the collection, use it for code generation
                    if self.collection:
                        try:
                            count = self.collection.count()
                            logger.info(f"Collection '{self.collection_name}' has {count} documents")
                            
                            # Use the repository chain with the question
                            chain_input = {"question": question, "query": question}
                            logger.info(f"Using repository chain with input: {chain_input}")
                            
                            # Process with repository context
                            if hasattr(self, 'protect_tool') and self.protect_tool is not None:
                                try:
                                    result = self.repository_chain.invoke(
                                        chain_input, 
                                        config={"callbacks": [self.prompt_handler] if hasattr(self, 'prompt_handler') else None}
                                    )
                                except Exception as protect_error:
                                    logger.warning(f"Error with repository chain: {str(protect_error)}")
                                    # Fall back to direct chain
                                    result = self.direct_chain.invoke(
                                        chain_input,
                                        config={"callbacks": [self.prompt_handler] if hasattr(self, 'prompt_handler') else None}
                                    )
                            else:
                                result = self.repository_chain.invoke(
                                    chain_input,
                                    config={"callbacks": [self.prompt_handler] if hasattr(self, 'prompt_handler') else None}
                                )
                            
                            # Include repository processing info in response for observability
                            processing_info = {
                                "processing_time_seconds": processing_time,
                                "metadata_only": metadata_only,
                                "document_count": count,
                                "repository_url": repository_url
                            }
                        except Exception as count_error:
                            logger.warning(f"Could not access collection: {str(count_error)}")
                            # Fall back to direct generation
                            result = self.direct_chain.invoke(
                                {"question": question},
                                config={"callbacks": [self.prompt_handler] if hasattr(self, 'prompt_handler') else None}
                            )
                            processing_info = {
                                "processing_time_seconds": processing_time,
                                "metadata_only": metadata_only,
                                "error": "collection_access_failed",
                                "repository_url": repository_url
                            }
                    else:
                        # If no collection is available, fall back to direct generation
                        logger.warning("No collection available, falling back to direct generation")
                        result = self.direct_chain.invoke(
                            {"question": question},
                            config={"callbacks": [self.prompt_handler] if hasattr(self, 'prompt_handler') else None}
                        )
                        processing_info = {
                            "processing_time_seconds": processing_time,
                            "metadata_only": metadata_only,
                            "error": "no_collection_created",
                            "repository_url": repository_url
                        }
                except Exception as repo_error:
                    logger.error(f"Error processing repository: {str(repo_error)}")
                    # Fall back to direct generation
                    result = self.direct_chain.invoke(
                        {"question": question},
                        config={"callbacks": [self.prompt_handler] if hasattr(self, 'prompt_handler') else None}
                    )
                    processing_info = {
                        "error": f"repository_processing_failed: {str(repo_error)[:100]}",
                        "repository_url": repository_url,
                        "metadata_only": metadata_only
                    }
            else:
                # Process the request using direct generation (no repository context)
                logger.info("No repository URL provided, using direct code generation")
                # Ensure we're not using any previous repository state
                self.reset_repository_state()
                result = self.direct_chain.invoke(
                    {"question": question},
                    config={"callbacks": [self.prompt_handler] if hasattr(self, 'prompt_handler') else None}
                )
                processing_info = {"mode": "direct_generation"}
            
            logger.info("Code generation processed successfully")
            
            # Clean up the result using the imported clean_code utility function
            clean_code_result = clean_code(result)
            
            # Log processing info 
            logger.info(f"Processing info: {json.dumps(processing_info)}")
            
            # Return only the clean code without any prefixes
            return pd.DataFrame([{"result": clean_code_result}])
        except Exception as e:
            error_message = f"Error processing code generation: {str(e)}"
            logger.error(error_message)
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame([{"result": f"# Error during processing\n# {error_message}\n\n# Falling back to basic response\n\n# Your question was: {question}\n\n# Please try again with metadata_only=True or a smaller repository"}])
    
    @classmethod
    def log_model(cls, secrets_path, config_path, model_path=None, embedding_model_path=None):
        """
        Log the model to MLflow.
        
        Args:
            secrets_path: Path to the secrets file
            config_path: Path to the configuration file
            model_path: Path to the LLM model file (optional)
            embedding_model_path: Path to the locally saved embedding model directory (optional)
                                 If provided, will be used instead of downloading the model
            
        Returns:
            None
        """
        import mlflow
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec
        import logging
        import os
        
        logger = logging.getLogger(__name__)
        
        # Define model input/output schema with all parameters
        input_schema = Schema([
            ColSpec("string", "question"),
            ColSpec("string", "repository_url", required=False),  # Optional repository URL
            ColSpec("boolean", "metadata_only", required=False),  # Optional metadata-only flag
            ColSpec("long", "process_timeout", required=False),   # Optional process timeout in seconds
            ColSpec("long", "batch_size", required=False)         # Optional batch size
        ])
        output_schema = Schema([
            ColSpec("string", "result")
        ])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        # Prepare artifacts
        artifacts = {
            "secrets": secrets_path,
            "config": config_path
        }
        
        if model_path:
            artifacts["models"] = model_path
            
        # Add embedding model path to artifacts if provided and exists
        # This will allow us to use a locally saved model instead of downloading it during initialization
        if embedding_model_path and os.path.exists(embedding_model_path):
            artifacts["embedding_model"] = embedding_model_path
            logger.info(f"Using local embedding model from: {embedding_model_path}")
        else:
            logger.warning("No local embedding model path provided or path doesn't exist. " 
                         "The service will download the embedding model during initialization.")
        
        # Log model to MLflow
        mlflow.pyfunc.log_model(
            artifact_path="code_generation_service",
            python_model=cls(),
            artifacts=artifacts,
            signature=signature,
            code_paths=["./core", "../../src"],
            pip_requirements=[
                "mlflow==2.9.2", 
                "langchain", 
                "promptquality", 
                "galileo-protect==0.15.1", 
                "galileo-observe==1.13.2",
                "chromadb",
                "langchain_core",
                "langchain_huggingface",
                "langchain_community",
                "sentence-transformers",
                "gitpython",
                "pyyaml"
            ]
        )
        logger.info("Model and artifacts successfully registered in MLflow.")
    
    def load_context(self, context) -> None:
        """
        Load context for the model, including configuration, model, and chains.
        This is an override of the BaseGenerativeService's load_context method.
        
        Args:
            context: MLflow model context
        """
        # First, initialize the embedding function - will check for artifact model first
        embedding_model_path = None
        if "embedding_model" in context.artifacts:
            embedding_model_path = context.artifacts["embedding_model"]
            if os.path.exists(embedding_model_path):
                logger.info(f"Found saved embedding model in artifacts: {embedding_model_path}")
            else:
                logger.warning(f"Embedding model path provided in artifacts but not found: {embedding_model_path}")
                embedding_model_path = None
        
        # Initialize the embedding function with the artifact path if available, otherwise use default
        try:
            self.initialize_embedding_function(embedding_model_path)
        except Exception as e:
            logger.warning(f"Failed to initialize embedding function: {str(e)}")
            logger.warning("Will attempt to initialize default embedding model as fallback")
            try:
                self.initialize_embedding_function()
            except Exception as e2:
                logger.error(f"Failed to initialize default embedding model: {str(e2)}")
                # Continue with initialization even if embedding fails - some functions might not need it
        
        # Call the parent load_context method to handle the rest of the initialization
        super().load_context(context)
        
        # Call the parent load_context method to handle the rest of the initialization
        super().load_context(context)
    
    def reset_repository_state(self, repository_url=None):
        """
        Reset the repository state if no specific repository URL is provided.
        
        Args:
            repository_url: If provided, keep this repository's data; otherwise, reset completely
        """
        if repository_url is None:
            logger.info("Resetting active repository state - no repository URL provided")
            self.collection = None
            
            # Reset vector store if possible
            if hasattr(self, 'vector_store') and self.vector_store is not None:
                try:
                    logger.info("Attempting to reset vector store")
                    # Create new empty vector store with the same embedding function
                    if self.embedding_function is not None:
                        self.vector_store = Chroma(embedding_function=self.embedding_function)
                        self.retriever = self.vector_store.as_retriever()
                        logger.info("Vector store reset successfully")
                except Exception as e:
                    logger.warning(f"Failed to reset vector store: {str(e)}")
        else:
            logger.info(f"Setting repository state for: {repository_url}")
            # If repository is in cache, activate it
            if repository_url in self.repository_cache:
                logger.info(f"Repository {repository_url} found in cache, activating")
                self.collection = self.repository_cache[repository_url]["collection"]
            # Repository isn't in cache yet - it will be processed and added later
