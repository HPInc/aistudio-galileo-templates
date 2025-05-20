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
from src.utils import get_context_window, dynamic_retriever, format_docs_with_adaptive_context

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
    
    def extract_repository(self, repository_url: str) -> List[Dict[str, Any]]:
        """
        Extract code and metadata from a GitHub repository.
        Uses a cache mechanism to avoid re-processing the same repository.
        
        Args:
            repository_url: URL of the GitHub repository
            
        Returns:
            List of dictionaries containing extracted code and metadata
        """
        try:
            # Check if repository is already in cache
            if repository_url in self.repository_cache:
                logger.info(f"Using cached data for repository: {repository_url}")
                
                # Update the collection reference from the cache
                self.collection = self.repository_cache[repository_url]["collection"]
                
                # Check if the collection exists and has documents
                if self.collection:
                    try:
                        count = self.collection.count()
                        logger.info(f"Cache hit: Collection has {count} documents")
                        return self.repository_cache[repository_url]["data"]
                    except Exception as e:
                        logger.warning(f"Cached collection error: {str(e)}. Re-processing repository.")
                        # Continue with fresh processing if we can't access the cached collection
            
            logger.info(f"Extracting code and metadata from repository: {repository_url}")
            
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
            logger.info(f"Extracted {len(extracted_data)} code snippets from repository")
            
            # Step 2: Use LLM to generate metadata for each code snippet
            # Create a template for metadata generation
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
            
            # Create the chain for generating metadata
            from langchain_core.prompts import PromptTemplate
            prompt = PromptTemplate.from_template(template)
            
            # Only use the LLM if it's been initialized
            if self.llm:
                logger.info("Using existing LLM model for metadata generation")
                llm_chain = prompt | self.llm
            else:
                logger.warning("LLM not initialized, skipping metadata generation")
                return extracted_data
                
            # Update contexts using LLM
            updater = LLMContextUpdater(
                llm_chain=llm_chain,
                prompt_template=template,
                verbose=False,
                print_prompt=False
            )
            updated_data = updater.update(extracted_data)
            logger.info("Metadata generation complete")
            
            # Step 3: Generate embeddings for each code snippet
            embedding_updater = EmbeddingUpdater(embedding_model=self.embedding_function, verbose=False)
            updated_data = embedding_updater.update(updated_data)
            logger.info("Embeddings generated successfully")
            
            # Step 4: Convert to DataFrame for easier processing
            converter = DataFrameConverter(verbose=False)
            df = converter.to_dataframe(updated_data)
            logger.info("Data conversion to DataFrame complete")
            
            # Step 5: Store in vector database
            writer = VectorStoreWriter(collection_name=self.collection_name, verbose=False)
            writer.upsert_dataframe(df)
            logger.info(f"Repository data stored in collection: {self.collection_name}")
            
            # Update the collection reference
            self.collection = writer.collection
            
            # Store in cache for future use
            import datetime
            self.repository_cache[repository_url] = {
                "data": updated_data,
                "collection": self.collection,
                "timestamp": datetime.datetime.now().isoformat()
            }
            logger.info(f"Repository {repository_url} added to cache")
            
            return updated_data
        except Exception as e:
            logger.error(f"Error extracting repository data: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
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
            
        Returns:
            DataFrame with the generated code in a "result" column
        """
        logger.info(f"Received model_input: {str(model_input)[:200]}...")
        
        # Extract input data from the MLFlow API format
        if "inputs" in model_input:
            input_data = model_input["inputs"]
        else:
            input_data = model_input
        
        # Extract fields from input data
        question = ""
        repository_url = None
        
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
        
        # Check if question field is provided
        if not question:
            logger.warning("No question provided for code generation")
            return pd.DataFrame([{"result": "Error: No question provided for code generation."}])
        
        try:
            logger.info(f"Processing code generation request for question: {str(question)[:50]}...")
            
            # If repository_url is provided, process it first
            if repository_url:
                logger.info(f"Repository URL provided: {repository_url}")
                try:
                    # Extract repository data and store it in vector database
                    self.extract_repository(repository_url)
                    
                    # If we have data in the collection, use it for code generation
                    if self.collection:
                        try:
                            count = self.collection.count()
                            logger.info(f"Collection '{self.collection_name}' has {count} documents")
                            
                            # Use the repository chain with the question
                            chain_input = {"question": question, "query": question}
                            logger.info(f"Using repository chain with input: {chain_input}")
                            
                            # Process with repository context
                            if self.protect_tool is not None:
                                try:
                                    result = self.repository_chain.invoke(
                                        chain_input, 
                                        config={"callbacks": [self.prompt_handler]}
                                    )
                                except Exception as protect_error:
                                    logger.warning(f"Error with repository chain: {str(protect_error)}")
                                    # Fall back to direct chain
                                    result = self.direct_chain.invoke(
                                        chain_input,
                                        config={"callbacks": [self.prompt_handler]}
                                    )
                            else:
                                result = self.repository_chain.invoke(
                                    chain_input,
                                    config={"callbacks": [self.prompt_handler]}
                                )
                        except Exception as count_error:
                            logger.warning(f"Could not access collection: {str(count_error)}")
                            # Fall back to direct generation
                            result = self.direct_chain.invoke(
                                {"question": question},
                                config={"callbacks": [self.prompt_handler]}
                            )
                    else:
                        # If no collection is available, fall back to direct generation
                        logger.warning("No collection available, falling back to direct generation")
                        result = self.direct_chain.invoke(
                            {"question": question},
                            config={"callbacks": [self.prompt_handler]}
                        )
                except Exception as repo_error:
                    logger.error(f"Error processing repository: {str(repo_error)}")
                    # Fall back to direct generation
                    result = self.direct_chain.invoke(
                        {"question": question},
                        config={"callbacks": [self.prompt_handler]}
                    )
            else:
                # Process the request using direct generation (no repository context)
                logger.info("No repository URL provided, using direct code generation")
                result = self.direct_chain.invoke(
                    {"question": question},
                    config={"callbacks": [self.prompt_handler]}
                )
            
            logger.info("Code generation processed successfully")
            
            # Clean up the result (remove markdown code blocks if present)
            clean_code = result.replace("```python", "").replace("```", "").strip()
            
            return pd.DataFrame([{"result": clean_code}])
        except Exception as e:
            error_message = f"Error processing code generation: {str(e)}"
            logger.error(error_message)
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame([{"result": error_message}])
    
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
        
        # Define model input/output schema with repository_url as optional parameter
        input_schema = Schema([
            ColSpec("string", "question"),
            ColSpec("string", "repository_url", required=False)  # Make repository_url explicitly optional in schema
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
