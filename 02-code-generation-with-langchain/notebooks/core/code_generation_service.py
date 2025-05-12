"""
Code Generation Service implementation that extends the BaseGenerativeService.

This service provides code generation capabilities using LLM models with vector retrieval
and integrates with Galileo for protection, observation, and evaluation.

It supports extracting code context from GitHub repositories to improve code generation quality.
"""

import os
import logging
from typing import Dict, Any, List
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from galileo_protect import ProtectParser
import chromadb
import uuid
from urllib.parse import urlparse

# Import base service class from the shared location
import sys
import os

# Add the src directory to the path to import base_service
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.service.base_service import BaseGenerativeService

# Import modules for GitHub repository extraction and processing
from core.extract_text.github_repository_extractor import GitHubRepositoryExtractor
from core.generate_metadata.llm_context_updater import LLMContextUpdater
from core.dataflow.dataflow import EmbeddingUpdater, DataFrameConverter
from core.vector_database.vector_store_writer import VectorStoreWriter

# Set up logger
logger = logging.getLogger(__name__)

class CodeGenerationService(BaseGenerativeService):
    """
    Code Generation Service that extends the BaseGenerativeService.
    
    This service can extract code context from GitHub repositories and use it
    to generate relevant code based on user questions.
    """

    def __init__(self):
        """Initialize the code generation service."""
        super().__init__()
        self.vector_store = None
        self.retriever = None
        self.collection = None
        self.collection_name = "my_collection"
        self.embedding_path = None
        self.temp_dir = os.path.join(os.path.dirname(__file__), "temp_repositories")
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def process_github_repository(self, repository_url: str) -> List[Dict]:
        """
        Process a GitHub repository and extract code snippets with context.
        
        Args:
            repository_url: URL of the GitHub repository to process
            
        Returns:
            List of extracted code snippets with context
        """
        try:
            logger.info(f"Processing GitHub repository: {repository_url}")
            
            # Create a unique directory name for this repository
            parsed_url = urlparse(repository_url)
            path_parts = [p for p in parsed_url.path.split('/') if p]
            repo_dir_name = f"{path_parts[0]}_{path_parts[1]}_{uuid.uuid4().hex[:8]}"
            save_dir = os.path.join(self.temp_dir, repo_dir_name)
            
            # Extract repository content
            extractor = GitHubRepositoryExtractor(
                repo_url=repository_url,
                save_dir=save_dir,
                verbose=True
            )
            extracted_data = extractor.run()
            
            logger.info(f"Extracted {len(extracted_data)} code snippets from repository")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing GitHub repository: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def generate_code_context(self, repository_url: str) -> bool:
        """
        Generate code context from a GitHub repository and store it in the vector database.
        
        Args:
            repository_url: URL of the GitHub repository
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Step 1: Extract code from repository
            extracted_data = self.process_github_repository(repository_url)
            if not extracted_data:
                logger.error("No data extracted from repository")
                return False
                
            # Step 2: Generate metadata with LLM if available
            if self.llm:
                # Define template for context generation
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
                from langchain_core.prompts import PromptTemplate
                prompt = PromptTemplate.from_template(template)
                llm_chain = prompt | self.llm
                
                # Update context with LLM
                updater = LLMContextUpdater(
                    llm_chain=llm_chain,
                    prompt_template=template,
                    verbose=True
                )
                extracted_data = updater.update(extracted_data)
                logger.info("Context metadata generated with LLM")
                
            # Step 3: Generate embeddings
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                # Fall back to older import path
                from langchain.embeddings import HuggingFaceEmbeddings
            
            # Initialize embedding model
            embedding_path = getattr(self, 'embedding_path', None)
            if embedding_path and os.path.exists(embedding_path):
                embeddings = HuggingFaceEmbeddings(model_name=embedding_path)
            else:
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Generate embeddings
            updater = EmbeddingUpdater(embedding_model=embeddings, verbose=True)
            embedded_data = updater.update(extracted_data)
            logger.info("Embeddings generated for extracted code snippets")
            
            # Step 4: Convert to DataFrame
            converter = DataFrameConverter(verbose=True)
            df = converter.to_dataframe(embedded_data)
            logger.info(f"DataFrame created with {len(df)} rows")
            
            # Step 5: Store in vector database
            writer = VectorStoreWriter(collection_name=self.collection_name, verbose=True)
            writer.upsert_dataframe(df)
            logger.info("Data stored in vector database")
            
            # Update collection reference
            self.collection = writer.collection
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating code context: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def custom_retriever(self, query: str, top_n: int = 10) -> List[Document]:
        """
        Custom retriever function to get relevant code snippets based on a query.
        
        Args:
            query: The query string for retrieval
            top_n: Number of documents to retrieve
            
        Returns:
            List of Document objects with content and metadata
        """
        if not self.collection:
            logger.error("No collection available for retrieval")
            return []
            
        try:
            logger.info(f"Retrieving documents for query: {query[:30]}... (top {top_n})")
            results = self.collection.query(
                query_texts=[query],
                n_results=top_n
            )
            
            documents = [
                Document(
                    page_content=str(results['documents'][i]),
                    metadata=results['metadatas'][i] if isinstance(results['metadatas'][i], dict) else results['metadatas'][i][0]
                )
                for i in range(len(results['documents']))
            ]
            
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
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
            
            # Check if embedding model was provided as an artifact
            embedding_path = getattr(self, 'embedding_path', None)
            
            if embedding_path and os.path.exists(embedding_path):
                logger.info(f"Loading embedding model from local path: {embedding_path}")
                
                # Import here to avoid issues with imports in different environments
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                except ImportError:
                    # Fall back to older import path
                    from langchain.embeddings import HuggingFaceEmbeddings
                
                # Load the model from the saved directory
                embedding_function = HuggingFaceEmbeddings(model_name=embedding_path)
                logger.info("Embedding model loaded successfully from local path.")
            else:
                # If no local path is available, initialize from Hugging Face hub
                logger.info("No local embedding model path found. Downloading from Hugging Face hub...")
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                except ImportError:
                    # Fall back to older import path
                    from langchain.embeddings import HuggingFaceEmbeddings
                
                embedding_function = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )
                logger.info("Embedding model downloaded and initialized from Hugging Face hub.")
            
            # Initialize chromadb client
            client = chromadb.PersistentClient(path=persist_directory)
            
            # Try to get existing collection or create a new one
            try:
                logger.info(f"Trying to get collection: {self.collection_name}")
                self.collection = client.get_collection(name=self.collection_name)
                logger.info(f"Collection '{self.collection_name}' found")
            except Exception as col_err:
                logger.info(f"Collection not found, creating new one: {str(col_err)}")
                self.collection = client.create_collection(name=self.collection_name)
                logger.info(f"Created new collection: {self.collection_name}")
            
            # Initialize LangChain vector store with the embedding function
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_function,
                collection_name=self.collection_name
            )
            self.retriever = self.vector_store.as_retriever()
            logger.info(f"Vector store successfully loaded from {persist_directory}")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.info("Creating new empty vector store")
            # Create an empty vector store if loading fails
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                # Fall back to older import path
                from langchain.embeddings import HuggingFaceEmbeddings
            
            # Fallback to a different model if the main one fails
            embedding_function = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
            self.vector_store = Chroma(embedding_function=embedding_function)
            self.retriever = self.vector_store.as_retriever()
            logger.info("Created new empty vector store")
    
    def load_model(self, context) -> None:
        """
        Load the appropriate model based on configuration.
        
        Args:
            context: MLflow model context containing artifacts
        """
        try:
            model_source = self.model_config.get("model_source", "local")
            logger.info(f"Attempting to load model from source: {model_source}")
            
            if model_source == "local":
                logger.info("Using local LlamaCpp model source")
                self.load_local_model(context)
            else:
                error_msg = f"Unsupported model source: {model_source}. Currently only 'local' is supported for code generation."
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            if self.llm is None:
                logger.error("Model failed to initialize - llm is None after loading")
                raise RuntimeError("Model initialization failed - llm is None")
                
            logger.info(f"Model of type {type(self.llm).__name__} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
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
                logger.error(f"Model file not found at: {model_path}")
                raise FileNotFoundError(f"The model file was not found at: {model_path}")
            
            logger.info(f"Model file exists. Size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
            
            logger.info("Setting up callback manager")
            self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            logger.info("Initializing LlamaCpp with the following parameters:")
            logger.info(f"  - Model Path: {model_path}")
            logger.info(f"  - n_gpu_layers: 30, n_batch: 512, n_ctx: 4096")
            logger.info(f"  - max_tokens: 1024, f16_kv: True, temperature: 0.2")
            
            try:
                self.llm = LlamaCpp(
                    model_path=model_path,
                    n_gpu_layers=30,
                    n_batch=512,
                    n_ctx=4096,
                    max_tokens=1024,
                    f16_kv=True,
                    callback_manager=self.callback_manager,
                    verbose=False,
                    stop=[],
                    streaming=False,
                    temperature=0.2,
                )
                logger.info("LlamaCpp model initialized successfully.")
            except Exception as model_error:
                logger.error(f"Failed to initialize LlamaCpp model: {str(model_error)}")
                logger.error(f"Exception type: {type(model_error).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
                
            logger.info("Using local LlamaCpp model for code generation.")
        except Exception as e:
            logger.error(f"Error in load_local_model: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def load_prompt(self) -> None:
        """Load the prompt template for code generation."""
        self.prompt_str = """You are a Python wizard tasked with generating code for a Jupyter Notebook (.ipynb) based on the given context.
Your answer should consist of just the Python code, without any additional text or explanation.

The context below contains code snippets from a GitHub repository that can help you generate the appropriate code.
Use this code as a reference to understand structure, style, and patterns, but adapt it to solve the specific problem in the question.

Context:
{context}

Question: {question}
"""
        self.prompt = ChatPromptTemplate.from_template(self.prompt_str)
    
    def format_docs(self, docs: List[Document]) -> str:
        """
        Format a list of documents into a single string.
        
        Args:
            docs: List of Document objects
            
        Returns:
            Formatted string of document contents
        """
        return "\n\n".join([doc.page_content for doc in docs])
    
    def load_chain(self) -> None:
        """Create the code generation chain using the loaded model, prompt, and retriever."""
        try:
            # Load the vector store first
            logger.info("Loading vector store for retrieval")
            self.load_vector_store()
            
            # We don't require the retriever to be ready immediately, as repository context will be loaded on demand
            logger.info("Creating code generation chain")
            
            # Create the chain with enhanced error handling for empty or missing collections
            self.chain = {
                "context": lambda inputs: self.format_docs(
                    self.custom_retriever(inputs["query"]) if self.collection else 
                    [Document(page_content="No code context available. Please provide a repository URL.", metadata={})]
                ),
                "question": RunnablePassthrough()
            } | self.prompt | self.llm | StrOutputParser()
                
            logger.info("Code generation chain created successfully")
        except Exception as e:
            logger.error(f"Error creating code generation chain: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def predict(self, context, model_input):
        """
        Generate code based on the input question and optional GitHub repository URL.
        
        Args:
            context: MLflow model context
            model_input: Input data for code generation, expecting either:
                         - A direct dict with "question" and optional "repository_url" fields
                         - A dict with "inputs" containing these fields
            
        Returns:
            Dictionary with the generated code in a "result" field
        """
        logger.info(f"Received model_input: {str(model_input)[:200]}...")
        
        # Handle MLFlow API format where input is in {"inputs": {...}} format
        if "inputs" in model_input:
            # Extract from the inputs wrapper
            input_data = model_input["inputs"]
        else:
            # Direct input format
            input_data = model_input
            
        # Extract question and repository_url fields
        question = ""
        repository_url = ""
        
        # Handle question field (used for code generation prompt)
        if "question" in input_data:
            if hasattr(input_data["question"], "iloc"):
                question = input_data["question"].iloc[0] if not input_data["question"].empty else ""
            else:
                # Handle when it's a list (from API format) or a direct string
                if isinstance(input_data["question"], list) and len(input_data["question"]) > 0:
                    question = input_data["question"][0]
                else:
                    question = input_data["question"]
        
        # Handle repository_url field (GitHub repository for context)
        if "repository_url" in input_data:
            if hasattr(input_data["repository_url"], "iloc"):
                repository_url = input_data["repository_url"].iloc[0] if not input_data["repository_url"].empty else ""
            else:
                # Handle when it's a list (from API format) or a direct string
                if isinstance(input_data["repository_url"], list) and len(input_data["repository_url"]) > 0:
                    repository_url = input_data["repository_url"][0]
                else:
                    repository_url = input_data["repository_url"]
        
        # Check if question is provided
        if not question:
            logger.warning("No question provided for code generation prompt")
            return pd.DataFrame([{"result": "Error: No question provided for code generation prompt."}])
        
        try:
            # Process GitHub repository if provided
            if repository_url:
                logger.info(f"Processing GitHub repository: {repository_url}")
                success = self.generate_code_context(repository_url)
                if not success:
                    return pd.DataFrame([{"result": f"Error: Failed to process GitHub repository: {repository_url}"}])
            else:
                logger.info("No repository URL provided, using existing context database")
            
            # Log some info about the collection state
            if self.collection:
                try:
                    count = self.collection.count()
                    logger.info(f"Collection '{self.collection_name}' has {count} documents")
                except Exception as count_error:
                    logger.warning(f"Could not get collection count: {str(count_error)}")
            else:
                logger.warning("No collection available for retrieval. Results may lack context.")
            
            # Use the question as the query for retrieval
            query = question
            
            # Create the input dictionary for the chain
            chain_input = {"query": query, "question": question}
            logger.info(f"Passing to chain: {chain_input}")
            
            # When using Galileo Protect, wrap the input in a custom format expected by Pydantic
            if self.protect_tool is not None:
                try:
                    # First try with the original format (already wrapped as required by Galileo)
                    result = self.protected_chain.invoke(
                        {"input": chain_input, "output": ""}, 
                        config={"callbacks": [self.prompt_handler]}
                    )
                except Exception as protect_error:
                    logger.warning(f"Error with protected chain using input/output format: {str(protect_error)}")
                    # Try additional wrapper format for Pydantic validation
                    try:
                        # Try with a different format as a fallback
                        result = self.protected_chain.invoke(
                            {"input": {"input": chain_input}, "output": {"output": ""}}, 
                            config={"callbacks": [self.prompt_handler]}
                        )
                    except Exception as nested_error:
                        logger.error(f"Both protection formats failed. Falling back to direct chain. Error: {str(nested_error)}")
                        # Fallback to direct chain if all else fails
                        result = self.chain.invoke(
                            chain_input,
                            config={"callbacks": [self.prompt_handler]}
                        )
            else:
                # Run the regular chain without Galileo Protect
                result = self.chain.invoke(
                    chain_input,
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
            import traceback
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
            
        Returns:
            None
        """
        import mlflow
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec
        import logging
        import os
        
        logger = logging.getLogger(__name__)
        
        # Define model input/output schema
        input_schema = Schema([
            ColSpec("string", "repository_url"),
            ColSpec("string", "question")
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
            
        # Add embedding model to artifacts if path is provided
        if embedding_model_path and os.path.exists(embedding_model_path):
            artifacts["embedding_model"] = embedding_model_path
            logger.info(f"Using local embedding model from: {embedding_model_path}")
        else:
            logger.warning("No local embedding model path provided or path doesn't exist. "
                         "The service will download the model during initialization.")
        
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
                "pyyaml"
            ]
        )
        logger.info("Model and artifacts successfully registered in MLflow.")
    
    def validate_github_url(self, url: str) -> bool:
        """
        Validate that the provided URL is a valid GitHub repository URL.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not url:
                return False
                
            parsed = urlparse(url)
            
            # Check if it's a GitHub URL
            if not parsed.netloc.endswith('github.com'):
                logger.error(f"Not a GitHub URL: {url}")
                return False
                
            # Check if it has owner/repo format
            path_parts = [p for p in parsed.path.split('/') if p]
            if len(path_parts) < 2:
                logger.error(f"Invalid GitHub repository URL format: {url}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating GitHub URL: {str(e)}")
            return False
    
    def cleanup_temp_repositories(self, max_age_hours: int = 24) -> None:
        """
        Clean up temporary repository directories older than the specified age.
        
        Args:
            max_age_hours: Maximum age in hours for repositories to keep
        """
        try:
            if not os.path.exists(self.temp_dir):
                return
                
            import time
            import shutil
            
            now = time.time()
            max_age_seconds = max_age_hours * 60 * 60
            
            for item in os.listdir(self.temp_dir):
                item_path = os.path.join(self.temp_dir, item)
                if os.path.isdir(item_path):
                    # Check if directory is older than max_age
                    mtime = os.path.getmtime(item_path)
                    if now - mtime > max_age_seconds:
                        shutil.rmtree(item_path)
                        logger.info(f"Removed old repository directory: {item_path}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary repositories: {str(e)}")
            
    def load_context(self, context) -> None:
        """
        Load context for the model, including configuration, model, and chains.
        This is an override of the BaseGenerativeService's load_context method to include
        loading the embedding model from artifacts and cleaning up temp repositories.
        
        Args:
            context: MLflow model context
        """
        # Get the embedding model path from artifacts if available
        if "embedding_model" in context.artifacts:
            self.embedding_path = context.artifacts["embedding_model"]
            logger.info(f"Found embedding model artifact at {self.embedding_path}")
        
        # Clean up old temporary repositories
        self.cleanup_temp_repositories()
        
        # Call the parent load_context method to handle the rest of the initialization
        super().load_context(context)
    
    def extract_github_repository(self, repo_url: str, branch: str = "main") -> str:
        """
        Extract code files from a GitHub repository and save them to a local directory.
        
        Args:
            repo_url: URL of the GitHub repository (e.g., 'https://github.com/user/repo')
            branch: Branch name to extract code from (default: 'main')
        
        Returns:
            Path to the directory containing the extracted code files
        """
        try:
            logger.info(f"Extracting GitHub repository: {repo_url} (branch: {branch})")
            
            # Generate a unique ID for this extraction task
            task_id = str(uuid.uuid4())
            temp_repo_dir = os.path.join(self.temp_dir, task_id)
            
            # Ensure the temp directory for this task exists
            os.makedirs(temp_repo_dir, exist_ok=True)
            
            # Extract the repository using the GitHubRepositoryExtractor
            extractor = GitHubRepositoryExtractor(repo_url, branch, temp_repo_dir)
            file_mapping = extractor.extract()
            
            logger.info(f"Extracted files: {file_mapping}")
            
            return temp_repo_dir
        except Exception as e:
            logger.error(f"Error extracting GitHub repository: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def update_context_with_repository(self, repo_path: str, question: str) -> str:
        """
        Update the code generation context with relevant code from the extracted repository.
        
        Args:
            repo_path: Path to the directory containing the extracted code files
            question: The question or prompt for code generation
            
        Returns:
            Updated context string including relevant code snippets
        """
        try:
            logger.info(f"Updating context with repository code: {repo_path}")
            
            # Use the LLMContextUpdater to analyze the repository and update the context
            updater = LLMContextUpdater(repo_path, question)
            updated_context = updater.update_context()
            
            logger.info("Context updated successfully")
            
            return updated_context
        except Exception as e:
            logger.error(f"Error updating context with repository: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise