"""
Code Generation Service implementation that extends the BaseGenerativeService.

This service provides code generation capabilities using LLM models with vector retrieval
and integrates with Galileo for protection, observation, and evaluation.
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

# Import base service class from the shared location
import sys
import os

# Add the src directory to the path to import base_service
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.service.base_service import BaseGenerativeService

# Set up logger
logger = logging.getLogger(__name__)

class CodeGenerationService(BaseGenerativeService):
    """Code Generation Service that extends the BaseGenerativeService."""

    def __init__(self):
        """Initialize the code generation service."""
        super().__init__()
        self.vector_store = None
        self.retriever = None
    
    def load_vector_store(self, persist_directory="./chroma_db"):
        """
        Load or create a vector store for code retrieval.
        
        Args:
            persist_directory: Directory to store vector database
        """
        try:
            logger.info(f"Loading vector store from {persist_directory}")
            self.vector_store = Chroma(persist_directory=persist_directory)
            self.retriever = self.vector_store.as_retriever()
            logger.info(f"Vector store successfully loaded from {persist_directory}")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.info("Creating new empty vector store")
            # Create an empty vector store if loading fails
            self.vector_store = Chroma()
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

Context:
{context}

Question: {query}
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
            
            if not self.retriever:
                logger.error("Retriever is not initialized")
                raise ValueError("Retriever must be initialized before creating the chain")
                
            logger.info("Creating code generation chain")
            # Create the chain
            self.chain = {
                "context": lambda inputs: self.format_docs(self.retriever.get_relevant_documents(inputs["question"])),
                "query": RunnablePassthrough()
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
        Generate code based on the input question.
        
        Args:
            context: MLflow model context
            model_input: Input data for code generation, expecting a "question" field
            
        Returns:
            Dictionary with the generated code in a "result" field
        """
        question = model_input.get("question", "")
        
        if not question:
            logger.warning("No question provided for code generation")
            return {"result": "Error: No question provided for code generation."}
        
        try:
            logger.info(f"Processing code generation request: {question[:50]}...")
            # Run the protected chain with monitoring
            result = self.protected_chain.invoke(
                {"input": question, "output": ""},
                config={"callbacks": [self.prompt_handler]}
            )
            logger.info("Code generation processed successfully")
            
            # Clean up the result (remove markdown code blocks if present)
            clean_code = result.replace("```python", "").replace("```", "").strip()
            
            return {"result": clean_code}
        except Exception as e:
            error_message = f"Error processing code generation: {str(e)}"
            logger.error(error_message)
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"result": error_message}
    
    @classmethod
    def log_model(cls, secrets_path, config_path, model_path=None):
        """
        Log the model to MLflow.
        
        Args:
            secrets_path: Path to the secrets file
            config_path: Path to the configuration file
            model_path: Path to the model file (optional)
            
        Returns:
            None
        """
        import mlflow
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec
        
        # Define model input/output schema
        input_schema = Schema([
            ColSpec("string", "query"),
            ColSpec("string", "question")
        ])
        output_schema = Schema([
            ColSpec("string", "generated_code")
        ])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        # Prepare artifacts
        artifacts = {
            "secrets": secrets_path,
            "config": config_path
        }
        
        if model_path:
            artifacts["models"] = model_path
        
        # Log model to MLflow
        mlflow.pyfunc.log_model(
            artifact_path="code_generation_service",
            python_model=cls(),
            artifacts=artifacts,
            signature=signature,
            pip_requirements=[
                "mlflow==2.9.2", 
                "langchain", 
                "promptquality", 
                "galileo-protect==0.15.1", 
                "galileo-observe==1.13.2",
                "chromadb",
                "langchain_core",
                "langchain_huggingface",
                "pyyaml"
            ]
        )
        logger.info("Model and artifacts successfully registered in MLflow.")