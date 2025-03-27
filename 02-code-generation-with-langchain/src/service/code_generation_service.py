"""
Code Generation Service implementation that extends the BaseGenerativeService.

This service provides code generation capabilities using LLM models with vector retrieval
and integrates with Galileo for protection, observation, and evaluation.
"""

import os
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))
from service.base_service import BaseGenerativeService

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
            self.vector_store = Chroma(persist_directory=persist_directory)
            self.retriever = self.vector_store.as_retriever()
            print(f"Vector store loaded from {persist_directory}")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            # Create an empty vector store if loading fails
            self.vector_store = Chroma()
            self.retriever = self.vector_store.as_retriever()
            print("Created new empty vector store")
    
    def load_model(self, context) -> None:
        """
        Load the appropriate model based on configuration.
        
        Args:
            context: MLflow model context containing artifacts
        """
        model_source = self.model_config.get("model_source", "local")
        
        if model_source == "local":
            self.load_local_model(context)
        else:
            raise ValueError(f"Unsupported model source: {model_source}. Currently only 'local' is supported for code generation.")
    
    def load_local_model(self, context):
        """
        Load a local LlamaCpp model.
        
        Args:
            context: MLflow model context containing artifacts
        """
        model_path = context.artifacts.get("models", None)
        
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        print(f"[INFO] Initializing local LlamaCpp model from {model_path}.")
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
        print("Using local LlamaCpp model for code generation.")
    
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
        # Load the vector store first
        self.load_vector_store()
        
        # Create the chain
        self.chain = {
            "context": lambda inputs: self.format_docs(self.retriever.get_relevant_documents(inputs["question"])),
            "query": RunnablePassthrough()
        } | self.prompt | self.llm | StrOutputParser()
    
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
            return {"result": "Error: No question provided for code generation."}
        
        try:
            # Run the protected chain with monitoring
            result = self.protected_chain.invoke(
                {"input": question, "output": ""},
                config={"callbacks": [self.prompt_handler]}
            )
            print("Code generation processed successfully.")
            
            # Clean up the result (remove markdown code blocks if present)
            clean_code = result.replace("```python", "").replace("```", "").strip()
            
            return {"result": clean_code}
        except Exception as e:
            error_message = f"Error processing code generation: {e}"
            print(error_message)
            return {"result": error_message}
    
    @classmethod
    def log_model(cls, model_folder: str, pip_requirements=None):
        """
        Log the model to MLflow with appropriate configuration.
        
        Args:
            model_folder: Path to the model folder
            pip_requirements: List of pip requirements for the model
        """
        import mlflow
        from mlflow.models import ModelSignature
        from mlflow.types.schema import Schema, ColSpec
        
        # Define the input and output schemas
        input_schema = Schema([ColSpec("string", "question")])
        output_schema = Schema([ColSpec("string", "result")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        # Default pip requirements
        if pip_requirements is None:
            pip_requirements = [
                "mlflow==2.9.2", 
                "langchain", 
                "promptquality", 
                "galileo-protect", 
                "chromadb"
            ]
        
        # Log the model to MLflow
        artifacts = {"models": model_folder}
        mlflow.pyfunc.log_model(
            artifact_path="CodeGeneration_Service",
            python_model=cls(),
            artifacts=artifacts,
            signature=signature,
            pip_requirements=pip_requirements,
        )