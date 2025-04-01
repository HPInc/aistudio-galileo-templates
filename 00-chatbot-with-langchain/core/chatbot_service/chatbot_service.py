"""
Chatbot Service implementation that extends the BaseGenerativeService.
This service provides a RAG (Retrieval-Augmented Generation) chatbot with 
Galileo integration for protection, observation, and evaluation.
"""
import os
import uuid
import base64
import logging
from typing import Dict, Any, List
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, RunnableMap
from langchain.schema.document import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from galileo_protect import ProtectParser

# Import base service class from the shared location
import sys
import os
# Add the src directory to the path to import base_service
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.service.base_service import BaseGenerativeService

# Set up logger
logger = logging.getLogger(__name__)

def format_docs(docs: List[Document]) -> str:
    """
    Format a list of documents into a single string.
    
    Args:
        docs: List of Document objects
        
    Returns:
        String containing the concatenated page content of all documents
    """
    return "\n\n".join([doc.page_content for doc in docs if isinstance(doc.page_content, str)])

class ChatbotService(BaseGenerativeService):
    """
    Chatbot Service that extends BaseGenerativeService to provide
    a RAG-based conversational AI with document retrieval capabilities.
    """
    def __init__(self):
        """Initialize the chatbot service."""
        super().__init__()
        self.memory = []
        self.embedding = None
        self.vectordb = None
        self.retriever = None
        self.prompt_str = None
        self.docs_path = None

    def load_config(self, context):
        """
        Load configuration from context artifacts and set up the docs path.

        Args:
            context: MLflow model context containing artifacts

        Returns:
            Dictionary containing the loaded configuration
        """
        # Load base configuration
        config = super().load_config(context)

        # Set docs path from artifacts
        self.docs_path = context.artifacts.get("docs", None)

        # Add additional chatbot-specific configuration
        config.update({
            "local_model_path": context.artifacts.get("models", "")
        })

        return config

    def load_model(self, context) -> None:
        """
        Load the appropriate model based on configuration.

        Args:
            context: MLflow model context containing artifacts
        """
        model_source = self.model_config.get("model_source", "local")

        if model_source == "local":
            self.load_local_model(context)
        elif model_source == "hugging-face-local":
            self.load_local_hf_model(context)
        elif model_source == "hugging-face-cloud":
            self.load_cloud_hf_model(context)
        else:
            raise ValueError(f"Unsupported model source: {model_source}")

    def load_local_model(self, context):
        """
        Load a local LlamaCpp model.

        Args:
            context: MLflow model context containing artifacts
        """
        logger.info("Initializing local LlamaCpp model.")
        model_path = self.model_config.get("local_model_path", context.artifacts.get("models", ""))

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model file was not found at: {model_path}")

        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
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
        logger.info("Using the local LlamaCpp model.")

    def load_local_hf_model(self, context):
        """
        Load a local Hugging Face model.

        Args:
            context: MLflow model context containing artifacts
        """
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, device=0)
        self.llm = HuggingFacePipeline(pipeline=pipe)        
        logger.info("Using the local Deep Seek model downloaded from HuggingFace.")

    def load_cloud_hf_model(self, context):
        """
        Load a cloud-based Hugging Face model.

        Args:
            context: MLflow model context containing artifacts
        """
        self.llm = HuggingFaceEndpoint(
            huggingfacehub_api_token=self.model_config["hf_key"],
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        )     
        logger.info("Using the cloud Mistral model on HuggingFace.")

    def load_vector_database(self):
        """
        Load documents and create the vector database for retrieval.
        """
        if not self.docs_path or not os.path.exists(self.docs_path):
            raise FileNotFoundError(f"The documents directory was not found at: {self.docs_path}")

        pdf_path = os.path.join(self.docs_path, "AIStudioDoc.pdf")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file 'AIStudioDoc.pdf' was not found at: {pdf_path}")

        logger.info(f"Reading and processing the PDF file: {pdf_path}")

        try:
            # Load PDF documents
            logger.info("Loading PDF data...")
            pdf_loader = PyPDFLoader(pdf_path)
            pdf_data = pdf_loader.load()

            # Ensure all content is string type
            for doc in pdf_data:
                if not isinstance(doc.page_content, str):
                    doc.page_content = str(doc.page_content)

            # Split documents into chunks
            logger.info("Splitting documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            splits = text_splitter.split_documents(pdf_data)
            logger.info(f"PDF split into {len(splits)} parts.")

            logger.info("Initializing embedding model...")
            try:
                self.embedding = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    cache_folder="/tmp/hf_cache"
                )
                logger.info("Embedding model loaded successfully.")
            except Exception as emb_error:
                logger.error(f"Error loading embedding model: {emb_error}")
                raise

            # Create vector database
            logger.info("Creating vector database...")
            self.vectordb = Chroma.from_documents(documents=splits, embedding=self.embedding)
            self.retriever = self.vectordb.as_retriever()

            logger.info("Vector database created successfully.")
        except Exception as e:
            logger.error(f"Error creating vector database: {e}")
            raise

    def load_prompt(self) -> None:
        """Load the prompt template for the chatbot."""
        self.prompt_str = """You are a virtual assistant for a Data Science platform called AI Studio. Answer the question based on the following context:
            {context}
            Question: {input}
            """
        self.prompt = ChatPromptTemplate.from_template(self.prompt_str)

    def load_chain(self) -> None:
        """Create the RAG chain using the loaded model, retriever, and prompt."""
        if not self.retriever:
            raise ValueError("Retriever must be initialized before creating the chain")

        input_normalizer = RunnableLambda(lambda x: {"input": x} if isinstance(x, str) else x)
        retriever_runnable = RunnableLambda(lambda x: self.retriever.get_relevant_documents(x["input"]))
        format_docs_r = RunnableLambda(format_docs)
        extract_input = RunnableLambda(lambda x: x["input"])

        self.chain = (
            input_normalizer
            | RunnableMap({
                "context": retriever_runnable | format_docs_r,
                "input": extract_input
            })
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def load_context(self, context) -> None:
        """
        Load context for the model, including configuration, model, vector database and chains.

        Args:
            context: MLflow model context
        """
        # Load configuration
        self.load_config(context)

        # Set up environment
        self.setup_environment()

        try:
            # Load model, vector database, prompt, and chain
            self.load_model(context)
            self.load_vector_database()
            self.load_prompt()
            self.load_chain()

            # Set up Galileo integration
            self.setup_protection()
            self.setup_monitoring()

            logger.info(f"{self.__class__.__name__} successfully loaded and configured.")
        except Exception as e:
            logger.error(f"Error loading context: {e}")
            raise

    def add_pdf(self, base64_pdf):
        """
        Add a new PDF to the vector database.

        Args:
            base64_pdf: Base64-encoded PDF content

        Returns:
            Dictionary with status information
        """
        try:
            pdf_bytes = base64.b64decode(base64_pdf)
            temp_pdf_path = f"/tmp/{uuid.uuid4()}.pdf"

            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_bytes)

            pdf_loader = PyPDFLoader(temp_pdf_path)
            pdf_data = pdf_loader.load()

            # Ensure all content is string type
            for doc in pdf_data:
                if not isinstance(doc.page_content, str):
                    doc.page_content = str(doc.page_content)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            new_splits = text_splitter.split_documents(pdf_data)

            # Create a new vector database with the new document
            embedding = HuggingFaceEmbeddings()
            vectordb = Chroma.from_documents(documents=new_splits, embedding=embedding)
            self.retriever = vectordb.as_retriever()

            # Clean up temporary file
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": "New PDF successfully added to the knowledge base.",
                "success": True
            }
        except Exception as e:
            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": f"Error adding PDF: {str(e)}",
                "success": False
            }

    def get_prompt_template(self):
        """
        Get the current prompt template.

        Returns:
            Dictionary containing the prompt template
        """
        return {
            "chunks": [],
            "history": [],
            "prompt": self.prompt_str,
            "output": "",
            "success": True
        }

    def set_prompt_template(self, new_prompt):
        """
        Update the prompt template.

        Args:
            new_prompt: New prompt template string

        Returns:
            Dictionary with status information
        """
        try:
            self.prompt_str = new_prompt
            self.prompt = ChatPromptTemplate.from_template(self.prompt_str)

            # Rebuild the chain with the new prompt
            self.load_chain()
            self.setup_protection()

            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": "Prompt template updated successfully.",
                "success": True
            }
        except Exception as e:
            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": f"Error updating prompt template: {str(e)}",
                "success": False
            }

    def reset_history(self):
        """
        Reset the conversation history.

        Returns:
            Dictionary with status information
        """
        self.memory = []
        return {
            "chunks": [],
            "history": [],
            "prompt": self.prompt_str,
            "output": "Conversation history has been reset.",
            "success": True
        }
    
    def inference(self, context, user_query):
        """
        Process a user query and generate a response.
        
        Args:
            context: MLflow model context
            user_query: User's query string
            
        Returns:
            Dictionary with response data
        """
        try:
            # Run the query through the protected chain with monitoring
            response = self.protected_chain.invoke(
                {"input": user_query, "output": ""},
                config={"callbacks": [self.monitor_handler]}
            )
            
            # Get relevant documents used in the response
            relevant_docs = self.retriever.get_relevant_documents(user_query)
            chunks = [doc.page_content for doc in relevant_docs]
            
            # Update conversation history
            self.memory.append({"role": "User", "content": user_query})
            self.memory.append({"role": "Assistant", "content": response})
            
            return {
                "chunks": chunks,
                "history": [f"<{m['role']}> {m['content']}\n" for m in self.memory],
                "prompt": self.prompt_str,
                "output": response,
                "success": True
            }
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": error_msg,
                "success": False
            }
    
    def predict(self, context, model_input, params=None):
        """
        Process inputs and generate appropriate responses based on parameters.
        
        Args:
            context: MLflow model context
            model_input: Input data dictionary with keys like 'query', 'prompt', 'document'
            params: Dictionary of parameters to control behavior
            
        Returns:
            Pandas DataFrame with the response data
        """
        if params is None:
            params = {}
            
        try:
            if params.get("add_pdf", False) and 'document' in model_input:
                result = self.add_pdf(model_input['document'][0])
            elif params.get("get_prompt", False):
                result = self.get_prompt_template()
            elif params.get("set_prompt", False) and 'prompt' in model_input:
                result = self.set_prompt_template(model_input['prompt'][0])
            elif params.get("reset_history", False):
                result = self.reset_history()
            elif 'query' in model_input:
                result = self.inference(context, model_input['query'][0])
            else:
                result = {
                    "chunks": [],
                    "history": [],
                    "prompt": self.prompt_str,
                    "output": "No valid input or operation specified.",
                    "success": False
                }
        except Exception as e:
            result = {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str if hasattr(self, 'prompt_str') else "",
                "output": f"Error processing request: {str(e)}",
                "success": False
            }
            
        return pd.DataFrame([result])

    @classmethod
    def log_model(cls, secrets_path, config_path, docs_path, model_path=None, demo_folder=None):
        """
        Log the model to MLflow.
        
        Args:
            secrets_path: Path to the secrets file
            config_path: Path to the configuration file
            docs_path: Path to the documents directory
            model_path: Path to the model file (optional)
            demo_folder: Path to the demo folder (optional)
            
        Returns:
            None
        """
        import mlflow
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec, ParamSchema, ParamSpec
        
        # Create demo folder if specified and doesn't exist
        if demo_folder and not os.path.exists(demo_folder):
            os.makedirs(demo_folder, exist_ok=True)
        
        # Define model input/output schema
        input_schema = Schema([
            ColSpec("string", "query"),
            ColSpec("string", "prompt"),
            ColSpec("string", "document")
        ])
        output_schema = Schema([
            ColSpec("string", "chunks"),
            ColSpec("string", "history"),
            ColSpec("string", "prompt"),
            ColSpec("string", "output"),
            ColSpec("boolean", "success")
        ])
        param_schema = ParamSchema([
            ParamSpec("add_pdf", "boolean", False),
            ParamSpec("get_prompt", "boolean", False),
            ParamSpec("set_prompt", "boolean", False),
            ParamSpec("reset_history", "boolean", False)
        ])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=param_schema)
        
        # Prepare artifacts
        artifacts = {
            "secrets": secrets_path, 
            "config": config_path, 
            "docs": docs_path
        }
        
        if demo_folder:
            artifacts["demo"] = demo_folder
        
        if model_path:
            artifacts["models"] = model_path
        
        # Log model to MLflow
        mlflow.pyfunc.log_model(
            artifact_path="chatbot_service",
            python_model=cls(),
            artifacts=artifacts,
            signature=signature,
            code_paths=["../core", "../../src"],
            pip_requirements=[
                "PyPDF",
                "pyyaml",
                "tokenizers>=0.13.0",
                "httpx>=0.24.0",
            ]
        )
        logger.info("Model and artifacts successfully registered in MLflow.")