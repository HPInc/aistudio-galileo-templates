"""
Text Summarization Service implementation that extends the BaseGenerativeService.

This service provides text summarization capabilities using different LLM options
and integrates with Galileo for protection, observation, and evaluation.
"""

import os
from typing import Dict, Any, Union
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from galileo_protect import ProtectParser

# Import base service class from the shared location
import sys
import os

# Add the src directory to the path to import base_service
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))
from service.base_service import BaseGenerativeService

class TextSummarizationService(BaseGenerativeService):
    """Text Summarization Service that extends the BaseGenerativeService."""

    def load_model(self, context) -> None:
        """
        Load the appropriate model based on configuration.
        
        Args:
            context: MLflow model context containing artifacts
        """
        model_source = self.model_config["model_source"]
        
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
        print(f"[INFO] Initializing local LlamaCpp model in {context.artifacts['model']}.")
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = LlamaCpp(
            model_path=context.artifacts["model"],
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
        print("Using the local LlamaCpp model.")
    
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
        print("Using the local Deep Seek model downloaded from HuggingFace.")

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
        print("Using the cloud Mistral model on HuggingFace.")
    
    def load_prompt(self) -> None:
        """Load the prompt template for text summarization."""
        self.prompt_str = '''
            The following text is an excerpt of a transcription:

            ### 
            {context} 
            ###

            Please, summarize this transcription, in a concise and comprehensive manner.
            '''
        self.prompt = ChatPromptTemplate.from_template(self.prompt_str)

    def load_chain(self) -> None:
        """Create the summarization chain using the loaded model and prompt."""
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def predict(self, context, model_input):
        """
        Generate a summary from the input text.
        
        Args:
            context: MLflow model context
            model_input: Input data for summarization, expecting a "text" field
            
        Returns:
            Dictionary with the summary in a "summary" field
        """
        text = model_input["text"][0]
        try:
            # Run the input through the protection chain with monitoring
            result = self.protected_chain.invoke(
                {"input": text, "output": ""},
                config={"callbacks": [self.monitor_handler]}
            )
            print("Successfully processed summarization request.")
        except Exception as e:
            result = f"Error processing request: {e}"
            print(result)
        
        # Return the result as a DataFrame with a summary column
        return pd.DataFrame([{"summary": result}])
        
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
            ColSpec("string", "text")
        ])
        output_schema = Schema([
            ColSpec("string", "summary")
        ])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        # Prepare artifacts
        artifacts = {
            "secrets": secrets_path,
            "config": config_path
        }
        
        if model_path:
            artifacts["model"] = model_path
        
        # Log model to MLflow
        mlflow.pyfunc.log_model(
            artifact_path="text_summarization_service",
            python_model=cls(),
            artifacts=artifacts,
            signature=signature,
            pip_requirements=[
                "galileo-protect==0.15.1",
                "galileo-observe==1.13.2",
                "pyyaml",
                "pandas",
                "sentence-transformers",
                "langchain_core",
                "langchain_huggingface",
                "tokenizers>=0.13.0",
                "httpx>=0.24.0",
            ]
        )
        print("Model and artifacts successfully registered in MLflow.")