"""
Base service class for AI Studio Galileo Templates.
This module provides the core functionality for all service classes,
including model loading, configuration, and integration with Galileo services.
"""

import os
import yaml
from typing import Dict, Any, Optional, List, Union
import mlflow
from mlflow.pyfunc import PythonModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser


class BaseGenerativeService(PythonModel):
    """Base class for all generative services in AI Studio Galileo Templates."""

    def __init__(self):
        """Initialize the base service with empty configuration."""
        self.model_config = {}
        self.llm = None
        self.chain = None
        self.protected_chain = None
        self.prompt = None
        self.callback_manager = None
        self.monitor_handler = None
        self.prompt_handler = None
        self.protect_tool = None

    def load_config(self, context) -> Dict[str, Any]:
        """
        Load configuration from context artifacts.
        
        Args:
            context: MLflow model context containing artifacts
            
        Returns:
            Dictionary containing the loaded configuration
        """
        config_path = os.path.join(context._artifacts_dir, "config")
        secrets_path = os.path.join(context._artifacts_dir, "secrets")
        
        # Load configuration
        if os.path.exists(config_path):
            with open(config_path) as file:
                config = yaml.safe_load(file)
        else:
            config = {}
            
        # Load secrets
        if os.path.exists(secrets_path):
            with open(secrets_path) as file:
                secrets = yaml.safe_load(file)
        else:
            secrets = {}
            
        # Merge configurations
        self.model_config = {
            "galileo_key": secrets.get("GALILEO_API_KEY", ""),
            "hf_key": secrets.get("HUGGINGFACE_API_KEY", ""),
            "galileo_url": config.get("galileo_url", "https://console.hp.galileocloud.io/"),
            "proxy": config.get("proxy", None),
            "model_source": config.get("model_source", "local"),
            "observe_project": f"{self.__class__.__name__}_Observations",
            "protect_project": f"{self.__class__.__name__}_Protection",
        }
        
        return self.model_config
    
    def setup_environment(self) -> None:
        """Configure environment variables based on loaded configuration."""
        # Configure proxy if needed
        if self.model_config["proxy"] is not None:
            os.environ["HTTPS_PROXY"] = self.model_config["proxy"]
            
        # Set up Galileo environment
        os.environ["GALILEO_API_KEY"] = self.model_config["galileo_key"]
        os.environ["GALILEO_CONSOLE_URL"] = self.model_config["galileo_url"]
    
    def load_model(self, context) -> None:
        """
        Load the appropriate model based on configuration.
        
        Args:
            context: MLflow model context containing artifacts
        """
        raise NotImplementedError("Each service must implement its own model loading logic")
    
    def load_prompt(self) -> None:
        """Load the prompt template for the service."""
        raise NotImplementedError("Each service must implement its own prompt loading logic")
    
    def load_chain(self) -> None:
        """Create the processing chain using the loaded model and prompt."""
        raise NotImplementedError("Each service must implement its own chain creation logic")
    
    def setup_protection(self) -> None:
        """Set up protection with Galileo Protect."""
        import galileo_protect as gp
        from galileo_protect import ProtectTool, ProtectParser, Ruleset
        
        # Create project and stage
        project = gp.create_project(self.model_config["protect_project"])
        project_id = project.id
        
        stage_name = f"{self.model_config['protect_project']}_stage"
        stage = gp.create_stage(name=stage_name, project_id=project_id)
        stage_id = stage.id
        
        # Create default ruleset for PII protection
        ruleset = Ruleset(
            rules=[
                {
                    "metric": "pii",
                    "operator": "contains",
                    "target_value": "ssn",
                },
            ],
            action={
                "type": "OVERRIDE",
                "choices": [
                    "Personal Identifiable Information detected in the model output. Sorry, I cannot answer that question."
                ]
            }
        )
        
        # Create protection tool
        self.protect_tool = ProtectTool(
            stage_id=stage_id,
            prioritized_rulesets=[ruleset],
            timeout=10
        )
        
        # Set up protection parser and chain
        protect_parser = ProtectParser(chain=self.chain)
        self.protected_chain = self.protect_tool | protect_parser.parser
    
    def setup_monitoring(self) -> None:
        """Set up monitoring with Galileo Observe."""
        from galileo_observe import GalileoObserveCallback
        
        self.monitor_handler = GalileoObserveCallback(
            project_name=self.model_config["observe_project"]
        )
    
    def setup_evaluation(self, scorers=None) -> None:
        """
        Set up evaluation with Galileo Prompt Quality.
        
        Args:
            scorers: List of scorer functions to use for evaluation
        """
        import promptquality as pq
        
        if scorers is None:
            scorers = [
                pq.Scorers.context_adherence_luna,
                pq.Scorers.correctness,
                pq.Scorers.toxicity,
                pq.Scorers.sexist
            ]
        
        self.prompt_handler = pq.GalileoPromptCallback(
            project_name=self.model_config["observe_project"],
            scorers=scorers
        )
    
    def load_context(self, context) -> None:
        """
        Load context for the model, including configuration, model, and chains.
        
        Args:
            context: MLflow model context
        """
        # Load configuration
        self.load_config(context)
        
        # Set up environment
        self.setup_environment()
        
        # Load model, prompt, and chain
        self.load_model(context)
        self.load_prompt()
        self.load_chain()
        
        # Set up Galileo integration
        self.setup_protection()
        self.setup_monitoring()
        self.setup_evaluation()
        
        print(f"{self.__class__.__name__} successfully loaded and configured.")
    
    def predict(self, context, model_input):
        """
        Make predictions using the loaded model.
        
        Args:
            context: MLflow model context
            model_input: Input data for prediction
            
        Returns:
            Model predictions
        """
        raise NotImplementedError("Each service must implement its own prediction logic")