"""
Utility functions for AI Studio Galileo Templates.

This module contains common functions used across notebooks in the project,
including configuration loading, model initialization, and Galileo integration.
"""

import os
import yaml
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple


def configure_hf_cache(cache_dir: str = "/home/jovyan/local/hugging_face") -> None:
    """
    Configure HuggingFace cache directories to persist models locally.

    Args:
        cache_dir: Base directory for HuggingFace cache. Defaults to "/home/jovyan/local/hugging_face".
    """
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")


def load_config_and_secrets(
    config_path: str = "../../configs/config.yaml",
    secrets_path: str = "../../configs/secrets.yaml"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load configuration and secrets from YAML files.

    Args:
        config_path: Path to the configuration YAML file.
        secrets_path: Path to the secrets YAML file.

    Returns:
        Tuple containing (config, secrets) as dictionaries.

    Raises:
        FileNotFoundError: If either the config or secrets file is not found.
    """
    # Convert to absolute paths if needed
    config_path = os.path.abspath(config_path)
    secrets_path = os.path.abspath(secrets_path)

    if not os.path.exists(secrets_path):
        raise FileNotFoundError(f"secrets.yaml file not found in path: {secrets_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml file not found in path: {config_path}")

    with open(config_path) as file:
        config = yaml.safe_load(file)

    with open(secrets_path) as file:
        secrets = yaml.safe_load(file)

    return config, secrets


def configure_proxy(config: Dict[str, Any]) -> None:
    """
    Configure proxy settings based on provided configuration.

    Args:
        config: Configuration dictionary that may contain a "proxy" key.
    """
    if "proxy" in config and config["proxy"]:
        os.environ["HTTPS_PROXY"] = config["proxy"]


def initialize_llm(
    model_source: str = "local",
    secrets: Optional[Dict[str, Any]] = None,
    local_model_path: str = "/home/jovyan/datafabric/llama2-7b/ggml-model-f16-Q5_K_M.gguf"
) -> Any:
    """
    Initialize a language model based on specified source.

    Args:
        model_source: Source of the model. Options are "local", "hugging-face-local", or "hugging-face-cloud".
        secrets: Dictionary containing API keys for cloud services.
        local_model_path: Path to local model file.

    Returns:
        Initialized language model object.

    Raises:
        ImportError: If required libraries are not installed.
        ValueError: If an unsupported model_source is provided.
    """
    # Check dependencies
    missing_deps = []
    for module in ["langchain_huggingface", "langchain_core.callbacks", "langchain_community.llms"]:
        if not importlib.util.find_spec(module):
            missing_deps.append(module)
    
    if missing_deps:
        raise ImportError(f"Missing required dependencies: {', '.join(missing_deps)}")
    
    # Import required libraries
    from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
    from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
    from langchain_community.llms import LlamaCpp

    # Initialize based on model source
    if model_source == "hugging-face-cloud":
        if not secrets or "HUGGINGFACE_API_KEY" not in secrets:
            raise ValueError("HuggingFace API key is required for cloud model access")
            
        huggingfacehub_api_token = secrets["HUGGINGFACE_API_KEY"]
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
        return HuggingFaceEndpoint(
            huggingfacehub_api_token=huggingfacehub_api_token,
            repo_id=repo_id,
        )
    elif model_source == "hugging-face-local":
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, device=0)
        return HuggingFacePipeline(pipeline=pipe)
    elif model_source == "local":
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        return LlamaCpp(
            model_path=local_model_path,
            n_gpu_layers=30,
            n_batch=512,
            n_ctx=4096,
            max_tokens=1024,
            f16_kv=True,
            callback_manager=callback_manager,
            verbose=False,
            stop=[],
            streaming=False,
            temperature=0.2,
        )
    else:
        raise ValueError(f"Unsupported model source: {model_source}")


def setup_galileo_environment(secrets: Dict[str, Any], console_url: str = "https://console.hp.galileocloud.io/") -> None:
    """
    Configure environment variables for Galileo services.

    Args:
        secrets: Dictionary containing the Galileo API key.
        console_url: URL for the Galileo console.

    Raises:
        ValueError: If Galileo API key is not found in secrets.
    """
    if "GALILEO_API_KEY" not in secrets:
        raise ValueError("Galileo API key not found in secrets")
    
    os.environ['GALILEO_API_KEY'] = secrets["GALILEO_API_KEY"]
    os.environ['GALILEO_CONSOLE_URL'] = console_url


def initialize_galileo_protect(project_name: str, stage_name: Optional[str] = None) -> Tuple[Any, str, str]:
    """
    Initialize Galileo Protect project and stage.

    Args:
        project_name: Name for the Galileo Protect project.
        stage_name: Optional name for the stage. If None, uses "{project_name}_stage".

    Returns:
        Tuple containing (project object, project_id, stage_id).

    Raises:
        ImportError: If galileo_protect is not installed.
    """
    try:
        import galileo_protect as gp
    except ImportError:
        raise ImportError("galileo_protect is required but not installed. Install it with pip install galileo_protect")
    
    if stage_name is None:
        stage_name = f"{project_name}_stage"
    
    project = gp.create_project(project_name)
    project_id = project.id
    
    stage = gp.create_stage(name=stage_name, project_id=project_id)
    stage_id = stage.id
    
    return project, project_id, stage_id


def initialize_galileo_evaluator(project_name: str, scorers: Optional[List] = None):
    """
    Initialize a Galileo Prompt Callback for evaluation.

    Args:
        project_name: Name for the evaluation project.
        scorers: List of scorers to use. If None, uses default scorers.

    Returns:
        Galileo prompt callback object.

    Raises:
        ImportError: If promptquality is not installed.
    """
    try:
        import promptquality as pq
    except ImportError:
        raise ImportError("promptquality is required but not installed")

    if scorers is None:
        scorers = [
            pq.Scorers.context_adherence_luna,
            pq.Scorers.correctness,
            pq.Scorers.toxicity,
            pq.Scorers.sexist
        ]

    return pq.GalileoPromptCallback(
        project_name=project_name,
        scorers=scorers
    )


def initialize_galileo_observer(project_name: str):
    """
    Initialize a Galileo Observer for monitoring.

    Args:
        project_name: Name for the observation project.

    Returns:
        Galileo observe callback object.

    Raises:
        ImportError: If galileo_observe is not installed.
    """
    try:
        from galileo_observe import GalileoObserveCallback
    except ImportError:
        raise ImportError("galileo_observe is required but not installed")
    
    return GalileoObserveCallback(project_name=project_name)