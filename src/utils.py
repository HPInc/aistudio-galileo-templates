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
        repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
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
            n_gpu_layers=-1,
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
    
def login_huggingface(secrets: Dict[str, Any]) -> None:
    """
    Login to Hugging Face using token from secrets.

    Args:
        secrets: Dictionary containing the Hugging Face token.

    Raises:
        ValueError: If the token is missing.
    """
    from huggingface_hub import login

    token = secrets.get("HUGGINGFACE_API_KEY")
    if not token:
        raise ValueError("❌ Hugging Face token not found in secrets.yaml.")
    
    login(token=token)
    print("✅ Logged into Hugging Face successfully.")


def clean_code(result: str) -> str:
    """
    Clean code extraction function that handles various formats.
    
    Args:
        result: The raw text output from an LLM that may contain code.
        
    Returns:
        str: Cleaned code without markdown formatting or explanatory text.
    """
    if not result or not isinstance(result, str):
        return ""
    
    # Remove common prefixes and wrapper text    
    prefixes = ["Answer:", "Expected Answer:", "Python code:", "Here's the code:", "My Response:", "Response:"]
    for prefix in prefixes:
        if result.lstrip().startswith(prefix):
            result = result.replace(prefix, "", 1)
    
    # Handle markdown code blocks
    if "```python" in result or "```" in result:
        # Extract code between markdown code blocks
        code_blocks = []
        in_code_block = False
        lines = result.split('\n')
        current_block = []
        
        for line in lines:
            if line.strip().startswith("```"):
                if in_code_block:
                    # End of block, add it to our list
                    code_blocks.append("\n".join(current_block))
                    current_block = []
                in_code_block = not in_code_block
                continue
            
            if in_code_block:
                current_block.append(line)
        
        if code_blocks:
            # Use the longest code block found
            result = max(code_blocks, key=len)
        else:
            # Fallback to simple replacement if block extraction fails
            result = result.replace("```python", "").replace("```", "")
    
    # Remove any remaining explanatory text before or after the code
    lines = result.split('\n')
    code_lines = []
    in_code_block = False
    
    # First, look for the first actual code line
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and (stripped.startswith('import ') or 
                       stripped.startswith('from ') or
                       stripped.startswith('def ') or
                       stripped.startswith('class ')):
            in_code_block = True
            lines = lines[i:]  # Start from this line
            break
    
    # Now process all the lines
    for line in lines:
        stripped = line.strip()
        # Skip empty lines at the beginning
        if not stripped and not code_lines:
            continue
            
        # Ignore lines that appear to be LLM "thinking" or explanations
        if any(text in stripped.lower() for text in ["here's", "i'll", "please provide", "this code will"]):
            if not any(code_indicator in stripped for code_indicator in ["import ", "def ", "class ", "="]):
                continue
                
        # If we see code-like content, include it
        if stripped and (stripped.startswith('import ') or 
                       stripped.startswith('from ') or
                       stripped.startswith('def ') or
                       stripped.startswith('class ') or
                       '=' in stripped or
                       stripped.startswith('#') or
                       '(' in stripped or
                       '.' in stripped and not stripped.endswith('.') or
                       stripped.startswith('with ') or
                       stripped.startswith('if ') or
                       stripped.startswith('for ') or
                       stripped.startswith('while ') or
                       stripped.startswith('@')):
            in_code_block = True
            code_lines.append(line)
        # Include indented lines or lines continuing code
        elif stripped and (in_code_block or line.startswith(' ') or line.startswith('\t')):
            code_lines.append(line)
    
    cleaned_code = '\n'.join(code_lines).strip()
    
    # One last check - if the cleaned code starts with text that looks like a response,
    # try to find the first actual code statement
    first_lines = cleaned_code.split('\n', 5)
    for i, line in enumerate(first_lines):
        if line.strip().startswith(('import ', 'from ', 'def ', 'class ')):
            if i > 0:
                cleaned_code = '\n'.join(first_lines[i:] + cleaned_code.split('\n')[5:])
            break
    
    return cleaned_code


def generate_code_with_retries(chain, example_input, callbacks=None, max_attempts=3, min_code_length=10):
    """
    Execute a chain with retry logic for empty or short responses.
    
    Args:
        chain: The LangChain chain to execute.
        example_input: Input dictionary with query and question.
        callbacks: Optional callbacks to pass to the chain.
        max_attempts: Maximum number of attempts before giving up.
        min_code_length: Minimum acceptable code length.
        
    Returns:
        tuple: (raw_output, clean_code_output)
    """
    import time
    
    attempts = 0
    output = None
    
    while attempts < max_attempts:
        attempts += 1
        try:
            # Add a small delay before each attempt (only needed for retries)
            if attempts > 1:
                time.sleep(1)  # Small delay between retries
                
            # Invoke the chain
            output = chain.invoke(
                example_input,
                config=dict(callbacks=callbacks) if callbacks else {}
            )
            
            # Clean the code
            clean_code_output = clean_code(output)
            
            # Only continue with retry if we got no usable output
            if clean_code_output and len(clean_code_output) > min_code_length:
                break
                
            print(f"Attempt {attempts}: Output too short or empty, retrying...")
            
        except Exception as e:
            print(f"Error in attempt {attempts}: {str(e)}")
            if attempts == max_attempts:
                raise
    
    return output, clean_code_output


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
