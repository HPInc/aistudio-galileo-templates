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
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")


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


def get_model_context_window(model) -> int:
    """
    Detect or estimate the context window size of a language model.
    
    This function systematically examines the model object to determine its context window size,
    using a comprehensive hierarchical approach:
    
    1. Try to access model-specific context window attributes directly
    2. Look for context window information in the model's configuration
    3. For pipeline models, inspect both the tokenizer and underlying model structure
    4. As a last resort, make an educated guess based on model name/type
    
    Args:
        model: Any language model object (LlamaCpp, HuggingFace, OpenAI, etc.)
        
    Returns:
        int: The determined context window size in tokens, defaulting to 2048 if detection fails
    """
    # === Direct attribute checks ===
    
    # Check for LlamaCpp models from LangChain
    if str(type(model)).find('LlamaCpp') > -1:
        # LlamaCpp models from LangChain store n_ctx in multiple possible places
        if hasattr(model, '_n_ctx') and model._n_ctx is not None:
            return model._n_ctx
        if hasattr(model, 'n_ctx') and model.n_ctx is not None:
            return model.n_ctx
    
    # Llama.cpp Python binding
    if hasattr(model, 'context_params') and hasattr(model.context_params, 'n_ctx'):
        return model.context_params.n_ctx
    
    # Common direct attributes
    for attr_name in ['max_seq_len', 'max_position_embeddings', 'n_positions', 'max_sequence_length']:
        if hasattr(model, attr_name):
            attr_value = getattr(model, attr_name)
            if isinstance(attr_value, int) and attr_value > 0:
                return attr_value
    
    # === Check model configuration ===
    
    # Access transformer models' configuration
    # This handles most HuggingFace models
    if hasattr(model, 'config'):
        config = model.config
        
        # Try common config attributes for context window
        for attr_name in [
            'max_position_embeddings', 'n_positions', 'n_ctx', 
            'max_sequence_length', 'context_length', 'model_max_length'
        ]:
            if hasattr(config, attr_name):
                attr_value = getattr(config, attr_name)
                if isinstance(attr_value, int) and attr_value > 0:
                    return attr_value
                
    # === Check for HuggingFace pipeline wrappers ===
    
    # HuggingFacePipeline wrapper from langchain
    if hasattr(model, 'pipeline'):
        pipeline = model.pipeline
        
        # Try to get from tokenizer in pipeline
        if hasattr(pipeline, 'tokenizer'):
            tokenizer = pipeline.tokenizer
            if hasattr(tokenizer, 'model_max_length'):
                if isinstance(tokenizer.model_max_length, int) and tokenizer.model_max_length > 0 and tokenizer.model_max_length != 1000000000000000019884624838656:  # Exclude infinity value
                    return tokenizer.model_max_length
        
        # Next, check pipeline.model configuration
        if hasattr(pipeline, 'model'):
            model_obj = pipeline.model
            
            # Check for config in model
            if hasattr(model_obj, 'config'):
                config = model_obj.config
                
                # Comprehensive check of all possible config attributes for context window
                for attr_name in [
                    'max_position_embeddings', 'n_positions', 'n_ctx', 
                    'max_sequence_length', 'context_length', 'model_max_length',
                    'n_positions', 'max_length', 'seq_length'
                ]:
                    if hasattr(config, attr_name):
                        attr_value = getattr(config, attr_name)
                        if isinstance(attr_value, int) and attr_value > 0:
                            return attr_value
            
            # For models without a normal config structure, check model attributes directly
            for attr_name in ['max_seq_len', 'n_ctx', 'n_positions', 'max_position_embeddings']:
                if hasattr(model_obj, attr_name):
                    attr_value = getattr(model_obj, attr_name)
                    if isinstance(attr_value, int) and attr_value > 0:
                        return attr_value
    
    # === Check model kwargs (initialization parameters) ===
    
    # Many wrappers store initialization parameters
    if hasattr(model, 'model_kwargs'):
        kwargs = model.model_kwargs
        for param_name in ['n_ctx', 'max_tokens', 'max_length', 'context_window']:
            if param_name in kwargs:
                if isinstance(kwargs[param_name], int) and kwargs[param_name] > 0:
                    return kwargs[param_name]
    
    # === Inspect actual model architecture for constants ===
    
    # For HuggingFace pipeline models, extract info from the underlying model architecture
    if hasattr(model, 'pipeline') and hasattr(model.pipeline, 'model'):
        try:
            # Try to extract useful constants from model configurations
            underlying_model = model.pipeline.model
            
            # Check the model config dictionary for relevant parameters
            if hasattr(underlying_model, 'config') and hasattr(underlying_model.config, 'to_dict'):
                config_dict = underlying_model.config.to_dict()
                
                # Look for parameters that might indicate context window size
                # These names cover most transformer model architectures
                for key in config_dict:
                    if isinstance(config_dict[key], int) and config_dict[key] > 1024:
                        key_lower = key.lower()
                        if ('context' in key_lower or 'window' in key_lower or 
                            'position' in key_lower or 'seq' in key_lower and 'len' in key_lower or
                            'embed' in key_lower and 'pos' in key_lower or 'n_ctx' in key_lower):
                            return config_dict[key]
        except:
            pass  # Continue to fallbacks if extraction fails
    
    # === String-based fallback (last resort) ===
    
    # Only use string matching as a last resort
    model_str = str(model).lower()
    
    # If the model is using LlamaCpp, default to 4096 tokens
    # (Check both class name and string representation)
    if "llamacpp" in model_str or "llama.cpp" in model_str:
        return 4096  # Common default for LlamaCpp models
    
    # Use a dictionary for known model families context windows
    model_families = {
        'llama-2': 4096,
        'llama-3': 8192,
        'mistral': 8192,
        'mixtral': 32768,
        'deepseek': 4096,
        'qwen': 8192,
        'falcon': 4096,
        'gpt-3.5': 16385,
        'gpt-4': 8192,
    }
    
    # Check for model family matches
    for family, ctx_size in model_families.items():
        if family in model_str:
            return ctx_size
    
    # Final fallback - conservative estimate
    return 2048


def dynamic_retriever(query: str, collection, top_n: int = None, context_window: int = None) -> List:
    """
    Retrieve relevant documents with dynamic adaptation based on context window.
    
    This function automatically determines how many documents to retrieve based on
    the available context window, optimizing for the specific model being used.
    
    Args:
        query: The search query
        collection: Vector database collection to search in
        top_n: Number of documents to retrieve (if None, will be determined dynamically)
        context_window: Size of the model's context window in tokens
        
    Returns:
        List: Document objects containing relevant content
    """
    from langchain.schema import Document
    
    # Dynamically determine how many documents to retrieve based on context window
    if top_n is None:
        if context_window:
            # Larger context windows can handle more documents
            # Using a heuristic: 1 document per 1000 tokens of context
            # with a minimum of 2 and maximum of 10
            suggested_top_n = max(2, min(10, context_window // 1000))
            top_n = suggested_top_n
        else:
            # Default if we can't determine context window
            top_n = 3
    
    # Get the most relevant documents
    results = collection.query(
        query_texts=[query],
        n_results=top_n
    )
    
    # Convert to Document objects
    documents = [
        Document(
            page_content=str(results['documents'][i]),
            metadata=results['metadatas'][i] if isinstance(results['metadatas'][i], dict) else results['metadatas'][i][0]  
        )
        for i in range(len(results['documents']))
    ]
    
    return documents


def format_docs_with_adaptive_context(docs, context_window: int = None) -> str:
    """
    Format retrieved documents using dynamic allocation based on model context window.
    
    This function:
    1. Adapts to the model's context window size
    2. Keeps full content for the most relevant document when possible
    3. Distributes remaining context based on document relevance
    4. Preserves code structure by breaking at logical points
    5. Provides diagnostics about context usage
    
    Args:
        docs: List of Document objects to format
        context_window: Size of the model's context window in tokens (if provided)
        
    Returns:
        Formatted context string for the LLM
    """
    if not docs:
        return ""
        
    # Average characters per token (this is an approximation)
    chars_per_token = 4
    
    # Determine the maximum character budget based on context window
    if context_window:
        # Reserve 20% for the prompt and response
        available_tokens = int(context_window * 0.8)
        max_total_chars = available_tokens * chars_per_token
    else:
        # Default conservative estimate if we don't know the context window
        max_total_chars = 8000
    
    # Track metrics for diagnostic output
    formatted_docs = []
    total_chars = 0
    doc_allocation = []
    
    # Process documents by relevance order
    for i, doc in enumerate(docs):
        content = doc.page_content
        original_length = len(content)
        
        # Distribute context budget based on relevance
        # First document gets up to 50% of remaining budget, but don't exceed its actual size
        if i == 0:
            # Give the first (most relevant) document up to 50% of the budget
            budget_fraction = 0.5
        else:
            # Distribute remaining budget exponentially declining by relevance
            budget_fraction = 0.5 / (2 ** i)
        
        chars_to_allocate = min(
            int(max_total_chars * budget_fraction),  # Relevance-based allocation
            original_length,  # Don't allocate more than needed
            max_total_chars - total_chars  # Don't exceed remaining budget
        )
        
        # If we can fit the whole document, do it
        if original_length <= chars_to_allocate:
            formatted_docs.append(content)
            used_chars = original_length
            truncated = False
        # Otherwise, truncate it
        elif chars_to_allocate > 0:
            # Try to break at a logical point like a line break
            truncation_point = min(chars_to_allocate, original_length)
            
            # Find a good break point - prefer newlines, then periods, then spaces
            last_newline = content[:truncation_point].rfind('\n')
            last_period = content[:truncation_point].rfind('.')
            last_space = content[:truncation_point].rfind(' ')
            
            # Use the best break point that's not too far from target (at least 80% of target)
            threshold = truncation_point * 0.8
            if last_newline > threshold:
                truncation_point = last_newline + 1  # +1 to include the newline
            elif last_period > threshold:
                truncation_point = last_period + 1  # +1 to include the period
            elif last_space > threshold:
                truncation_point = last_space + 1  # +1 to include the space
            
            formatted_content = f"{content[:truncation_point]}... (truncated)"
            formatted_docs.append(formatted_content)
            used_chars = truncation_point + 15  # +15 for the truncation message
            truncated = True
        else:
            # No budget left for this document
            break
            
        # Track allocation for diagnostic output
        doc_allocation.append({
            'document': i+1,
            'original_chars': original_length,
            'allocated_chars': used_chars,
            'truncated': truncated,
            'percent_used': round(100 * used_chars / original_length, 1) if original_length > 0 else 100
        })
        
        total_chars += used_chars
        
        # Stop if we've reached our budget
        if total_chars >= max_total_chars:
            break
    
    # Add diagnostic info about context usage
    estimated_tokens_used = total_chars // chars_per_token
    estimated_context_percent = round(100 * estimated_tokens_used / context_window, 1) if context_window else "unknown"
    
    diagnostic = f"/* Context usage: {estimated_tokens_used} tokens ({estimated_context_percent}% of context window) */\n"
    
    # Join everything together with clear separators
    formatted_text = diagnostic + "\n\n".join(formatted_docs)
            
    return formatted_text

def diagnose_model_context_window(model) -> Dict[str, Any]:
    """
    Diagnose model context window detection issues by examining model attributes.
    
    This helper function explores various attributes of a model object that might
    contain information about its context window size. It's useful for debugging
    why context window detection might be failing.
    
    Args:
        model: The model object to diagnose
        
    Returns:
        Dict with diagnosis information
    """
    diagnosis = {
        'model_type': str(type(model)),
        'detected_attributes': [],
        'recommended_context_window': None
    }
    
    # Check direct model attributes
    direct_attrs = {
        '_n_ctx': None, 'n_ctx': None, 'max_seq_len': None, 
        'max_position_embeddings': None, 'n_positions': None, 
        'max_sequence_length': None, 'context_length': None
    }
    for attr_name in direct_attrs:
        if hasattr(model, attr_name):
            attr_value = getattr(model, attr_name)
            direct_attrs[attr_name] = attr_value
            if isinstance(attr_value, int) and attr_value > 0:
                diagnosis['detected_attributes'].append(f"{attr_name}: {attr_value}")
                if not diagnosis['recommended_context_window']:
                    diagnosis['recommended_context_window'] = attr_value
    
    diagnosis['direct_attributes'] = direct_attrs
    
    # Check config
    config_data = {}
    if hasattr(model, 'config'):
        config = model.config
        config_data['has_config'] = True
        config_attrs = {
            'max_position_embeddings': None, 'n_positions': None, 
            'n_ctx': None, 'max_sequence_length': None, 
            'context_length': None, 'model_max_length': None
        }
        
        for attr_name in config_attrs:
            if hasattr(config, attr_name):
                attr_value = getattr(config, attr_name) 
                config_attrs[attr_name] = attr_value
                if isinstance(attr_value, int) and attr_value > 0:
                    diagnosis['detected_attributes'].append(f"config.{attr_name}: {attr_value}")
                    if not diagnosis['recommended_context_window']:
                        diagnosis['recommended_context_window'] = attr_value
    else:
        config_data['has_config'] = False
    
    diagnosis['config'] = config_data
    
    # Check pipeline
    pipeline_data = {}
    if hasattr(model, 'pipeline'):
        pipeline = model.pipeline
        pipeline_data['has_pipeline'] = True
        
        # Check tokenizer
        if hasattr(pipeline, 'tokenizer'):
            tokenizer = pipeline.tokenizer
            pipeline_data['has_tokenizer'] = True
            
            if hasattr(tokenizer, 'model_max_length'):
                pipeline_data['tokenizer_model_max_length'] = tokenizer.model_max_length
                diagnosis['detected_attributes'].append(f"pipeline.tokenizer.model_max_length: {tokenizer.model_max_length}")
                if not diagnosis['recommended_context_window'] and isinstance(tokenizer.model_max_length, int) and tokenizer.model_max_length > 0:
                    diagnosis['recommended_context_window'] = tokenizer.model_max_length
        else:
            pipeline_data['has_tokenizer'] = False
            
        # Check pipeline.model
        if hasattr(pipeline, 'model'):
            pipeline_model = pipeline.model
            pipeline_data['has_model'] = True
            
            # Check pipeline.model.config
            if hasattr(pipeline_model, 'config'):
                config = pipeline_model.config
                pipeline_data['model_has_config'] = True
                
                for attr_name in ['max_position_embeddings', 'n_positions', 'n_ctx', 
                                 'max_sequence_length', 'context_length', 'model_max_length']:
                    if hasattr(config, attr_name):
                        attr_value = getattr(config, attr_name)
                        pipeline_data[f"model_config_{attr_name}"] = attr_value
                        
                        if isinstance(attr_value, int) and attr_value > 0:
                            diagnosis['detected_attributes'].append(f"pipeline.model.config.{attr_name}: {attr_value}")
                            if not diagnosis['recommended_context_window']:
                                diagnosis['recommended_context_window'] = attr_value
            else:
                pipeline_data['model_has_config'] = False
        else:
            pipeline_data['has_model'] = False
    else:
        pipeline_data['has_pipeline'] = False
        
    diagnosis['pipeline'] = pipeline_data
    
    # Check model_kwargs
    kwargs_data = {}
    if hasattr(model, 'model_kwargs'):
        kwargs = model.model_kwargs
        kwargs_data['has_model_kwargs'] = True
        kwargs_data['model_kwargs'] = kwargs
        
        for param_name in ['n_ctx', 'max_tokens', 'max_length', 'context_window']:
            if param_name in kwargs:
                param_value = kwargs[param_name]
                if isinstance(param_value, int) and param_value > 0:
                    diagnosis['detected_attributes'].append(f"model_kwargs[{param_name}]: {param_value}")
                    if not diagnosis['recommended_context_window']:
                        diagnosis['recommended_context_window'] = param_value
    else:
        kwargs_data['has_model_kwargs'] = False
        
    diagnosis['model_kwargs'] = kwargs_data
    
    # String-based detection as last resort
    model_str = str(model).lower()
    
    # Look for specific model types in string representation
    for model_name, window_size in [
        ('llama-2-7b', 4096), ('llama-2-13b', 4096), ('llama-2-70b', 4096),
        ('llama-3', 8192), ('gpt-3.5', 16385), ('gpt-4', 8192),
        ('mistral', 8192), ('mixtral', 32768), ('deepseek', 4096),
        ('qwen', 8192), ('falcon', 4096)
    ]:
        if model_name in model_str:
            diagnosis['detected_attributes'].append(f"String match: '{model_name}' -> {window_size}")
            if not diagnosis['recommended_context_window']:
                diagnosis['recommended_context_window'] = window_size
    
    # Fallback recommendation if nothing was found
    if not diagnosis['recommended_context_window']:
        diagnosis['recommended_context_window'] = 2048  # Conservative fallback
        diagnosis['detected_attributes'].append("Using conservative fallback: 2048")
    
    return diagnosis
