from abc import ABC, abstractmethod
from typing import Any, Dict

from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class BaseLLMModel(ABC):
    """Base interface for all LLM models."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates a response based on the provided prompt."""
        pass


class LlamaCPPModel(BaseLLMModel):
    """
    Local model using LlamaCPP.

    It is mandatory to provide the model path via the model_path parameter.
    """
    def __init__(self, model_path: str, **kwargs: Any) -> None:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=kwargs.get("n_gpu_layers", 30),
            n_batch=kwargs.get("n_batch", 512),
            n_ctx=kwargs.get("n_ctx", 4096),
            max_tokens=kwargs.get("max_tokens", 1024),
            f16_kv=kwargs.get("f16_kv", True),
            callback_manager=callback_manager,
            verbose=kwargs.get("verbose", False),
            stop=kwargs.get("stop", []),
            streaming=kwargs.get("streaming", False),
            temperature=kwargs.get("temperature", 0.2),
        )

    def generate(self, prompt: str) -> str:
        return self.llm(prompt)


class HuggingFacePipelineModel(BaseLLMModel):
    """
    Model using HuggingFacePipeline via API.

    For this model, the user can provide an authentication token via the api argument
    (internally mapped to huggingfacehub_api_token). If not provided, the default model
    "mistralai/Mistral-7B-v0.1" will be used
    """
    def __init__(self, model_id: str = "mistralai/Mistral-7B-v0.1", **kwargs: Any) -> None:
        token = kwargs.pop("huggingfacehub_api_token", None)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=token)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=kwargs.get("max_new_tokens", 100),
            device=kwargs.get("device", 0),
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    def generate(self, prompt: str) -> str:
        return self.llm(prompt)


class HuggingFaceLocalModel(BaseLLMModel):
    """
    Local model using Hugging Face Transformers.

    Use this model to locally load a Hugging Face model.
    It is mandatory to provide the local model path via the model_path parameter (mapped to path in the interface).
    """
    def __init__(self, model_path: str, **kwargs: Any) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=kwargs.get("max_new_tokens", 100),
            device=kwargs.get("device", 0),
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    def generate(self, prompt: str) -> str:
        return self.llm(prompt)


class LLMManager:
    """
    Simplified LLM model manager.
    
    Allows instantiating different models easily.
    
    Usage examples:
    
      # Local LlamaCPP Model
      manager = LLMManager(model="llamacpp", path="/path/to/model")
      
      # API-based model (HuggingFacePipeline):
      manager = LLMManager(model="huggingfacepipeline", api="YOUR_TOKEN_HERE")
      
      # Local HuggingFace model:
      manager = LLMManager(model="huggingfacelocal", path="/path/to/model")
    """
    _MODELOS: Dict[str, Any] = {
        "llamacpp": LlamaCPPModel,
        "huggingfacepipeline": HuggingFacePipelineModel,
        "huggingfacelocal": HuggingFaceLocalModel,
    }

    def __init__(self, model: str, **kwargs: Any) -> None:
        model_key = model.lower()

        if model_key in ("llamacpp", "huggingfacelocal"):
            if "path" not in kwargs:
                raise ValueError(f"For the model '{model_key}' the 'path' parameter is mandatory.")
            kwargs["model_path"] = kwargs.pop("path")
        if "api" in kwargs:
            kwargs["huggingfacehub_api_token"] = kwargs.pop("api")

        if model_key not in self._MODELOS:
            raise ValueError(
                f"Model '{model}' is not supported. Available models: {list(self._MODELOS.keys())}"
            )
        self.modelo_instancia = self._MODELOS[model_key](**kwargs)

    def generate(self, prompt: str) -> str:
        """Generates a response using the instantiated model."""
        return self.modelo_instancia.generate(prompt)