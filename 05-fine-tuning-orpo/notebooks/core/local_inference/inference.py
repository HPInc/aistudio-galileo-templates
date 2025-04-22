import os
import torch
import yaml
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

class InferenceRunner:
    """
    A utility class for loading a language model from a local snapshot path
    and performing inference with GPU/CPU-aware configuration.

    This class automatically detects available GPU resources and loads the appropriate
    Accelerate configuration. It supports inference using models downloaded via
    `ModelSelector`, allowing seamless switching between models.
    """

    def __init__(self, model_selector, config_dir="config"):
        """
        Initializes the inference runner.

        Args:
            model_selector (ModelSelector): An instance of ModelSelector containing the selected model_id.
            config_dir (str): Path to the directory containing accelerate config YAMLs.
        """
        self.model_selector = model_selector
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config_dir = config_dir

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("InferenceRunner")

        self.config = self.load_optimal_config()

    def log(self, message):
        """
        Utility method to log messages with context-specific prefix.
        """
        self.logger.info(f"[InferenceRunner] {message}")

    def load_optimal_config(self):
        """
        Loads the appropriate accelerate configuration file depending on
        how many GPUs are available.

        Returns:
            dict: Parsed YAML configuration content.
        """
        num_gpus = torch.cuda.device_count()

        if num_gpus >= 2:
            config_file = os.path.join(self.config_dir, "default_config_multi-gpu.yaml")
            self.log(f"Detected {num_gpus} GPUs, loading {config_file}")
        elif num_gpus == 1:
            config_file = os.path.join(self.config_dir, "default_config_one-gpu.yaml")
            self.log(f"Detected 1 GPU, loading {config_file}")
        else:
            config_file = os.path.join(self.config_dir, "cpu_config.yaml")
            self.log(f"No GPU detected, loading {config_file}")

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def load_model_from_snapshot(self):
        """
        Loads the model and tokenizer from the path resolved by the selected model ID.

        Raises:
            RuntimeError: If model or tokenizer loading fails.
        """
        model_path = self.model_selector.format_model_path(self.model_selector.model_id)
        self.log(f"Loading model and tokenizer from snapshot at {model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model/tokenizer: {e}")

    def infer(self, prompt, max_new_tokens=100, temperature=0.7):
        """
        Runs a single inference step on the given prompt using the selected model.

        Args:
            prompt (str): Input prompt text.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.

        Returns:
            str: Decoded text output from the model.
        """
        if self.model is None or self.tokenizer is None:
            self.load_model_from_snapshot()

        self.log(f"Running inference on input: {prompt}")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.log("Inference completed.")
        return decoded
