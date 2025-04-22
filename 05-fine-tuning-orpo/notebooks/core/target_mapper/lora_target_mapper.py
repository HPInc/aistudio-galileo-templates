class LoRATargetMapper:
    """
    Utility class to map model types to their corresponding target modules
    for LoRA (Low-Rank Adaptation) fine-tuning.

    This allows the training script to dynamically choose the correct set
    of modules based on the base model being used (e.g., Mistral, LLaMA, Gemma).

    This is especially useful when performing QLoRA or PEFT-based finetuning
    across multiple architectures with shared transformer structures.
    """

    # Dictionary of model name keywords mapped to their target LoRA modules
    TARGET_MODULES_MAP = {
        "mistral": ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        "llama": ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        "gemma": ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    }

    @classmethod
    def get_target_modules(cls, model_id: str) -> list[str]:
        """
        Retrieve the target modules for LoRA injection based on the model ID.

        Args:
            model_id (str): The Hugging Face model ID (e.g., 'meta-llama/Llama-2-7b-chat-hf').

        Returns:
            list[str]: A list of module names to target for LoRA adaptation.

        Raises:
            ValueError: If no matching target modules are defined for the provided model_id.
        """
        for key in cls.TARGET_MODULES_MAP:
            if key in model_id.lower():
                return cls.TARGET_MODULES_MAP[key]
        raise ValueError(f"No LoRA target_modules defined for model: {model_id}")
