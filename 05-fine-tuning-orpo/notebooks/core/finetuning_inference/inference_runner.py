import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download


class AcceleratedInferenceRunner:
    def __init__(self, base_model_path, finetuned_path=None, dtype=torch.float16, max_memory=None, base_local_dir=None):
        self.original_model_id = base_model_path
        self.base_local_dir = base_local_dir or os.path.join("..", "..", "..", "local", "models")

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("AcceleratedInferenceRunner")  # ✅ mover isso antes

        self.base_model_path = self.resolve_model_path(base_model_path)
        self.finetuned_path = finetuned_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        self.max_memory = max_memory or self._auto_max_memory()
        self.model = None
        self.tokenizer = None


        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("AcceleratedInferenceRunner")

    def _auto_max_memory(self):
        gpu_count = torch.cuda.device_count()
        default_mem = "40GiB"
        return {i: default_mem for i in range(gpu_count)} if gpu_count > 0 else {"cpu": "64GiB"}

    def resolve_model_path(self, model_id):
        local_model_dir = os.path.join(self.base_local_dir, model_id.replace("/", "__"))
        if os.path.exists(local_model_dir):
            self.logger.info(f"📦 Using existing local model at: {local_model_dir}")
            return local_model_dir
        else:
            self.logger.info(f"⬇️ Downloading model from Hugging Face Hub: {model_id}")
            downloaded_path = snapshot_download(model_id, local_dir=local_model_dir, local_dir_use_symlinks=False)
            self.logger.info(f"✅ Model downloaded to: {downloaded_path}")
            return local_model_dir

    def load_model(self):
        self.logger.info("🔄 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.finetuned_path or self.base_model_path)

        self.logger.info("📦 Initializing model with Accelerate (empty weights)...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(self.base_model_path)

        self.logger.info("🧠 Inferring device map with memory limits: %s", self.max_memory)
        device_map = infer_auto_device_map(
            model,
            max_memory=self.max_memory,
            no_split_module_classes=["LlamaDecoderLayer"]
        )

        self.logger.info(f"🚀 Dispatching model with device_map: {device_map}")
        model = load_checkpoint_and_dispatch(
            model,
            self.base_model_path,
            device_map=device_map,
            dtype=self.dtype
        )

        if self.finetuned_path:
            self.logger.info("🎯 Applying LoRA fine-tuned weights...")
            model = PeftModel.from_pretrained(model, self.finetuned_path)

        self.model = model.eval()

    def infer(self, prompt, max_new_tokens=100, temperature=0.7):
        if self.model is None or self.tokenizer is None:
            self.load_model()

        self.logger.info(f"🔍 Running inference for prompt: {prompt[:80]}...")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.logger.info("✅ Inference complete.")
        return result
