import torch
from transformers import BitsAndBytesConfig

class QuantizationSelector:
    """
    Automatically selects the best quantization strategy (4bit or 8bit)
    based on GPU availability and memory.
    """

    def __init__(self, vram_threshold_8bit=30, min_gpus_for_8bit=2):
        self.vram_threshold_8bit = vram_threshold_8bit  # GB
        self.min_gpus_for_8bit = min_gpus_for_8bit
        self.num_gpus = torch.cuda.device_count()
        self.vram_list = self._get_gpu_memory_list()

    def _get_gpu_memory_list(self):
        return [
            torch.cuda.get_device_properties(i).total_memory / 1024**3
            for i in range(self.num_gpus)
        ]

    def _is_eligible_for_8bit(self):
        return self.num_gpus >= self.min_gpus_for_8bit and min(self.vram_list) >= self.vram_threshold_8bit

    def get_config(self) -> BitsAndBytesConfig:
        if self._is_eligible_for_8bit():
            print("✅ Using 8-bit quantization (multi-GPU & high VRAM).")
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
        else:
            print("⚠️ Using 4-bit quantization (fallback for low resources).")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
