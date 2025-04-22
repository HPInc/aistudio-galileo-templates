import os
import torch
import logging
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import promptquality as pq
from core.finetuning_inference.inference_runner import AcceleratedInferenceRunner


class GalileoLocalComparer:
    """
    A class that compares base and fine-tuned model responses and logs evaluation results to Galileo.
    Uses PromptQuality with Galileo callback and integrates with AcceleratedInferenceRunner for inference.
    """

    def __init__(
        self,
        base_model_path: str,
        finetuned_path: str,
        prompts: list[str],
        galileo_project_name: str,
        galileo_url: str,
        dtype=torch.float16
    ):
        self.prompts = prompts
        self.project_name = galileo_project_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype

        os.environ["GALILEO_CONSOLE_URL"] = galileo_url
        pq.login(galileo_url)

        # Initialize both runners
        self.runner_base = AcceleratedInferenceRunner(base_model_path=base_model_path, dtype=dtype)
        self.runner_ft = AcceleratedInferenceRunner(base_model_path=base_model_path, finetuned_path=finetuned_path, dtype=dtype)

        # Load models beforehand for speed
        self.runner_base.load_model()
        self.runner_ft.load_model()

        # Setup Galileo evaluation handler
        self.prompt_handler = pq.GalileoPromptCallback(
            project_name=self.project_name,
            scorers=[
                pq.Scorers.correctness,
                pq.Scorers.context_adherence_luna,
                pq.Scorers.instruction_adherence_plus,
                pq.Scorers.chunk_attribution_utilization_luna,
            ]
        )

    def compare(self):
        """
        Runs inference for all prompts using both base and fine-tuned models, logging both results to Galileo.
        """
        batch_payload = []

        for idx, prompt in enumerate(self.prompts):
            print(f"⚙️ Running prompt {idx + 1}/{len(self.prompts)}")
            response_base = self.runner_base.infer(prompt)
            response_ft = self.runner_ft.infer(prompt)

            # Log both to Galileo with distinct metadata
            batch_payload.append({
                "input": prompt,
                "output": response_base,
                "model": "BASE_MODEL_LOCAL",
                "metadata": {"example_id": idx, "type": "base"},
            })

            batch_payload.append({
                "input": prompt,
                "output": response_ft,
                "model": "FINETUNED_MODEL_LOCAL",
                "metadata": {"example_id": idx, "type": "fine-tuned"},
            })

        # Log in batch using PromptQuality
        self.prompt_handler.batch(batch_payload)
        self.prompt_handler.finish()

        print("✅ Finished logging both models to Galileo.")
