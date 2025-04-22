import os
import mlflow
import torch
import logging
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from mlflow.types import Schema, ColSpec
from mlflow.models import ModelSignature

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMComparisonModel(mlflow.pyfunc.PythonModel):
    """
    MLflow-compatible class for serving and comparing two LLMs:
    - A base model (no fine-tuning)
    - A fine-tuned version

    This model can dynamically switch between the two based on input flags,
    allowing side-by-side evaluation or production routing.
    """

    def load_context(self, context):
        """
        Loads both the base and fine-tuned models from provided artifacts.

        Args:
            context (mlflow.pyfunc.PythonModelContext): MLflow context object containing model paths.
        """
        self.model_base_path = context.artifacts["model_no_finetuning"]
        self.model_finetuned_path = context.artifacts["finetuned_model"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logging.info(f"Loading models to device: {self.device}")

        self.tokenizer_base = AutoTokenizer.from_pretrained(self.model_base_path)
        self.model_base = AutoModelForCausalLM.from_pretrained(self.model_base_path).to(self.device)

        self.tokenizer_ft = AutoTokenizer.from_pretrained(self.model_finetuned_path)
        self.model_ft = AutoModelForCausalLM.from_pretrained(self.model_finetuned_path).to(self.device)

        self.current_model = None
        self.current_tokenizer = None

    def load_model(self, use_finetuning):
        """
        Switches the current model and tokenizer depending on the flag.

        Args:
            use_finetuning (bool): Whether to use the fine-tuned model or the base model.
        """
        if use_finetuning:
            self.current_model = self.model_ft
            self.current_tokenizer = self.tokenizer_ft
            logging.info("Using fine-tuned model.")
        else:
            self.current_model = self.model_base
            self.current_tokenizer = self.tokenizer_base
            logging.info("Using base model.")

    def predict(self, context, model_input):
        """
        Generates a response using either the base or fine-tuned model.

        Args:
            context (Unused): MLflow context.
            model_input (pd.DataFrame): DataFrame with columns:
                - prompt (str)
                - use_finetuning (bool)
                - max_tokens (int, optional)

        Returns:
            pd.DataFrame: Single-column DataFrame with the generated response.
        """
        prompt = model_input["prompt"].iloc[0] if isinstance(model_input["prompt"], pd.Series) else model_input["prompt"]
        use_finetuning = model_input["use_finetuning"].iloc[0] if isinstance(model_input["use_finetuning"], pd.Series) else model_input["use_finetuning"]
        max_tokens = model_input.get("max_tokens", 128)
        if isinstance(max_tokens, pd.Series):
            max_tokens = max_tokens.iloc[0]

        self.load_model(use_finetuning)

        inputs = self.current_tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.current_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.current_tokenizer.eos_token_id
            )

        output_text = self.current_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logging.info(f"Inference done: {output_text[:80]}...")
        return pd.DataFrame({"response": [output_text]})


def register_llm_comparison_model(
    model_base_path,
    model_finetuned_path,
    experiment_name="LLM-Finetune-Comparison",
    run_name="llm_serving",
    model_name="LLMComparisonService"
):
    """
    Logs and registers a MLflow model capable of serving both base and fine-tuned LLMs.

    Args:
        model_base_path (str): Path to the base (non-finetuned) model.
        model_finetuned_path (str): Path to the fine-tuned model.
        experiment_name (str): Name of the MLflow experiment.
        run_name (str): Name of the run.
        model_name (str): Registered model name.

    Returns:
        None
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        logging.info(f"MLflow Run started: {run.info.run_id}")

        # Define input/output schema
        input_schema = Schema([
            ColSpec("string", "prompt"),
            ColSpec("boolean", "use_finetuning"),
            ColSpec("integer", "max_tokens"),
        ])
        output_schema = Schema([ColSpec("string", "response")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # Define artifacts
        artifacts = {
            "finetuned_model": model_finetuned_path,
            "model_no_finetuning": model_base_path,
        }

        # Log the model
        mlflow.pyfunc.log_model(
            artifact_path="llm_serving_model",
            python_model=LLMComparisonModel(),
            artifacts=artifacts,
            signature=signature,
            pip_requirements=["torch", "transformers", "mlflow", "pandas"]
        )

        # Register the model
        model_uri = f"runs:/{run.info.run_id}/llm_serving_model"
        mlflow.register_model(model_uri=model_uri, name=model_name)
        logging.info(f"✅ Model registered: {model_name}")
