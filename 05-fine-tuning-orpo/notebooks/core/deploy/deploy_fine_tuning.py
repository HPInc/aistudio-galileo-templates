import os
import mlflow
import torch
import logging
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from mlflow.types import Schema, ColSpec
from mlflow.models import ModelSignature

# Setup logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMComparisonModel(mlflow.pyfunc.PythonModel):
    """
    A MLflow-compatible PythonModel for serving and comparing two LLMs:
    - A base model (pretrained, no fine-tuning)
    - A fine-tuned model (adapted for a specific task)

    The model dynamically switches between the two depending on the input request,
    enabling side-by-side evaluation or production routing.
    """

    def load_context(self, context):
        """
        Loads both the base and fine-tuned models along with their tokenizers from provided artifacts.

        Args:
            context (mlflow.pyfunc.PythonModelContext): MLflow context object containing artifact paths.
        """
        self.model_base_path = context.artifacts["model_no_finetuning"]
        self.model_finetuned_path = context.artifacts["finetuned_model"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logging.info(f"Loading models onto device: {self.device}")

        # Load base model and tokenizer
        self.tokenizer_base = AutoTokenizer.from_pretrained(self.model_base_path)
        self.model_base = AutoModelForCausalLM.from_pretrained(self.model_base_path).to(self.device)

        # Load fine-tuned model and tokenizer
        self.tokenizer_ft = AutoTokenizer.from_pretrained(self.model_finetuned_path)
        self.model_ft = AutoModelForCausalLM.from_pretrained(self.model_finetuned_path).to(self.device)

        # Initialize placeholders for dynamic switching
        self.current_model = None
        self.current_tokenizer = None

    def load_model(self, use_finetuning: bool):
        """
        Sets the current model and tokenizer based on whether fine-tuning is requested.

        Args:
            use_finetuning (bool): If True, load the fine-tuned model; otherwise, load the base model.
        """
        if use_finetuning:
            self.current_model = self.model_ft
            self.current_tokenizer = self.tokenizer_ft
            logging.info("Switched to fine-tuned model.")
        else:
            self.current_model = self.model_base
            self.current_tokenizer = self.tokenizer_base
            logging.info("Switched to base model.")

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a model output based on the selected model (base or fine-tuned).

        Args:
            context (Unused): MLflow context.
            model_input (pd.DataFrame): Input DataFrame containing:
                - 'prompt' (str): The input text.
                - 'use_finetuning' (bool): Whether to use the fine-tuned model.
                - 'max_tokens' (int, optional): Maximum number of tokens to generate.

        Returns:
            pd.DataFrame: Output DataFrame with a single column 'response' containing the generated text.
        """
        prompt = model_input["prompt"].iloc[0] if isinstance(model_input["prompt"], pd.Series) else model_input["prompt"]
        use_finetuning = model_input["use_finetuning"].iloc[0] if isinstance(model_input["use_finetuning"], pd.Series) else model_input["use_finetuning"]
        max_tokens = model_input.get("max_tokens", 128)
        if isinstance(max_tokens, pd.Series):
            max_tokens = max_tokens.iloc[0]

        # Select the correct model
        self.load_model(use_finetuning)

        # Tokenize input
        inputs = self.current_tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate output without gradient tracking
        with torch.no_grad():
            output_ids = self.current_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.current_tokenizer.eos_token_id
            )

        # Decode and return the output
        output_text = self.current_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logging.info(f"Inference completed: {output_text[:80]}...")
        return pd.DataFrame({"response": [output_text]})


def register_llm_comparison_model(
    model_base_path: str,
    model_finetuned_path: str,
    experiment_name: str = "LLM-Finetune-Comparison",
    run_name: str = "llm_serving",
    model_name: str = "LLMComparisonService"
):
    """
    Logs and registers a MLflow PyFunc model capable of serving both base and fine-tuned LLMs.

    Args:
        model_base_path (str): Path to the pretrained base model (no fine-tuning).
        model_finetuned_path (str): Path to the fine-tuned model.
        experiment_name (str, optional): Name of the MLflow experiment. Default is "LLM-Finetune-Comparison".
        run_name (str, optional): Name of the MLflow run. Default is "llm_serving".
        model_name (str, optional): Name to register the model under in MLflow. Default is "LLMComparisonService".

    Returns:
        None
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        logging.info(f"Started MLflow run: {run.info.run_id}")

        # Define input and output schemas
        input_schema = Schema([
            ColSpec("string", "prompt"),
            ColSpec("boolean", "use_finetuning"),
            ColSpec("integer", "max_tokens"),
        ])
        output_schema = Schema([
            ColSpec("string", "response")
        ])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # Artifacts to bundle the models
        artifacts = {
            "finetuned_model": model_finetuned_path,
            "model_no_finetuning": model_base_path,
        }

        # Log the model with MLflow
        mlflow.pyfunc.log_model(
            artifact_path="llm_serving_model",
            python_model=LLMComparisonModel(),
            artifacts=artifacts,
            signature=signature,
            pip_requirements=["torch", "transformers", "mlflow", "pandas"]
        )

        # Register the model in the MLflow Model Registry
        model_uri = f"runs:/{run.info.run_id}/llm_serving_model"
        mlflow.register_model(model_uri=model_uri, name=model_name)
        logging.info(f"✅ Model successfully registered as: {model_name}")
