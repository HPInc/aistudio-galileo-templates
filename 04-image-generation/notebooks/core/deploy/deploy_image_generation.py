import mlflow
import torch
import logging
import os
import subprocess
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64
from diffusers import StableDiffusionPipeline
from mlflow.types import Schema, ColSpec
from mlflow.models import ModelSignature

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageGenerationModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        logging.info("Loading model artifacts...")
        self.model_no_finetuning_path = context.artifacts["model_no_finetuning"]
        self.model_finetuning_path = context.artifacts["finetuned_model"]

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus >= 2:
            config_file = "../data/config/default_config_multi-gpu.yaml"
            logging.info(f"Detected {self.num_gpus} GPUs, using multi-GPU configuration: {config_file}")
        elif self.num_gpus == 1:
            config_file = "../data/config/default_config_one-gpu.yaml"
            logging.info(f"1 GPU detected, using single-GPU configuration: {config_file}")
        else:
            config_file = "../data/config/config/default_config-cpu.yaml"
            logging.info("No GPU detected, using CPU configuration.")
        self.current_pipeline = None
        self.current_model = None

    def load_pipeline(self, use_finetuning):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.current_model == "finetuning" and not use_finetuning:
            logging.info("Switching to the model without fine-tuning...")
            del self.current_pipeline
            torch.cuda.empty_cache()
            self.current_pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_no_finetuning_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
            ).to(device)
            self.current_model = "no_finetuning"
        elif self.current_model == "no_finetuning" and use_finetuning:
            logging.info("Switching to the finetuned model...")
            del self.current_pipeline
            torch.cuda.empty_cache()
            self.current_pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_finetuning_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
            ).to(device)
            self.current_model = "finetuning"
        elif self.current_pipeline is None:
            if use_finetuning:
                logging.info("Loading finetuned pipeline for the first time...")
                self.current_pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_finetuning_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
                ).to(device)
                self.current_model = "finetuning"
            else:
                logging.info("Loading pipeline without fine-tuning for the first time...")
                self.current_pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_no_finetuning_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
                ).to(device)
                self.current_model = "no_finetuning"

    def predict(self, context, model_input):
        prompt = (
            model_input["prompt"].iloc[0]
            if isinstance(model_input["prompt"], pd.Series)
            else model_input["prompt"]
        )
        use_finetuning = (
            model_input["use_finetuning"].iloc[0]
            if isinstance(model_input["use_finetuning"], pd.Series)
            else model_input["use_finetuning"]
        )
        height = (
            model_input.get("height", 512).iloc[0]
            if isinstance(model_input["height"], pd.Series)
            else model_input["height"]
        )
        width = (
            model_input.get("width", 512).iloc[0]
            if isinstance(model_input["width"], pd.Series)
            else model_input["width"]
        )
        num_images = (
            model_input.get("num_images", 1).iloc[0]
            if isinstance(model_input["num_images"], pd.Series)
            else model_input["num_images"]
        )
        num_inference_steps = (
            model_input.get("num_inference_steps", 100).iloc[0]
            if isinstance(model_input["num_inference_steps"], pd.Series)
            else model_input["num_inference_steps"]
        )

        logging.info("Starting inference with the prompt: %s", prompt)
        self.load_pipeline(use_finetuning)

        images = []
        with torch.no_grad():
            for i in range(num_images):
                logging.info("Running inference for the image %d/%d", i+1, num_images)
                image = self.current_pipeline(
                    prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=7.5
                ).images[0]

                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
                images.append(img_base64)
                image_filename = f"local_model_result_{i}.png"
                image.save(image_filename)
                logging.info("Image saved as:: %s", image_filename)

        logging.info("Inference completed.")
        return pd.DataFrame({"output_images": images})

    @classmethod
    def log_model(cls, finetuned_model_path, model_no_finetuning_path, artifact_path="image_generation_model"):
        logging.info("Starting model logging...")
        input_schema = Schema([
            ColSpec("string", "prompt"),
            ColSpec("boolean", "use_finetuning"),
            ColSpec("integer", "height"),
            ColSpec("integer", "width"),
            ColSpec("integer", "num_images"),
            ColSpec("integer", "num_inference_steps")
        ])
        output_schema = Schema([ColSpec("string", "output_images")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        artifacts = {
            "finetuned_model": finetuned_model_path,
            "model_no_finetuning": model_no_finetuning_path
        }
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=cls(),
            artifacts=artifacts,
            signature=signature,
            pip_requirements=["torch", "diffusers", "transformers", "accelerate"]
        )
        logging.info("Model successfully logged into MLflow.")

def setup_accelerate():
    subprocess.run(["pip", "install", "accelerate"], check=True)
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        config_file = "../data/config/default_config_multi-gpu.yaml"
        logging.info("Using multi-GPU configuration with %d GPUs.", num_gpus)
    elif num_gpus == 1:
        config_file = "../data/config/default_config_one-gpu.yaml"
        logging.info("Using single-GPU configuration with 1 GPU.")
    else:
        config_file = "../data/config/default_config-cpu.yaml"
        logging.info("No GPU detected, using CPU configuration.")
    os.environ['ACCELERATE_CONFIG_FILE'] = config_file

def deploy_model():
    setup_accelerate()
    mlflow.set_experiment(experiment_name='ImageGeneration')
    finetuned_model_path = "./dreambooth"
    model_no_finetuning_path = "../../../local/stable-diffusion-2-1"
    with mlflow.start_run(run_name='image_generation_service') as run:
        logging.info("Run started: %s", run.info.run_id)
        mlflow.log_artifact(os.environ['ACCELERATE_CONFIG_FILE'], artifact_path="accelerate_config")
        logging.info("Accelerate configuration file logged in.")
        ImageGenerationModel.log_model(
            finetuned_model_path=finetuned_model_path,
            model_no_finetuning_path=model_no_finetuning_path
        )
        logging.info("Model successfully logged in.")
        model_uri = f"runs:/{run.info.run_id}/image_generation_model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="ImageGenerationService"
        )
        logging.info("Registered model: %s", registered_model.name)

if __name__ == "__main__":
    setup_accelerate()
    mlflow.set_experiment(experiment_name='ImageGeneration')
    logging.info("Experiment 'ImageGeneration' configured.")
    finetuned_model_path = "./dreambooth"
    model_no_finetuning_path = "../../../local/stable-diffusion-2-1/"
    with mlflow.start_run(run_name='image_generation_service') as run:
        logging.info("Run started: %s", run.info.run_id)
        mlflow.log_artifact(os.environ['ACCELERATE_CONFIG_FILE'], artifact_path="accelerate_config")
        logging.info("Accelerate configuration file logged in.")
        ImageGenerationModel.log_model(
            finetuned_model_path=finetuned_model_path,
            model_no_finetuning_path=model_no_finetuning_path
        )
        logging.info("Model successfully logged in.")
        model_uri = f"runs:/{run.info.run_id}/image_generation_model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="ImageGenerationService"
        )
        logging.info("Registered model:: %s", registered_model.name)
