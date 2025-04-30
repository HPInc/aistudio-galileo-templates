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
        
        # Log all available artifacts for debugging
        logging.info(f"Available artifacts: {list(context.artifacts.keys())}")
        
        # Check for required model artifacts
        if "model_no_finetuning" not in context.artifacts:
            logging.error("Required artifact 'model_no_finetuning' not found!")
            raise ValueError("Required artifact 'model_no_finetuning' not found in context")
        
        if "finetuned_model" not in context.artifacts:
            logging.error("Required artifact 'finetuned_model' not found!")
            raise ValueError("Required artifact 'finetuned_model' not found in context")
        
        self.model_no_finetuning_path = context.artifacts["model_no_finetuning"]
        self.model_finetuning_path = context.artifacts["finetuned_model"]
        
        # Check if demo folder is present
        if "demo" in context.artifacts:
            logging.info(f"Found demo folder at: {context.artifacts['demo']}")
            self.demo_path = context.artifacts["demo"]
        else:
            logging.warning("No demo folder found in artifacts. UI will not be available.")
            self.demo_path = None

        # Determine GPU configuration
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus >= 2:
            config_file = "config/default_config_multi-gpu.yaml"
            logging.info(f"Detected {self.num_gpus} GPUs, using multi-GPU configuration: {config_file}")
        elif self.num_gpus == 1:
            config_file = "config/default_config_one-gpu.yaml"
            logging.info(f"1 GPU detected, using single-GPU configuration: {config_file}")
        else:
            config_file = "config/default_config-cpu.yaml"
            logging.info("No GPU detected, using CPU configuration.")
        
        self.current_pipeline = None
        self.current_model = None
        
        logging.info("Context loading completed successfully")

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
        try:
            logging.info(f"Received model_input of type: {type(model_input)}")
            logging.info(f"Model input keys: {model_input.keys() if hasattr(model_input, 'keys') else 'Not a dictionary-like object'}")
            
            # Extract and validate the prompt
            prompt = (
                model_input["prompt"].iloc[0]
                if isinstance(model_input["prompt"], pd.Series)
                else model_input["prompt"]
            )
            
            if not prompt or not isinstance(prompt, str):
                error_msg = f"Invalid prompt: {prompt}"
                logging.error(error_msg)
                return pd.DataFrame({"error": [error_msg]})
            
            # Extract other parameters with more robust error handling
            try:
                use_finetuning = (
                    model_input["use_finetuning"].iloc[0]
                    if isinstance(model_input["use_finetuning"], pd.Series)
                    else model_input["use_finetuning"]
                )
            except (KeyError, AttributeError) as e:
                logging.warning(f"Error extracting use_finetuning, defaulting to False: {str(e)}")
                use_finetuning = False
            
            try:
                height = (
                    model_input.get("height", 512).iloc[0]
                    if isinstance(model_input["height"], pd.Series)
                    else model_input["height"]
                )
            except (KeyError, AttributeError) as e:
                logging.warning(f"Error extracting height, defaulting to 512: {str(e)}")
                height = 512
            
            try:
                width = (
                    model_input.get("width", 512).iloc[0]
                    if isinstance(model_input["width"], pd.Series)
                    else model_input["width"]
                )
            except (KeyError, AttributeError) as e:
                logging.warning(f"Error extracting width, defaulting to 512: {str(e)}")
                width = 512
            
            try:
                num_images = (
                    model_input.get("num_images", 1).iloc[0]
                    if isinstance(model_input["num_images"], pd.Series)
                    else model_input["num_images"]
                )
            except (KeyError, AttributeError) as e:
                logging.warning(f"Error extracting num_images, defaulting to 1: {str(e)}")
                num_images = 1
            
            try:
                num_inference_steps = (
                    model_input.get("num_inference_steps", 100).iloc[0]
                    if isinstance(model_input["num_inference_steps"], pd.Series)
                    else model_input["num_inference_steps"]
                )
            except (KeyError, AttributeError) as e:
                logging.warning(f"Error extracting num_inference_steps, defaulting to 50: {str(e)}")
                num_inference_steps = 50

            logging.info(f"Starting inference with parameters:")
            logging.info(f"  - Prompt: {prompt[:50]}...")
            logging.info(f"  - Height: {height}, Width: {width}")
            logging.info(f"  - Number of images: {num_images}")
            logging.info(f"  - Inference steps: {num_inference_steps}")
            logging.info(f"  - Use finetuning: {use_finetuning}")
            
            # Load the appropriate pipeline
            self.load_pipeline(use_finetuning)

            images = []
            with torch.no_grad():
                for i in range(num_images):
                    logging.info(f"Running inference for image {i+1}/{num_images}")
                    
                    try:
                        image_result = self.current_pipeline(
                            prompt,
                            height=height,
                            width=width,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=7.5
                        )
                        image = image_result.images[0]
                        
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
                        images.append(f"data:image/png;base64,{img_base64}")
                        
                        # Save the image locally for debugging
                        image_filename = f"local_model_result_{i}.png"
                        image.save(image_filename)
                        logging.info(f"Image saved locally as: {image_filename}")
                    except Exception as img_error:
                        error_msg = f"Error generating image {i+1}: {str(img_error)}"
                        logging.error(error_msg)
                        logging.error(f"Exception type: {type(img_error).__name__}")
                        import traceback
                        logging.error(f"Traceback: {traceback.format_exc()}")
                        # Continue with other images even if one fails
                        continue

            if not images:
                logging.error("No images were successfully generated")
                return pd.DataFrame({"error": ["Failed to generate any images"]})

            logging.info(f"Successfully generated {len(images)} images")
            return pd.DataFrame({"output_images": images})
        
        except Exception as e:
            error_msg = f"Error in predict method: {str(e)}"
            logging.error(error_msg)
            logging.error(f"Exception type: {type(e).__name__}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame({"error": [error_msg]})

    @classmethod
    def log_model(cls, finetuned_model_path, model_no_finetuning_path, artifact_path="image_generation_model", demo_folder=None):
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
        
        # Prepare artifacts dictionary
        artifacts = {
            "finetuned_model": finetuned_model_path,
            "model_no_finetuning": model_no_finetuning_path
        }
        
        # Add demo folder to artifacts if provided
        if demo_folder:
            logging.info(f"Adding demo folder to artifacts: {demo_folder}")
            if not os.path.exists(demo_folder):
                logging.warning(f"Demo folder does not exist, creating it: {demo_folder}")
                os.makedirs(demo_folder, exist_ok=True)
            artifacts["demo"] = demo_folder
        else:
            logging.warning("No demo folder specified, UI will not be available")
        
        # Prepare pip requirements for the model
        pip_requirements = [
            "torch", 
            "diffusers", 
            "transformers", 
            "accelerate"
        ]
        
        # Log model to MLflow
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=cls(),
            artifacts=artifacts,
            signature=signature,
            pip_requirements=pip_requirements,
            # Add code paths to ensure all necessary files are included
            code_paths=["../core", "../../src"]
        )
        logging.info("Model successfully logged into MLflow.")

def setup_accelerate():
    subprocess.run(["pip", "install", "accelerate"], check=True)
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        config_file = "config/default_config_multi-gpu.yaml"
        logging.info("Using multi-GPU configuration with %d GPUs.", num_gpus)
    elif num_gpus == 1:
        config_file = "config/default_config_one-gpu.yaml"
        logging.info("Using single-GPU configuration with 1 GPU.")
    else:
        config_file = "../data/config/default_config-cpu.yaml"
        logging.info("No GPU detected, using CPU configuration.")
    os.environ['ACCELERATE_CONFIG_FILE'] = config_file

def deploy_model(demo_folder=None):
    setup_accelerate()
    mlflow.set_experiment(experiment_name='ImageGeneration')
    finetuned_model_path = "./dreambooth"
    model_no_finetuning_path = "../../../local/stable-diffusion-2-1"
    
    # Check if demo folder exists
    if demo_folder:
        logging.info(f"Using demo folder: {demo_folder}")
        if not os.path.exists(demo_folder):
            logging.warning(f"Demo folder does not exist, creating it: {demo_folder}")
            os.makedirs(demo_folder, exist_ok=True)
    else:
        logging.warning("No demo folder specified, UI will not be available")
        demo_folder = None
    
    with mlflow.start_run(run_name='image_generation_service') as run:
        logging.info("Run started: %s", run.info.run_id)
        mlflow.log_artifact(os.environ['ACCELERATE_CONFIG_FILE'], artifact_path="accelerate_config")
        logging.info("Accelerate configuration file logged in.")
        ImageGenerationModel.log_model(
            finetuned_model_path=finetuned_model_path,
            model_no_finetuning_path=model_no_finetuning_path,
            demo_folder=demo_folder
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
    
    # Define demo folder path
    demo_folder = "../../demo"
    if not os.path.isabs(demo_folder):
        demo_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), demo_folder))
    
    logging.info(f"Using demo folder at: {demo_folder}")
    if not os.path.exists(demo_folder):
        logging.warning(f"Demo folder does not exist, creating it: {demo_folder}")
        os.makedirs(demo_folder, exist_ok=True)
    
    with mlflow.start_run(run_name='image_generation_service') as run:
        logging.info("Run started: %s", run.info.run_id)
        mlflow.log_artifact(os.environ['ACCELERATE_CONFIG_FILE'], artifact_path="accelerate_config")
        logging.info("Accelerate configuration file logged in.")
        ImageGenerationModel.log_model(
            finetuned_model_path=finetuned_model_path,
            model_no_finetuning_path=model_no_finetuning_path,
            demo_folder=demo_folder
        )
        logging.info("Model successfully logged in.")
        model_uri = f"runs:/{run.info.run_id}/image_generation_model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="ImageGenerationService"
        )
        logging.info("Registered model: %s", registered_model.name)
