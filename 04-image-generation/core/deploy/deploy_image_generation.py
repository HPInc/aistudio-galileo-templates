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
        try:
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
            
            # Log artifact paths for debugging
            self.model_no_finetuning_path = context.artifacts["model_no_finetuning"]
            self.model_finetuning_path = context.artifacts["finetuned_model"]
            
            logging.info(f"Model no finetuning path: {self.model_no_finetuning_path}")
            logging.info(f"Finetuned model path: {self.model_finetuning_path}")
            
            # Verify model directories exist
            if not os.path.exists(self.model_no_finetuning_path):
                logging.error(f"Model directory not found at: {self.model_no_finetuning_path}")
                raise FileNotFoundError(f"Model directory not found at: {self.model_no_finetuning_path}")
                
            if not os.path.exists(self.model_finetuning_path):
                logging.error(f"Finetuned model directory not found at: {self.model_finetuning_path}")
                raise FileNotFoundError(f"Finetuned model directory not found at: {self.model_finetuning_path}")
            
            # Check if demo folder is present
            if "demo" in context.artifacts:
                self.demo_path = context.artifacts["demo"]
                logging.info(f"Found demo folder at: {self.demo_path}")
                
                if not os.path.exists(self.demo_path):
                    logging.warning(f"Demo folder path exists in artifacts but directory not found at: {self.demo_path}")
                else:
                    # Log demo contents for debugging
                    try:
                        demo_files = os.listdir(self.demo_path)
                        logging.info(f"Demo folder contents: {demo_files}")
                        
                        # Check if index.html exists
                        if "index.html" not in demo_files:
                            logging.warning(f"index.html not found in demo folder: {self.demo_path}")
                    except Exception as e:
                        logging.error(f"Error listing demo folder contents: {str(e)}")
            else:
                logging.warning("No demo folder found in artifacts. UI will not be available.")
                self.demo_path = None

            # Determine GPU configuration
            self.num_gpus = torch.cuda.device_count()
            logging.info(f"Detected {self.num_gpus} GPUs")
            
            if self.num_gpus >= 2:
                config_file = "config/default_config_multi-gpu.yaml"
                logging.info(f"Using multi-GPU configuration: {config_file}")
            elif self.num_gpus == 1:
                config_file = "config/default_config_one-gpu.yaml"
                logging.info(f"Using single-GPU configuration: {config_file}")
            else:
                config_file = "config/default_config-cpu.yaml"
                logging.info(f"No GPU detected, using CPU configuration: {config_file}")
            
            # Initialize pipeline placeholders
            self.current_pipeline = None
            self.current_model = None
            
            logging.info("Context loading completed successfully")
        except Exception as e:
            logging.error(f"Error loading context: {str(e)}")
            logging.error(f"Exception type: {type(e).__name__}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise

    def load_pipeline(self, use_finetuning):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Loading pipeline on device: {device}")
            
            # Check if model paths exist
            if not os.path.exists(self.model_no_finetuning_path):
                logging.error(f"Model directory not found: {self.model_no_finetuning_path}")
                raise FileNotFoundError(f"Model directory not found: {self.model_no_finetuning_path}")
                
            if not os.path.exists(self.model_finetuning_path):
                logging.error(f"Finetuned model directory not found: {self.model_finetuning_path}")
                raise FileNotFoundError(f"Finetuned model directory not found: {self.model_finetuning_path}")
            
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            logging.info(f"Using torch dtype: {torch_dtype}")
            
            if self.current_model == "finetuning" and not use_finetuning:
                logging.info("Switching to the model without fine-tuning...")
                if hasattr(self, 'current_pipeline') and self.current_pipeline is not None:
                    del self.current_pipeline
                    torch.cuda.empty_cache() if device == "cuda" else None
                
                logging.info(f"Loading model from: {self.model_no_finetuning_path}")
                self.current_pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_no_finetuning_path, 
                    torch_dtype=torch_dtype, 
                    low_cpu_mem_usage=True,
                    safety_checker=None  # Disable safety checker for faster loading
                ).to(device)
                self.current_model = "no_finetuning"
                logging.info("Model without fine-tuning loaded successfully")
                
            elif self.current_model == "no_finetuning" and use_finetuning:
                logging.info("Switching to the finetuned model...")
                if hasattr(self, 'current_pipeline') and self.current_pipeline is not None:
                    del self.current_pipeline
                    torch.cuda.empty_cache() if device == "cuda" else None
                
                logging.info(f"Loading finetuned model from: {self.model_finetuning_path}")
                self.current_pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_finetuning_path, 
                    torch_dtype=torch_dtype, 
                    low_cpu_mem_usage=True,
                    safety_checker=None  # Disable safety checker for faster loading
                ).to(device)
                self.current_model = "finetuning"
                logging.info("Finetuned model loaded successfully")
                
            elif self.current_pipeline is None:
                if use_finetuning:
                    logging.info(f"Loading finetuned pipeline for the first time from: {self.model_finetuning_path}")
                    self.current_pipeline = StableDiffusionPipeline.from_pretrained(
                        self.model_finetuning_path,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                        safety_checker=None  # Disable safety checker for faster loading
                    ).to(device)
                    self.current_model = "finetuning"
                    logging.info("Finetuned model loaded successfully")
                else:
                    logging.info(f"Loading pipeline without fine-tuning for the first time from: {self.model_no_finetuning_path}")
                    self.current_pipeline = StableDiffusionPipeline.from_pretrained(
                        self.model_no_finetuning_path,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                        safety_checker=None  # Disable safety checker for faster loading
                    ).to(device)
                    self.current_model = "no_finetuning"
                    logging.info("Model without fine-tuning loaded successfully")
                    
        except Exception as e:
            logging.error(f"Error loading pipeline: {str(e)}")
            logging.error(f"Exception type: {type(e).__name__}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise

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
        try:
            logging.info("Starting model logging...")
            
            # Define model input/output schema
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
            
            # Handle demo folder
            if demo_folder:
                logging.info(f"Adding demo folder to artifacts: {demo_folder}")
                artifacts["demo"] = demo_folder
            else:
                logging.warning("No demo folder specified, UI will not be available")
            
            # Prepare pip requirements for the model
            pip_requirements = [
                "torch", 
                "diffusers", 
                "transformers", 
                "accelerate",
                "pillow",
                "pandas"
            ]
        except Exception as e:
            logging.error(f"Error in log_model setup: {str(e)}")
            logging.error(f"Exception type: {type(e).__name__}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Log model to MLflow
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=cls(),
            artifacts=artifacts,
            signature=signature,
            pip_requirements=pip_requirements,
            code_paths=["../core", "../../src"]
        )
        logging.info("Model successfully logged into MLflow.")

def setup_accelerate():
    try:
        logging.info("Setting up accelerate...")
        # Try to install accelerate, but don't fail if it's already installed
        try:
            subprocess.run(["pip", "install", "accelerate"], check=False)
        except Exception as e:
            logging.warning(f"Could not install accelerate: {str(e)}. Will continue anyway.")
        
        # Try to detect GPUs, default to 0 if CUDA is not available
        try:
            num_gpus = torch.cuda.device_count()
            logging.info(f"Detected {num_gpus} GPU(s)")
        except Exception as e:
            logging.warning(f"Could not detect GPUs: {str(e)}. Defaulting to CPU mode.")
            num_gpus = 0
        
        # Base directory for config files
        config_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs")
        
        if num_gpus >= 2:
            config_file_name = "default_config_multi-gpu.yaml"
            logging.info(f"Using multi-GPU configuration with {num_gpus} GPUs.")
        elif num_gpus == 1:
            config_file_name = "default_config_one-gpu.yaml"
            logging.info("Using single-GPU configuration with 1 GPU.")
        else:
            config_file_name = "default_config-cpu.yaml"
            logging.info("No GPU detected, using CPU configuration.")
        
        # Check multiple possible locations for config files
        possible_config_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", config_file_name),
            os.path.join(config_base_dir, config_file_name),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "config", config_file_name)
        ]
        
        config_file = None
        for path in possible_config_paths:
            if os.path.exists(path):
                config_file = path
                logging.info(f"Found config file at: {config_file}")
                break
        
        # If no config file found, create a default one
        if not config_file:
            # Create default config directory if it doesn't exist
            default_config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
            os.makedirs(default_config_dir, exist_ok=True)
            
            # Create a default config file
            config_file = os.path.join(default_config_dir, config_file_name)
            logging.info(f"Creating default config file at: {config_file}")
            
            # Write minimal config content
            with open(config_file, "w") as f:
                f.write("compute_environment: LOCAL_MACHINE\n")
                f.write("distributed_type: NO\n")
                f.write("mixed_precision: fp16\n")
                f.write("use_cpu: false\n" if num_gpus > 0 else "use_cpu: true\n")
        
        # Set accelerate config file environment variable
        logging.info(f"Setting ACCELERATE_CONFIG_FILE to: {config_file}")
        os.environ['ACCELERATE_CONFIG_FILE'] = config_file
        return config_file
    except Exception as e:
        logging.error(f"Error setting up accelerate: {str(e)}")
        logging.error(f"Exception type: {type(e).__name__}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        # Create a default config file in /tmp if the normal one fails
        try:
            default_config = "/tmp/default_accelerate_config.yaml"
            with open(default_config, "w") as f:
                f.write("compute_environment: LOCAL_MACHINE\n")
                f.write("distributed_type: NO\n")
                f.write("mixed_precision: no\n")
                f.write("use_cpu: true\n")
            logging.warning(f"Created fallback accelerate config at: {default_config}")
            os.environ['ACCELERATE_CONFIG_FILE'] = default_config
            return default_config
        except Exception as inner_e:
            logging.error(f"Could not create default accelerate config: {str(inner_e)}")
            # Don't re-raise, allow the process to continue
            return None

def deploy_model(demo_folder=None, finetuned_model_path=None, model_no_finetuning_path=None):
    """
    Deploy the image generation model to MLflow.
    
    Args:
        demo_folder (str, optional): Path to the demo folder to include with the model.
        finetuned_model_path (str, optional): Path to the finetuned model. If None, will use default path.
        model_no_finetuning_path (str, optional): Path to the base model without finetuning. If None, will use default path.
    """
    try:
        logging.info("Starting model deployment...")
        setup_accelerate()
        
        mlflow.set_experiment(experiment_name='ImageGeneration')
        logging.info("Set experiment name to 'ImageGeneration'")
        
        # Ensure paths are absolute
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set model paths if not provided
        if finetuned_model_path is None:
            finetuned_model_path = os.path.abspath(os.path.join(current_dir, "./dreambooth"))
        elif not os.path.isabs(finetuned_model_path):
            finetuned_model_path = os.path.abspath(os.path.join(os.getcwd(), finetuned_model_path))
            
        if model_no_finetuning_path is None:
            model_no_finetuning_path = os.path.abspath(os.path.join(current_dir, "../../../local/stable-diffusion-2-1"))
        elif not os.path.isabs(model_no_finetuning_path):
            model_no_finetuning_path = os.path.abspath(os.path.join(os.getcwd(), model_no_finetuning_path))
        
        logging.info(f"Finetuned model path: {finetuned_model_path}")
        logging.info(f"Model without fine-tuning path: {model_no_finetuning_path}")
        
        # Verify model paths exist
        if not os.path.exists(finetuned_model_path):
            logging.error(f"Finetuned model path not found: {finetuned_model_path}")
            raise FileNotFoundError(f"Finetuned model path not found: {finetuned_model_path}")
            
        if not os.path.exists(model_no_finetuning_path):
            logging.error(f"Model path not found: {model_no_finetuning_path}")
            raise FileNotFoundError(f"Model path not found: {model_no_finetuning_path}")
        
        # Simple demo folder handling
        if demo_folder is None:
            # Default to project demo folder
            demo_folder = os.path.abspath(os.path.join(current_dir, "../../demo"))
            
        # Ensure demo folder path is absolute
        if not os.path.isabs(demo_folder):
            if os.path.dirname(os.path.abspath(__file__)) == os.path.abspath(os.getcwd()):
                # When running from the deploy directory
                demo_folder = os.path.abspath(os.path.join(current_dir, demo_folder))
            else:
                # When running from a different directory
                demo_folder = os.path.abspath(os.path.join(os.getcwd(), demo_folder))
            
        logging.info(f"Using demo folder: {demo_folder}")
            
    except Exception as e:
        logging.error(f"Error in deploy_model setup: {str(e)}")
        logging.error(f"Exception type: {type(e).__name__}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    with mlflow.start_run(run_name='image_generation_service') as run:
        logging.info("Run started: %s", run.info.run_id)
        
        # Log accelerate config file if it exists
        try:
            if 'ACCELERATE_CONFIG_FILE' in os.environ and os.path.exists(os.environ['ACCELERATE_CONFIG_FILE']):
                mlflow.log_artifact(os.environ['ACCELERATE_CONFIG_FILE'], artifact_path="accelerate_config")
                logging.info("Accelerate configuration file logged in.")
            else:
                logging.warning("Accelerate configuration file not found, skipping artifact logging.")
        except Exception as e:
            logging.warning(f"Could not log accelerate config file: {str(e)}")
            
        # Log the model
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
        
        return registered_model

if __name__ == "__main__":
    # This script is primarily meant to be imported by notebooks for deployment
    # But we provide a direct execution option for debugging purposes
    import argparse
    import traceback

    parser = argparse.ArgumentParser(description='Deploy image generation model to MLflow')
    parser.add_argument('--demo-folder', type=str, default="../../demo", help='Path to demo folder')
    parser.add_argument('--finetuned-model', type=str, default="./dreambooth", help='Path to finetuned model')
    parser.add_argument('--base-model', type=str, default="../../../local/stable-diffusion-2-1", help='Path to base model')
    
    args = parser.parse_args()
    
    print("This script is primarily intended to be imported by notebooks.")
    print("Running deploy_model with default paths for debugging purposes...")
    
    try:
        print(f"Demo folder: {args.demo_folder}")
        print(f"Finetuned model: {args.finetuned_model}")
        print(f"Base model: {args.base_model}")
        
        # Call the deploy_model function with provided paths
        deploy_model(
            demo_folder=args.demo_folder,
            finetuned_model_path=args.finetuned_model,
            model_no_finetuning_path=args.base_model
        )
    except Exception as e:
        print(f"ERROR: Failed to deploy model: {e}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Traceback: {traceback.format_exc()}")
        # Exit with error code but don't crash immediately
        import sys
        sys.exit(1)
