"""
Image Generation Service implementation for interfacing with MLflow image generation models.

This service is a wrapper for the Stable Diffusion image generation model deployed with MLflow.
It handles parameter validation, MLflow integration, and result processing.
"""

import os
import json
import logging
import pandas as pd
import base64
from typing import Dict, List, Any, Union, Optional

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ImageGenerationService:
    """
    Service for handling image generation through the MLflow-deployed Stable Diffusion model.
    
    This service validates input parameters and formats data for the MLflow model 
    """
    
    def __init__(self, mlflow_client=None):
        """
        Initialize the image generation service.
        
        Args:
            mlflow_client: An MLflow client that can invoke the deployed model
        """
        self.mlflow_client = mlflow_client
        # Default model parameters
        self.default_params = {
            "height": 512, 
            "width": 512,
            "num_images": 1,
            "num_inference_steps": 30,
            "use_finetuning": False
        }
        logger.info("Image Generation Service initialized")

    def _validate_parameters(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize the parameters for the image generation model.
        
        Args:
            request_data: The request data containing generation parameters
            
        Returns:
            Dict containing validated parameters
        """
        # Extract required prompt
        prompt = request_data.get("prompt", "")
        if not prompt:
            raise ValueError("Prompt cannot be empty")
            
        # Extract and validate other parameters
        params = {**self.default_params}
        for key in params:
            if key in request_data:
                params[key] = request_data[key]
        
        # Validate dimensions
        if not isinstance(params["height"], int) or params["height"] < 128 or params["height"] > 1024:
            raise ValueError("Height must be an integer between 128 and 1024")
        
        if not isinstance(params["width"], int) or params["width"] < 128 or params["width"] > 1024:
            raise ValueError("Width must be an integer between 128 and 1024")
        
        # Validate count parameters
        if not isinstance(params["num_images"], int) or params["num_images"] < 1 or params["num_images"] > 4:
            raise ValueError("Number of images must be an integer between 1 and 4")
        
        if not isinstance(params["num_inference_steps"], int) or params["num_inference_steps"] < 1 or params["num_inference_steps"] > 100:
            raise ValueError("Number of inference steps must be an integer between 1 and 100")
        
        # Add the prompt to params
        params["prompt"] = prompt
        
        logger.info(f"Processing request with prompt: {prompt[:50]}...")
        logger.info(f"Parameters: height={params['height']}, width={params['width']}, " +
                   f"num_images={params['num_images']}, num_inference_steps={params['num_inference_steps']}")
        
        return params

    def generate_images(self, request_data: Dict[str, Any]) -> List[str]:
        """
        Generate images based on the provided parameters.
        
        This method:
        1. Validates the input parameters
        2. Formats the data for the MLflow model
        3. Invokes the model through the MLflow client
        4. Processes and returns the results
        
        Args:
            request_data: Dictionary containing generation parameters
            
        Returns:
            List of base64-encoded images or image URLs
        """
        try:
            # Validate parameters
            params = self._validate_parameters(request_data)
            
            # Create DataFrame for model input (MLflow expects pandas DataFrame)
            data = pd.DataFrame({
                "prompt": [params["prompt"]],
                "height": [params["height"]],
                "width": [params["width"]],
                "num_images": [params["num_images"]],
                "num_inference_steps": [params["num_inference_steps"]],
                "use_finetuning": [params["use_finetuning"]]
            })
            
            # Run prediction through MLflow client if available
            if self.mlflow_client:
                logger.info("Invoking MLflow model for image generation")
                result = self.mlflow_client.invoke(data=data)
                
                # Extract images from result based on MLflow model response format
                if isinstance(result, pd.DataFrame) and "output_images" in result.columns:
                    images = result["output_images"].tolist()
                else:
                    images = result
                    
                logger.info(f"Generated {len(images) if isinstance(images, list) else 1} images successfully")
                return images
            else:
                # For development/testing without MLflow
                logger.warning("No MLflow client available, returning mock response")
                return self._mock_response(params)
        
        except Exception as e:
            logger.error(f"Error generating images: {str(e)}")
            raise
    
    def _mock_response(self, params: Dict[str, Any]) -> List[str]:
        """
        Create a mock response for development/testing without MLflow.
        
        Args:
            params: The validated parameters
            
        Returns:
            List of mock base64 image strings
        """
        logger.info("Creating mock images for testing")
        # Create a simple colored image for testing
        from PIL import Image, ImageDraw
        import io
        
        num_images = params["num_images"]
        width = params["width"]
        height = params["height"]
        
        mock_images = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        for i in range(num_images):
            img = Image.new('RGB', (width, height), color=colors[i % len(colors)])
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"Mock Image {i+1}\nPrompt: {params['prompt'][:20]}", fill=(255, 255, 255))
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
            mock_images.append(f"data:image/png;base64,{img_base64}")
            
        return mock_images
    
    def process_request(self, request_body: Union[str, Dict]) -> Dict[str, Any]:
        """
        Process an incoming request and return the response.
        
        Args:
            request_body: JSON string or dictionary from request
            
        Returns:
            List of images or error response
        """
        try:
            if isinstance(request_body, str):
                data = json.loads(request_body)
            else:
                data = request_body
            
            # Generate images
            images = self.generate_images(data)
            
            # Return the images directly
            return images
        
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return {"error": str(e)}
