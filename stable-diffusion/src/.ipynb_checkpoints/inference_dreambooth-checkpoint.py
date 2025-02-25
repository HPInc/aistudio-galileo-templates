import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
import time
import numpy as np
from PIL import Image
import yaml
from typing import List, Union

class StableDiffusionPipelineOutput:
    def __init__(self, images: Union[List[Image.Image], np.ndarray], nsfw_content_detected: List[bool]):
        self.images = images
        self.nsfw_content_detected = nsfw_content_detected

def load_config():
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        config_file = "/home/jovyan/GenAI-Demos/Stable-Diffusion/config/default_config_multi-gpu.yaml"
        print(f"Detected {num_gpus} GPUs, using {config_file}")
    else:
        config_file = "/home/jovyan/GenAI-Demos/Stable-Diffusion/config/default_config_one-gpu.yaml"
        print(f"Detected {num_gpus} GPU, using {config_file}")
    
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_max_memory_per_gpu():
    """Detecta a quantidade máxima de memória de cada GPU disponível."""
    max_memory = {}
    num_gpus = torch.cuda.device_count()
    
    for gpu_index in range(num_gpus):
        props = torch.cuda.get_device_properties(gpu_index)
        max_memory[gpu_index] = f"{int(props.total_memory / 1024**3 - 2)}GB"
        print(f"GPU {gpu_index}: {max_memory[gpu_index]} of available memory.")
    
    return max_memory

def run_inference(prompt=None, height=512, width=512, num_images=5, num_inference_steps=50):
    default_prompt = (
        "A sleek, modern laptop open on a sandy beach, positioned in front of a vibrant blue ocean. "
        "The sun is shining brightly, casting soft shadows across the sand. The screen of the laptop "
        "displays a bright, colorful interface, perhaps with a tropical background. Surrounding the "
        "laptop are a few seashells, and the gentle waves are just a few feet away. The sky is clear, "
        "with a few fluffy white clouds, and there are palm trees in the background swaying slightly "
        "in the breeze. The overall atmosphere is serene and inviting, perfect for remote work or relaxation."
    )

    prompt = prompt if prompt else default_prompt

    accelerator = Accelerator()

    max_memory = get_max_memory_per_gpu()

    model_id = "./dreambooth"  
    
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="balanced", 
        max_memory=max_memory,
        low_cpu_mem_usage=True 
    )

    inference_times = []
    images = []
    nsfw_flags = []

    if accelerator.process_index == 0:  
        for i in range(num_images): 
            start_time = time.time()
            result = pipe(prompt, height=height, width=width, num_inference_steps=num_inference_steps)
            end_time = time.time()

            output = StableDiffusionPipelineOutput(images=result.images, nsfw_content_detected=[False] * len(result.images))
            
            nsfw_flags.append(output.nsfw_content_detected)

            inference_time = end_time - start_time
            inference_times.append(inference_time)
            output.images[0].save(f"local_model_result_{i}.png")  
            images.append(output.images[0])

        avg_inference_time = np.mean(inference_times)
        median_inference_time = np.median(inference_times)
        min_inference_time = min(inference_times)
        max_inference_time = max(inference_times)

        print(f"Average Inference Time: {avg_inference_time:.2f} seconds")
        print(f"Median Inference Time: {median_inference_time:.2f} seconds")
        print(f"Min Inference Time: {min_inference_time:.2f} seconds")
        print(f"Max Inference Time: {max_inference_time:.2f} seconds")

    accelerator.end_training()
