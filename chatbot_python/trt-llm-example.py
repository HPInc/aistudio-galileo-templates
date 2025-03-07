#import tensorrt_llm
from tensorrt_llm import LLM, SamplingParams

import os
os.environ["HF_HOME"] = "/home/jovyan/local/hugging_face"
os.environ["HF_HUB_CACHE"] = "/home/jovyan/local/hugging_face/hub"
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="/home/jovyan/local/mistral-7b-engine")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
