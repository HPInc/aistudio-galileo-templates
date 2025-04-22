import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import setup_chat_format
from huggingface_hub import login

def merge_lora_and_push(
    base_model_id: str,
    finetuned_lora_path: str,
    push_to: str,
    use_bfloat16: bool = False,
    add_chat_template: bool = True,
    hf_token: str = None
):
    """
    Merge LoRA weights into base model and push to Hugging Face Hub.

    Args:
        base_model_id (str): The Hugging Face model ID or local path to base model.
        finetuned_lora_path (str): Local directory where LoRA weights were saved.
        push_to (str): Target repo name on Hugging Face (e.g., "username/model-merged").
        use_bfloat16 (bool): Whether to use bfloat16 instead of float16.
        add_chat_template (bool): If True, applies setup_chat_format to ensure prompt compatibility.
        hf_token (str): Optional Hugging Face token to login before pushing.
    """
    print("🧹 Cleaning up memory...")
    gc.collect()
    torch.cuda.empty_cache()

    if hf_token:
        print("🔐 Logging into Hugging Face...")
        login(token=hf_token)

    torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    print("🔄 Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch_dtype,
        device_map="auto"
    )

    vocab_size_tokenizer = len(tokenizer)
    vocab_size_model = model.get_input_embeddings().num_embeddings

    if vocab_size_tokenizer != vocab_size_model:
        print(f"⚠️ Resizing token embeddings: model ({vocab_size_model}) → tokenizer ({vocab_size_tokenizer})")
        model.resize_token_embeddings(vocab_size_tokenizer)

    if add_chat_template:
        if tokenizer.chat_template is None:
            print("💬 Applying chat format...")
            model, tokenizer = setup_chat_format(model, tokenizer)
        else:
            print("⚠️ Tokenizer already has chat_template. Skipping setup_chat_format.")

    print("🔗 Loading LoRA weights from:", finetuned_lora_path)
    model = PeftModel.from_pretrained(model, finetuned_lora_path)

    print("🧠 Merging LoRA weights...")
    model = model.merge_and_unload()

    print(f"🚀 Pushing model to Hugging Face Hub: {push_to}")
    model.push_to_hub(push_to, use_temp_dir=False)
    tokenizer.push_to_hub(push_to, use_temp_dir=False)

    print("✅ Finished! Model successfully merged and uploaded.")
