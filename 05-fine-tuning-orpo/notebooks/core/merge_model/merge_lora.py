import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import setup_chat_format

def merge_lora_and_save(
    base_model_id: str,
    finetuned_lora_path: str,
    base_local_dir: str = None,
    use_bfloat16: bool = False,
    add_chat_template: bool = True
):
    """
    Merge LoRA weights into base model and save locally.

    Args:
        base_model_id (str): HF model ID or local path to base model.
        finetuned_lora_path (str): Path to directory with LoRA adapter weights.
        base_local_dir (str): Base path where merged model will be saved.
        use_bfloat16 (bool): Use bfloat16 if supported, otherwise fallback to float16.
        add_chat_template (bool): Apply chat template if not present in tokenizer.
    """
    print("🧹 Cleaning up memory...")
    gc.collect()
    torch.cuda.empty_cache()

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

    base_model_name = base_model_id.split("/")[-1]
    merged_model_name = f"Orpo-{base_model_name}-FT"
    save_path = os.path.join(
        base_local_dir or os.path.join("..", "..", "..", "local", "models_llora"),
        merged_model_name
    )
    os.makedirs(save_path, exist_ok=True)

    print(f"💾 Saving merged model to: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("✅ Finished! Model successfully merged and saved locally.")
