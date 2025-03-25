import torch
import os

from transformers import T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model

    
def merge_lora_weights(checkpoint_path, base_model_name, output_dir=None):
    """
    Merges LoRA adapter weights back into the base model.
    
    Args:
        checkpoint_path (str): Path to the LoRA adapter checkpoint file
        base_model_name (str): Name or path of the base model
        output_dir (str, optional): Directory to save the merged model, defaults to f"{base_model_name}-merged"
        
    Returns:
        The path to the saved merged model
    """
    print(f"Loading base model: {base_model_name}")
    
    model = T5ForConditionalGeneration.from_pretrained(base_model_name, device_map="auto")
    
    print(f"Loading adapter weights from: {checkpoint_path}")
    
    if os.path.isdir(checkpoint_path):
        checkpoint_file = os.path.join(checkpoint_path, "model.checkpoint")
    else:
        checkpoint_file = checkpoint_path
    
    if "t5" in base_model_name:
        config = LoraConfig(
            r=32, # harcoded from local_gpu_config.yaml
            lora_alpha=16, # idem
            target_modules= ["q", "v"],
            lora_dropout=0.0,
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )
    
    peft_model = get_peft_model(model, config)
    
    # Load the adapter weights 
    adapter_weights = torch.load(checkpoint_file)
    # Convert from DDP format
    adapter_weights = {k.replace('_LLM_model.', '').replace('module.', ''): v for k, v in adapter_weights.items()}
    peft_model.load_state_dict(adapter_weights, strict=False)
    
    print("Merging weights")
    merged_model = peft_model.merge_and_unload()
    
    if output_dir is None:
        output_dir = f"{base_model_name.split('/')[-1]}-merged"
    
    print(f"Saving merged model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    
    print(f"Model successfully merged and saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    
    checkpoint_path = "flan-t5-small/model.checkpoint"
    base_model_name = "google/flan-t5-small"
    output_dir = "flan-t5-small"
    
    checkpoint_path = "t5-efficient-small/model.checkpoint"
    base_model_name = "google/t5-efficient-small"
    output_dir = "t5-efficient-small"

    merged_model_path = merge_lora_weights(
        checkpoint_path=checkpoint_path,
        base_model_name=base_model_name,
        output_dir=output_dir
    )