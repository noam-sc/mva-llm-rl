import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration, T5Tokenizer,
    GPT2LMHeadModel, GPT2Tokenizer, 
    LlamaForCausalLM, LlamaTokenizer
)
from transformers import AutoTokenizer

class PromptEncoder:
    """
    Extracts hidden representations from various models for a set of prompts.
    """
    def __init__(self, model_path, model_type="t5", model_name=None, local_path=None, device=None):
        """
        Args:
            model_path: Path to the model
            model_type: Type of model ('t5', 'gpt', 'llama', 'auto')
            model_name: HF model name if loading from hub
            device: Device to run the model on (default: cuda if available, otherwise cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type.lower()
        model_source = model_name if model_name else model_path
        model_source_path = model_source if local_path is None else local_path
        
        print(f"Loading {model_type} model from {model_source} on {self.device}...")
        
        # Load model and tokenizer based on model type
        if model_type == "t5":
            self.model = T5ForConditionalGeneration.from_pretrained(model_source_path, local_files_only=(local_path is not None))
            self.tokenizer = T5Tokenizer.from_pretrained(model_source)
            self.hidden_size = self.model.config.d_model
            self.is_encoder_decoder = True
        elif model_type == "gpt":
            self.model = GPT2LMHeadModel.from_pretrained(model_source)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_source)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.hidden_size = self.model.config.hidden_size
            self.is_encoder_decoder = False
        elif model_type == "llama":
            self.model = LlamaForCausalLM.from_pretrained(model_source)
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.hidden_size = self.model.config.hidden_size
            self.is_encoder_decoder = False
        elif model_type == "auto":
            # Try to automatically determine the model architecture
            if hasattr(AutoModelForSeq2SeqLM, "from_pretrained"):
                try:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_source)
                    self.is_encoder_decoder = True
                except:
                    self.model = AutoModelForCausalLM.from_pretrained(model_source)
                    self.is_encoder_decoder = False
            else:
                self.model = AutoModel.from_pretrained(model_source)
                self.is_encoder_decoder = hasattr(self.model, "encoder") and hasattr(self.model, "decoder")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Try to determine hidden size
            if hasattr(self.model.config, "d_model"):
                self.hidden_size = self.model.config.d_model
            elif hasattr(self.model.config, "hidden_size"):
                self.hidden_size = self.model.config.hidden_size
            else:
                raise ValueError("Could not determine hidden size for the model")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded with hidden size: {self.hidden_size}")
        print(f"Model architecture: {'encoder-decoder' if self.is_encoder_decoder else 'decoder-only'}")
    
    def extract_hidden_states(self, prompts, batch_size=8, output_attentions=False):
        """
        Extract hidden representations from the model for a list of prompts.
        
        Args:
            prompts: List of text prompts
            batch_size: Batch size for processing
            output_attentions: Whether to return attention weights as well
            
        Returns:
            List of dictionaries containing prompt text, hidden states, and mean pooled representation
        """
        # Process in batches
        num_prompts = len(prompts)
        num_batches = (num_prompts + batch_size - 1) // batch_size
        
        all_states = []
        
        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Extracting embeddings"):
                # Get batch
                batch_prompts = prompts[i * batch_size:(i + 1) * batch_size]
                
                # Tokenize
                inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Extract hidden states based on model type
                if self.is_encoder_decoder:
                    
                    if hasattr(self.model, "encoder") and callable(getattr(self.model.encoder, "forward", None)):
                        # Models like T5 with direct encoder access
                        outputs = self.model.encoder(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            output_hidden_states=True,
                            output_attentions=output_attentions,
                            return_dict=True
                        )
                        last_hidden_states = outputs.last_hidden_state
                    else:
                        # For models like Blenderbot where we need to access encoder differently
                        # Use the full model but extract only encoder representations
                        encoder_outputs = self.model.forward(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            output_hidden_states=True,
                            output_attentions=output_attentions,
                            return_dict=True,
                            use_cache=False,
                            encoder_outputs=None,  # Force encoder computation
                            decoder_input_ids=torch.zeros((inputs["input_ids"].shape[0], 1), 
                                                        dtype=torch.long, 
                                                        device=self.device)  # Minimal decoder input
                        )
                        
                        # Access encoder's last hidden state
                        if hasattr(encoder_outputs, "encoder_last_hidden_state"):
                            last_hidden_states = encoder_outputs.encoder_last_hidden_state
                        elif hasattr(encoder_outputs, "encoder_hidden_states"):
                            last_hidden_states = encoder_outputs.encoder_hidden_states[-1]
                        else:
                            # Fallback to using the full model's hidden states
                            last_hidden_states = encoder_outputs.hidden_states[-1]
                else:
                    # For decoder-only models like GPT, LLaMA
                    outputs = self.model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        output_hidden_states=True,
                        output_attentions=output_attentions,
                        return_dict=True
                    )
                    last_hidden_states = outputs.hidden_states[-1]
                
                # Mean pooling (average over tokens)
                mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden_states.size()).float()
                masked_hidden = last_hidden_states * mask
                mean_pooled = torch.sum(masked_hidden, dim=1) / torch.sum(mask, dim=1)
                
                # Get sequence lengths
                seq_lengths = inputs["attention_mask"].sum(dim=1).cpu()
                
                # Store results for this batch
                for j in range(len(batch_prompts)):
                    # Only store non-padding tokens
                    length = seq_lengths[j].item()
                    hidden_j = last_hidden_states[j, :length, :].cpu()
                    mean_pooled_j = mean_pooled[j].cpu()
                    
                    all_states.append({
                        "prompt": batch_prompts[j],
                        "hidden_states": hidden_j,
                        "mean_pooled": mean_pooled_j
                    })
        
        return all_states

def extract_and_save_embeddings(prompts_file, model_path, output_dir, model_type="t5", model_name=None, local_path=None, batch_size=8):
    """
    Extract embeddings for all prompts and save them.
    
    Args:
        prompts_file: Path to the pickle file containing the prompts
        model_path: Path to the model
        output_dir: Directory to save the embeddings
        model_type: Type of model ('t5', 'gpt', 'llama', 'auto')
        model_name: HF model name if loading from hub
        batch_size: Batch size for processing
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load prompts
    print(f"Loading prompts from {prompts_file}...")
    with open(prompts_file, 'rb') as f:
        prompt_samples = pickle.load(f)
    
    # Initialize the encoder
    encoder = PromptEncoder(model_path, model_type, model_name, local_path)
    
    # Extract and save embeddings for each prompt type
    for prompt_type, prompts in prompt_samples.items():
        if prompt_type == 'all_prompts':
            continue  # Skip the combined list to avoid duplication
            
        print(f"Processing {len(prompts)} prompts for {prompt_type}...")
        embeddings = encoder.extract_hidden_states(prompts, batch_size=batch_size)
        
        # Save to file
        output_file = os.path.join(output_dir, f"{prompt_type}_embeddings.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        print(f"Saved embeddings to {output_file}")
        
        # Create a numpy array of just the mean pooled representations for easy access
        mean_pooled = torch.stack([item["mean_pooled"] for item in embeddings]).numpy()
        mean_output_file = os.path.join(output_dir, f"{prompt_type}_mean_embeddings.npy")
        np.save(mean_output_file, mean_pooled)
        print(f"Saved mean pooled embeddings to {mean_output_file}")


if __name__ == "__main__":
    prompts_file = "prompt_samples.pkl"
    model_path = "t5-efficient-small"
    model_name = "google/t5-efficient-small" 
    model_type = "t5"
    local_path = "t5-efficient-small"
    
    local_path = None
    model_path = "gpt2"
    model_name = "openai-community/gpt2"
    model_type = "gpt"
    output_dir = "gpt2/prompts_rep"
    
    model_path = "blenderbot"
    model_name = "facebook/blenderbot_small-90M"
    model_type = "auto"
    output_dir = "blenderbot/prompts_rep"
    
    model_path = "meta-llama"
    model_name = "unsloth/Llama-3.2-1B"
    model_type = "llama"
    output_dir = "llama/prompts_rep"
    
    model_path = "opt"
    model_name = "facebook/opt-125m"
    model_type = "auto"
    output_dir = "opt/prompts_rep"
    
    output_dir = f"{model_path}/prompts_rep_original"
    
    extract_and_save_embeddings(prompts_file, model_path, output_dir, model_type, model_name, local_path=local_path, batch_size=8)