import pickle

def display_prompt_examples(prompts_file):
    with open(prompts_file, 'rb') as f:
        prompt_samples = pickle.load(f)
    print(f"{len(prompt_samples)} prompt types")
    
    examples = {}
    
    for prompt_sample_type, prompts in prompt_samples.items():
        if prompt_sample_type == 'all_prompts':
            continue
        
        examples[prompt_sample_type] = prompts[0]
        
    
    for prompt_type, example in examples.items():
        print(f"\n--- {prompt_type} ---")
        print(example)
        print("-" * 80)

if __name__ == "__main__":
    prompts_file = "prompt_samples.pkl"
    display_prompt_examples(prompts_file)
    