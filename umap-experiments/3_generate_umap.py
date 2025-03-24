import os
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def visualize_prompt_embeddings(embeddings_dir, output_file=None, n_samples=100):
    """
    Create a UMAP visualization of prompt embeddings from different formulations.
    
    Args:
        embeddings_dir: Directory containing the embeddings
        output_file: Path to save the visualization (optional)
        n_samples: Number of samples to use from each prompt type
    """
    prompt_types = ['Glam_prompt', 'swap_prompt', 'xml_prompt', 'paraphrase_prompt']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    all_embeddings = []
    all_labels = []
    
    plt.figure(figsize=(12, 10))
    
    for i, prompt_type in enumerate(prompt_types):
        embedding_file = os.path.join(embeddings_dir, f"{prompt_type}_mean_embeddings.npy")
        if not os.path.exists(embedding_file):
            print(f"Warning: {embedding_file} not found, skipping")
            continue
            
        embeddings = np.load(embedding_file)
        print(f"Loaded {embeddings.shape[0]} embeddings for {prompt_type}")
        
        n_to_use = min(n_samples, embeddings.shape[0])
        if n_to_use < embeddings.shape[0]:
            indices = np.random.choice(embeddings.shape[0], n_to_use, replace=False)
            embeddings = embeddings[indices]
        
        all_embeddings.append(embeddings)
        all_labels.extend([prompt_type] * n_to_use)
    
    combined_embeddings = np.vstack(all_embeddings)
    
    print("Standardizing data")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_embeddings)
    
    print("Applying UMAP dimensionality reduction")
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(scaled_data)
    
    label_to_color = {prompt_type: colors[i] for i, prompt_type in enumerate(prompt_types)}
    point_colors = [label_to_color[label] for label in all_labels]
    
    
    plt.figure(figsize=(12, 10))
    
    sns.set_style("whitegrid")
    
    scatter = plt.scatter(
        embedding[:, 0], 
        embedding[:, 1], 
        c=point_colors, 
        alpha=0.7,
        s=80
    )
    
    plt.title('UMAP Visualization of Prompt Embeddings', fontsize=18)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_to_color[prompt_type], 
                          markersize=10, label=prompt_type) for prompt_type in prompt_types]
    plt.legend(handles=handles, title="Prompt Type", title_fontsize=14, 
               fontsize=12, loc='best', frameon=True)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    
    plt.show()

if __name__ == "__main__":
    embeddings_dir = "prompts_rep_original"
    output_file = "prompt_embeddings_umap_flan-t5-small-original.png"
    
    embeddings_dir = "gpt2/prompts_rep_original"
    output_file = "prompt_embeddings_umap_gpt2.png"
    
    embeddings_dir = "blenderbot/prompts_rep_original"
    output_file = "prompt_embeddings_umap_blenderbot-small.png"
    
    embeddings_dir = "meta-llama/prompts_rep_original"
    output_file = "prompt_embeddings_umap_meta-llama.png"
    
    embeddings_dir = "opt/prompts_rep_original"
    output_file = "prompt_embeddings_umap_opt.png"
    
    visualize_prompt_embeddings(embeddings_dir, output_file)