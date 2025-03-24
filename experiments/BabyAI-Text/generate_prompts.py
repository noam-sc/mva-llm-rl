import numpy as np
import torch
import random
import os
import pickle
import hydra
from tqdm import tqdm

from utils.generate_prompt import Glam_prompt, swap_prompt, xml_prompt, paraphrase_prompt
from environments import EnvEnum

def generate_prompt_samples(env, n_samples_per_type=100):
    """
    Generate prompt samples using all 4 prompt generation templates.
    
    Args:
        env: The environment to collect data from
        n_samples_per_type: Number of samples to generate per prompt type (default: 100)
        
    Returns:
        Dictionary with prompt samples organized by prompt type
    """
    prompt_generator = [Glam_prompt, swap_prompt, xml_prompt, paraphrase_prompt]
    
    collected_data = []
    prompt_samples = {
        'Glam_prompt': [],
        'swap_prompt': [],
        'xml_prompt': [],
        'paraphrase_prompt': [],
        'all_prompts': []
    }
    
    try:
        num_envs = env.num_envs
    except AttributeError:
        # For BabyAITextEnv, use the number of environments in _env
        num_envs = len(env._env.envs)
    
    (obs, infos), ep_ret, ep_len = env.reset(), [0] * num_envs, [0] * num_envs
    transitions_buffer = [[] for _ in range(num_envs)]
    
    with tqdm(total=n_samples_per_type) as pbar:
        while len(collected_data) < n_samples_per_type:
            possible_actions = [info["possible_actions"] for info in infos]
            
            # Take random actions to collect diverse data
            actions_id = [np.random.randint(0, len(pa)) for pa in possible_actions]
            actions_command = [pa[aid] for pa, aid in zip(possible_actions, actions_id)]
            
            # Store current state for prompt generation
            for i in range(num_envs):
                collected_data.append((
                    list(transitions_buffer[i]),
                    obs[i],
                    dict(infos[i])
                ))
                
                pbar.update(1)
                
                if len(collected_data) >= n_samples_per_type:
                    break
            
            # Update transitions buffer
            for i in range(num_envs):
                transitions_buffer[i].append({"obs": obs[i], "act": actions_command[i]})
                # Keep buffer at appropriate length
                transitions_buffer[i] = transitions_buffer[i][-20:]  # Adjust history length as needed
            
            # Step the environment
            obs, r, d, infos = env.step(actions_id=actions_id, actions_command=actions_command)
            
            # Update episode tracking and reset if needed
            for i in range(num_envs):
                ep_len[i] += 1
                ep_ret[i] += r[i]
                
                if d[i] or ep_len[i] >= 50:  # Reset if done or max length reached
                    # For environment instances that need reset
                    obs_reset, infos_reset = env.reset()
                    obs[i] = obs_reset[i]
                    infos[i] = infos_reset[i]
                    ep_ret[i] = 0
                    ep_len[i] = 0
                    transitions_buffer[i] = []
    
    collected_data = collected_data[:n_samples_per_type]
    
    for generator_fn in prompt_generator:
        generator_name = generator_fn.__name__
        print(f"Generating with {generator_name}")
        
        with tqdm(total=n_samples_per_type, desc=f"{generator_name}") as pbar:
            for j in range(n_samples_per_type):
                buff, o, info = collected_data[j]
                prompt = generator_fn(buff, o, info)
                prompt_samples[generator_name].append(prompt)
                prompt_samples['all_prompts'].append(prompt)
                pbar.update(1)
    
    print(f"Generated {len(prompt_samples['all_prompts'])} total prompts")
    return prompt_samples

@hydra.main(config_path='configs', config_name='local_gpu_config')
def main(config_args):
    seed = config_args.rl_script_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    name_env = config_args.rl_script_args.name_environment
    envs = EnvEnum[name_env].value(config_args.rl_script_args)
    
    samples = generate_prompt_samples(envs, n_samples_per_type=100)
    
    output_dir = config_args.rl_script_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/prompt_samples.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(samples, f)
    
    print(f"Saved prompt samples to {output_file}")

if __name__ == '__main__':
    main()