import argparse
import re
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from rosetta.model.projector import create_projector
from rosetta.model.wrapper import RosettaModel
from rosetta.train.dataset_adapters import MMLUChatDataset

def load_qwen_model(model_name):
    model_path = "Qwen/" + model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).eval().to(DEVICE)
    return model, tokenizer

def load_rosetta_model(checkpoint_dir):
    """Load Rosetta model using the same approach as evaluate.py"""
    from rosetta.model.projector import load_projector
    import re
    
    slm_model_path = "Qwen/Qwen3-8B"
    llm_model_path = "Qwen/Qwen2.5-7B-Instruct"

    # Load tokenizer
    slm_tokenizer = AutoTokenizer.from_pretrained(slm_model_path)
    if slm_tokenizer.pad_token is None:
        slm_tokenizer.pad_token = slm_tokenizer.eos_token
    
    # Load models
    slm_model = AutoModelForCausalLM.from_pretrained(
        slm_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE}
    ).eval()
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE}
    ).eval()
    
    # Load projectors
    num_projectors = len([f for f in os.listdir(checkpoint_dir) if re.match(r"projector_\d+\.pt", f)])
    projector_list = []
    for t in range(num_projectors):
        json_cfg = os.path.join(checkpoint_dir, f"projector_{t}.json")
        proj = load_projector(json_cfg)
        proj = proj.to(DEVICE)
        pt_path = os.path.join(checkpoint_dir, f"projector_{t}.pt")
        if os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location=DEVICE)
            proj.load_state_dict(state_dict, strict=False)
        projector_list.append(proj)
    
    # Initialize Rosetta model
    rosetta_model = RosettaModel(
        model_list=[slm_model, llm_model],
        base_model_idx=0,
        projector_list=projector_list,
    ).to(DEVICE).eval()

    # Load projector mapping configs
    proj_cfg_path = os.path.join(checkpoint_dir, "projector_config.json")
    rosetta_model.load_projector_config(proj_cfg_path)

    return rosetta_model, slm_tokenizer

def extract_k_cache(model, tokenizer, dataset, layer_idx, num_samples=10):
    all_values = []
    subset = [dataset[i] for i in range(0, min(num_samples, len(dataset)))]
    for i, sample in enumerate(tqdm(subset, desc=f"Layer {layer_idx}")):
        instruction = tokenizer.apply_chat_template(sample[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input = tokenizer(instruction, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(input['input_ids'].shape[1], 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if isinstance(model, RosettaModel):
                output = model(**input, kv_cache_index=[instruction_index], use_cache=True)
            else:
                output = model(**input, use_cache=True)

        # (batch, num_heads, seq_len, head_dim)
        k_value = output.past_key_values[layer_idx][0].squeeze(0).float().cpu()  # (num_heads, seq_len, head_dim)
        
        # For each token, flatten across heads and head_dim
        # k_value: (num_heads, seq_len, head_dim) -> (seq_len, num_heads * head_dim)
        k_value_flat = k_value.permute(1, 0, 2).reshape(k_value.shape[1], -1)  # (seq_len, num_heads * head_dim)
        
        # Add all tokens for this sample
        all_values.append(k_value_flat.numpy())

    return all_values

def extract_v_cache(model, tokenizer, dataset, layer_idx, num_samples=20):
    all_values = []
    subset = [dataset[i] for i in range(0, min(num_samples, len(dataset)))]
    for i, sample in enumerate(tqdm(subset, desc=f"Layer {layer_idx}")):
        instruction = tokenizer.apply_chat_template(sample[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input = tokenizer(instruction, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(input['input_ids'].shape[1], 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if isinstance(model, RosettaModel):
                output = model(**input, kv_cache_index=[instruction_index], use_cache=True)
            else:
                output = model(**input, use_cache=True)

        # (batch, num_heads, seq_len, head_dim)
        v_value = output.past_key_values[layer_idx][1].squeeze(0).float().cpu()  # (num_heads, seq_len, head_dim)
        
        # For each token, flatten across heads and head_dim
        # v_value: (num_heads, seq_len, head_dim) -> (seq_len, num_heads * head_dim)
        v_value_flat = v_value.permute(1, 0, 2).reshape(v_value.shape[1], -1)  # (seq_len, num_heads * head_dim)
        
        # Add all tokens for this sample
        all_values.append(v_value_flat.numpy())

    return all_values

def align_dimensions(all_embeddings, target_dim=None):
    """
    Align embedding dimensions across models using PCA.
    Different models may have different num_kv_heads, resulting in different
    flattened KV cache dimensions (e.g., Qwen3-8B: 1024, Qwen2.5-7B: 512).
    This projects all embeddings to a common dimension space.
    """
    dims = [emb[0].shape[1] for emb in all_embeddings if len(emb) > 0]
    if len(set(dims)) == 1:
        return all_embeddings  # Already aligned

    if target_dim is None:
        target_dim = min(dims)

    aligned = []
    for emb_list in all_embeddings:
        d = emb_list[0].shape[1]
        if d == target_dim:
            aligned.append(emb_list)
        else:
            # Fit PCA on all tokens from this model, then transform each sample
            all_tokens = np.concatenate(emb_list, axis=0)
            pca = PCA(n_components=target_dim)
            pca.fit(all_tokens)
            aligned.append([pca.transform(arr) for arr in emb_list])
    return aligned


def plot_tsne_per_token(all_embeddings, label, model_names, layer_idx, output_path, show_correspondence=True):
    # Align dimensions across models (e.g., 1024 vs 512)
    all_embeddings = align_dimensions(all_embeddings)

    tsne = TSNE(n_components=2, perplexity=30, random_state=2)
    
    # Flatten all embeddings from all models and samples
    X = np.concatenate([np.concatenate(emb, axis=0) for emb in all_embeddings], axis=0)
    tsne_result = tsne.fit_transform(X)

    color_labels = []
    token_indices = []
    current_idx = 0
    
    for i, emb in enumerate(all_embeddings):
        for sample_idx, arr in enumerate(emb):
            total_tokens = arr.shape[0]
            color_labels.extend([model_names[i]] * total_tokens)
            for token_idx in range(total_tokens):
                token_indices.append({
                    'model_idx': i,
                    'model_name': model_names[i],
                    'sample_idx': sample_idx,
                    'token_idx': token_idx,
                    'global_idx': current_idx + token_idx
                })
            current_idx += total_tokens

    plt.figure(figsize=(15, 10))
    
    for model_name in set(color_labels):
        indices = [j for j, lbl in enumerate(color_labels) if lbl == model_name]
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], 
                   label=model_name, s=8, alpha=0.7)

    plt.title(f"TSNE of {label} Cache (Layer {layer_idx}) - Per Token")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"tsne_layer_{layer_idx}_{label}_per_token.png"), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: tsne_layer_{layer_idx}_{label}_per_token.png")
    
    if show_correspondence:
        plot_correspondence_lines(tsne_result, token_indices, model_names, label)
        
        plt.title(f"TSNE of {label} Cache (Layer {layer_idx}) - Per Token with Correspondence")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"tsne_layer_{layer_idx}_{label}_per_token_with_correspondence.png"), 
                    dpi=300, bbox_inches='tight')
        print(f"Saved: tsne_layer_{layer_idx}_{label}_per_token_with_correspondence.png")


def plot_tsne_per_sequence(all_embeddings, label, model_names, layer_idx, output_path, show_correspondence=True):
    # Align dimensions across models (e.g., 1024 vs 512)
    all_embeddings = align_dimensions(all_embeddings)

    tsne = TSNE(n_components=2, perplexity=min(30, len(all_embeddings[0])-1), random_state=42)
    
    sequence_embeddings = []
    sequence_indices = []
    
    for i, emb in enumerate(all_embeddings):
        for sample_idx, arr in enumerate(emb):
            seq_avg = np.mean(arr, axis=0)
            sequence_embeddings.append(seq_avg)
            sequence_indices.append({
                'model_idx': i,
                'model_name': model_names[i],
                'sample_idx': sample_idx,
                'global_idx': len(sequence_embeddings) - 1
            })
    
    X = np.array(sequence_embeddings)
    tsne_result = tsne.fit_transform(X)
    
    color_labels = [info['model_name'] for info in sequence_indices]
    
    plt.figure(figsize=(15, 10))
    
    for model_name in set(color_labels):
        indices = [j for j, lbl in enumerate(color_labels) if lbl == model_name]
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], 
                   label=model_name, s=50, alpha=0.7)

    plt.title(f"TSNE of {label} Cache (Layer {layer_idx}) - Per Sequence")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"tsne_layer_{layer_idx}_{label}_per_sequence.png"), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: tsne_layer_{layer_idx}_{label}_per_sequence.png")
    
    if show_correspondence:
        plot_sequence_correspondence_lines(tsne_result, sequence_indices, model_names, label)
        
        plt.title(f"TSNE of {label} Cache (Layer {layer_idx}) - Per Sequence with Correspondence")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"tsne_layer_{layer_idx}_{label}_per_sequence_with_correspondence.png"), 
                    dpi=300, bbox_inches='tight')
        print(f"Saved: tsne_layer_{layer_idx}_{label}_per_sequence_with_correspondence.png")


def plot_sequence_correspondence_lines(tsne_result, sequence_indices, model_names, label):
    model_idx_map = {name: idx for idx, name in enumerate(model_names)}
    
    # Qwen3-8B (base) <-> Rosetta correspondence
    if 'Qwen3-8B' in model_idx_map and 'Rosetta' in model_idx_map:
        plot_sequence_model_correspondence(tsne_result, sequence_indices, 
                                         'Qwen3-8B', 'Rosetta', 'blue', alpha=0.5)
    
    # Rosetta <-> Qwen2.5-7B-Instruct (teacher) correspondence
    if 'Rosetta' in model_idx_map and 'Qwen2.5-7B-Instruct' in model_idx_map:
        plot_sequence_model_correspondence(tsne_result, sequence_indices, 
                                         'Rosetta', 'Qwen2.5-7B-Instruct', 'red', alpha=0.5)


def plot_sequence_model_correspondence(tsne_result, sequence_indices, model1_name, model2_name, color, alpha=0.5):
    model1_sequences = {}
    model2_sequences = {}
    
    for seq_info in sequence_indices:
        sample_idx = seq_info['sample_idx']
        if seq_info['model_name'] == model1_name:
            model1_sequences[sample_idx] = seq_info['global_idx']
        elif seq_info['model_name'] == model2_name:
            model2_sequences[sample_idx] = seq_info['global_idx']
    
    for sample_idx in model1_sequences:
        if sample_idx in model2_sequences:
            idx1 = model1_sequences[sample_idx]
            idx2 = model2_sequences[sample_idx]
            
            x1, y1 = tsne_result[idx1]
            x2, y2 = tsne_result[idx2]
            
            plt.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=1.5)
    
    plt.plot([], [], color=color, alpha=alpha, linewidth=2, 
             label=f'{model1_name} ↔ {model2_name} correspondence')


def plot_correspondence_lines(tsne_result, token_indices, model_names, label):
    model_idx_map = {name: idx for idx, name in enumerate(model_names)}
    
    # Qwen3-8B (base) <-> Rosetta correspondence
    if 'Qwen3-8B' in model_idx_map and 'Rosetta' in model_idx_map:
        plot_model_correspondence(tsne_result, token_indices, 
                                'Qwen3-8B', 'Rosetta', 'blue', alpha=0.3)
    
    # Rosetta <-> Qwen2.5-7B-Instruct (teacher) correspondence
    if 'Rosetta' in model_idx_map and 'Qwen2.5-7B-Instruct' in model_idx_map:
        plot_model_correspondence(tsne_result, token_indices, 
                                'Rosetta', 'Qwen2.5-7B-Instruct', 'red', alpha=0.3)


def plot_model_correspondence(tsne_result, token_indices, model1_name, model2_name, color, alpha=0.3):
    model1_tokens = {}
    model2_tokens = {}
    
    for token_info in token_indices:
        key = (token_info['sample_idx'], token_info['token_idx'])
        if token_info['model_name'] == model1_name:
            model1_tokens[key] = token_info['global_idx']
        elif token_info['model_name'] == model2_name:
            model2_tokens[key] = token_info['global_idx']
    
    for key in model1_tokens:
        if key in model2_tokens:
            idx1 = model1_tokens[key]
            idx2 = model2_tokens[key]
            
            x1, y1 = tsne_result[idx1]
            x2, y2 = tsne_result[idx2]
            
            plt.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=0.5)
    
    plt.plot([], [], color=color, alpha=alpha, linewidth=2, 
             label=f'{model1_name} ↔ {model2_name} correspondence')


def free_model(model):
    """Delete model and free GPU memory"""
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


def main(args):
    global DEVICE
    if 'device' in args and args['device'] is not None:
        DEVICE = args['device']
        print(f"Using specified device: {DEVICE}")
    
    # Changed: OpenBookChatDataset instead of MMLUChatDataset
    dataset = MMLUChatDataset(split="test", num_samples=None)

    os.makedirs(args['output_dir'], exist_ok=True)

    # Changed: Analyze last 7 layers of 36-layer Qwen3-8B (layers 29-35)
    layers_to_analyze = [29, 30, 31, 32, 33, 34, 35]
    # Offsets: [Rosetta(base=8B), Qwen3-8B, Qwen2.5-7B-Instruct]
    # Rosetta & Qwen3-8B both have 36 layers -> offset 0
    # Qwen2.5-7B-Instruct has 28 layers -> offset -8 (layer 29-8=21 maps to equivalent depth)
    layer_idx_offset_list = [0, 0, -8]

    if args.get('mode', 'both') in ['sequence', 'both']:
        num_samples = args.get('num_samples') or 100
    else:
        num_samples = args.get('num_samples') or 10

    # ---- Extract KV cache one model at a time to avoid OOM ----
    # k_cache_per_model[model_idx][layer_idx] = list of arrays
    k_cache_per_model = {i: {} for i in range(len(args['models']))}

    for model_idx, (model_path, layer_offset) in enumerate(zip(args['models'], layer_idx_offset_list)):
        print(f"\n{'='*60}")
        print(f"Loading model [{model_idx+1}/{len(args['models'])}]: {model_path}")
        print(f"{'='*60}")

        if "Rosetta" in model_path:
            model, tokenizer = load_rosetta_model("local/checkpoints/C2C_Fuser/qwen3_8b+qwen2.5_7b_Fuser/final")
        else:
            model, tokenizer = load_qwen_model(model_path)
        model.eval()

        for layer_idx in layers_to_analyze:
            actual_layer = layer_idx + layer_offset
            print(f"  Extracting K cache: layer {layer_idx} (actual={actual_layer})")
            k_values = extract_k_cache(model, tokenizer, dataset, layer_idx=actual_layer, num_samples=num_samples)
            k_cache_per_model[model_idx][layer_idx] = k_values

        # Free GPU memory before loading the next model
        print(f"  Unloading {model_path} to free GPU memory...")
        free_model(model)

    # ---- Plot t-SNE (all data is on CPU now, GPU is free) ----
    print(f"\n{'='*60}")
    print("Generating t-SNE plots...")
    print(f"{'='*60}")

    for layer_idx in layers_to_analyze:
        k_layer_embeddings = [k_cache_per_model[i][layer_idx] for i in range(len(args['models']))]

        if args.get('mode', 'both') in ['token', 'both']:
            plot_tsne_per_token(k_layer_embeddings, "k", args['models'], layer_idx, args['output_dir'], 
                               args.get('show_correspondence', True))
        
        if args.get('mode', 'both') in ['sequence', 'both']:
            plot_tsne_per_sequence(k_layer_embeddings, "k", args['models'], layer_idx, args['output_dir'], 
                                  args.get('show_correspondence', True))


if __name__ == "__main__":
    """
    Usage examples:
    python tsne.py                                    # Auto-detect device, show correspondence, both modes
    python tsne.py --device cuda                     # Use CUDA
    python tsne.py --device cpu                      # Use CPU
    python tsne.py --output_dir my_plots             # Custom output directory
    python tsne.py --models Rosetta Qwen3-8B         # Specify models to analyze
    python tsne.py --no-correspondence               # Disable correspondence lines
    python tsne.py --mode token                      # Token-level only (10 samples)
    python tsne.py --mode sequence                   # Sequence-level only (50 samples)
    python tsne.py --mode both                       # Both modes (default)
    python tsne.py --num_samples 100                 # Use 100 samples
    python tsne.py --mode sequence --num_samples 30  # Sequence mode with 30 samples
    """
    parser = argparse.ArgumentParser(description='Generate t-SNE plots for KV cache analysis')
    parser.add_argument('--device', type=str, default=None, 
                       help='Device to use (cuda, mps, cpu). If not specified, auto-detect.')
    parser.add_argument('--output_dir', type=str, default="tsne_outputs_openbookqa",
                       help='Output directory for t-SNE plots')
    # Changed: Default models to match Fuser config (Qwen3-8B + Qwen2.5-7B-Instruct)
    parser.add_argument('--models', nargs='+', 
                       default=["Rosetta", "Qwen3-8B", "Qwen2.5-7B-Instruct"],
                       help='Models to analyze')
    parser.add_argument('--no-correspondence', action='store_true',
                       help='Disable correspondence lines between models')
    parser.add_argument('--mode', type=str, choices=['token', 'sequence', 'both'], 
                       default='both',
                       help='Visualization mode: token (per-token), sequence (per-sequence), or both (default)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to use. If not specified, uses 10 for token mode and 50 for sequence mode')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            DEVICE = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            DEVICE = "mps"
        else:
            DEVICE = "cpu"
    else:
        DEVICE = args.device
    
    # Convert to dict for compatibility
    args_dict = {
        'device': args.device,
        'output_dir': args.output_dir,
        'models': args.models,
        'show_correspondence': not args.no_correspondence,
        'mode': args.mode,
        'num_samples': args.num_samples
    }

    main(args_dict)
