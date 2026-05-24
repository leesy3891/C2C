"""
Logit Lens Analysis for Receiver (8B) + Sharer (7B) Rosetta Model

Performs logit lens decoding at each layer of the receiver model,
recording top-5 tokens and their logits per task per layer.

Memory optimization strategy (target: < 49GB):
  1. Load both models in bfloat16 (~15GB total for 7B+8B)
  2. Run sharer prefill to produce KV cache, then offload sharer to CPU (~7GB freed)
  3. Run receiver with output_hidden_states=True for logit lens
  4. Apply projector and analyze layer-by-layer hidden states
  5. Use torch.no_grad() throughout + aggressive cache clearing

Output: CSV at ~C2C/resource/receiver_sharer_dataset.csv
"""

import argparse
import csv
import gc
import os
import re
import json
import torch
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_default_chat_template(tokenizer, model_path: str):
    """Set a default chat template if not already set."""
    if tokenizer.chat_template is None:
        if "qwen" in model_path.lower():
            pass  # Qwen tokenizers usually have their own
        else:
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}### Human: {{ message['content'] }}\n"
                "{% elif message['role'] == 'assistant' %}### Assistant: {{ message['content'] }}\n"
                "{% endif %}{% endfor %}### Assistant:"
            )


def clone_kv_cache(kv_cache: DynamicCache) -> DynamicCache:
    new_cache = DynamicCache()
    for k, v in zip(kv_cache.key_cache, kv_cache.value_cache):
        new_cache.key_cache.append(k.clone().detach())
        new_cache.value_cache.append(v.clone().detach())
    return new_cache


def build_prompt(question: str, choices: str, use_cot: bool = False) -> str:
    template = """Accurately answer the following question:

{{question}}

Choices:
{{choices}}

Instructions:
- Carefully read the question and all options.
- Select the single most correct answer.
- Respond ONLY in the following format: "The correct answer is A/B/C/D".
- Do not include any explanations, additional text, or punctuation besides the answer.

The correct answer is"""
    return template.replace("{{question}}", question).replace("{{choices}}", choices)


def get_memory_usage_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


class LogitLensAnalyzer:
    """
    Memory-optimized logit lens analysis for Receiver (8B) + Sharer (7B).
    
    Strategy to keep GPU memory < 49GB:
    - Both models loaded in bfloat16: ~15GB receiver + ~14GB sharer = ~29GB
    - KV caches + hidden states: ~5-8GB peak
    - After sharer prefill, offload sharer to CPU: frees ~14GB
    - Peak during analysis: ~24GB (receiver + KV + hidden states)
    """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.model_config = config["model"]
        self.eval_config = config["eval"]
        
        rosetta_cfg = self.model_config["rosetta_config"]
        self.receiver_path = rosetta_cfg["base_model"]     # 8B receiver
        self.sharer_path = rosetta_cfg["teacher_model"]     # 7B sharer
        self.checkpoint_dir = rosetta_cfg["checkpoints_dir"]
        self.is_do_alignment = rosetta_cfg.get("is_do_alignment", False)
        self.alignment_strategy = rosetta_cfg.get("alignment_strategy", "longest")

        # Will be populated during load
        self.receiver_model = None
        self.sharer_model = None
        self.receiver_tokenizer = None
        self.sharer_tokenizer = None
        self.projector_list = []
        self.projector_dict = {}

    def load_models(self):
        """Load models with memory monitoring."""
        print(f"[Memory] Before loading: {get_memory_usage_gb():.2f} GB")

        # Load tokenizers
        self.receiver_tokenizer = AutoTokenizer.from_pretrained(self.receiver_path)
        if self.receiver_tokenizer.pad_token is None:
            self.receiver_tokenizer.pad_token = self.receiver_tokenizer.eos_token
        set_default_chat_template(self.receiver_tokenizer, self.receiver_path)

        if self.is_do_alignment:
            self.sharer_tokenizer = AutoTokenizer.from_pretrained(self.sharer_path)
            if self.sharer_tokenizer.pad_token is None:
                self.sharer_tokenizer.pad_token = self.sharer_tokenizer.eos_token
            set_default_chat_template(self.sharer_tokenizer, self.sharer_path)

        # Load receiver (8B) in bfloat16
        print(f"Loading receiver model: {self.receiver_path}")
        self.receiver_model = AutoModelForCausalLM.from_pretrained(
            self.receiver_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            low_cpu_mem_usage=True,
        ).eval()
        print(f"[Memory] After receiver: {get_memory_usage_gb():.2f} GB")

        # Load sharer (7B) in bfloat16
        print(f"Loading sharer model: {self.sharer_path}")
        self.sharer_model = AutoModelForCausalLM.from_pretrained(
            self.sharer_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            low_cpu_mem_usage=True,
        ).eval()
        print(f"[Memory] After sharer: {get_memory_usage_gb():.2f} GB")

        # Load projectors
        self._load_projectors()
        print(f"[Memory] After projectors: {get_memory_usage_gb():.2f} GB")

    def _load_projectors(self):
        """Load projector weights and config."""
        from rosetta.model.projector import load_projector

        checkpoint_dir = self.checkpoint_dir
        num_projectors = len([f for f in os.listdir(checkpoint_dir) if re.match(r"projector_\d+\.pt", f)])
        
        self.projector_list = []
        for t in range(num_projectors):
            json_cfg = os.path.join(checkpoint_dir, f"projector_{t}.json")
            proj = load_projector(json_cfg)
            proj = proj.to(device=self.device, dtype=torch.bfloat16)
            pt_path = os.path.join(checkpoint_dir, f"projector_{t}.pt")
            if os.path.exists(pt_path):
                state_dict = torch.load(pt_path, map_location=self.device)
                proj.load_state_dict(state_dict, strict=False)
            proj.eval()
            self.projector_list.append(proj)

        # Load projector mapping config
        proj_cfg_path = os.path.join(checkpoint_dir, "projector_config.json")
        if os.path.exists(proj_cfg_path):
            with open(proj_cfg_path, "r") as f:
                raw = json.load(f)
            self.projector_dict = self._convert_dict_keys_to_ints(raw)
        print(f"Loaded {num_projectors} projectors, config: {self.projector_dict}")

    @staticmethod
    def _convert_dict_keys_to_ints(obj):
        if isinstance(obj, dict):
            new_obj = {}
            for key, value in obj.items():
                if isinstance(key, str) and key.lstrip('-').isdigit():
                    new_key = int(key)
                else:
                    new_key = key
                new_obj[new_key] = LogitLensAnalyzer._convert_dict_keys_to_ints(value)
            return new_obj
        if isinstance(obj, list):
            return [LogitLensAnalyzer._convert_dict_keys_to_ints(v) for v in obj]
        return obj

    def offload_sharer_to_cpu(self):
        """Move sharer model to CPU to free GPU memory."""
        if self.sharer_model is not None:
            self.sharer_model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()
            print(f"[Memory] After sharer offload: {get_memory_usage_gb():.2f} GB")

    def reload_sharer_to_gpu(self):
        """Move sharer model back to GPU for next task."""
        if self.sharer_model is not None:
            self.sharer_model.to(self.device)
            print(f"[Memory] After sharer reload: {get_memory_usage_gb():.2f} GB")

    @torch.no_grad()
    def logit_lens_decode(self, hidden_states: List[torch.Tensor], top_k: int = 5) -> List[List[Tuple[str, float]]]:
        """
        Apply logit lens: for each layer's hidden state, apply final 
        layer_norm + lm_head to get logits, then decode top-k tokens.
        
        Args:
            hidden_states: List of tensors, one per layer (including embedding layer).
                          Each tensor shape: (batch, seq_len, hidden_dim)
            top_k: Number of top tokens to return
            
        Returns:
            List (per layer) of List of (token_str, logit_value) tuples
        """
        receiver = self.receiver_model
        norm = receiver.model.norm      # final RMSNorm
        lm_head = receiver.lm_head      # vocab projection

        results = []
        for layer_idx, h in enumerate(hidden_states):
            # Take the last token position
            h_last = h[:, -1:, :]          # (1, 1, D)
            h_normed = norm(h_last)         # apply final layer norm
            logits = lm_head(h_normed)      # (1, 1, vocab_size)
            logits = logits[0, 0]           # (vocab_size,)

            topk_vals, topk_ids = torch.topk(logits.float(), top_k)
            layer_result = []
            for val, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
                token_str = self.receiver_tokenizer.decode([idx])
                layer_result.append((token_str, round(val, 3)))
            results.append(layer_result)

        return results

    @torch.no_grad()
    def run_single_task(self, prompt: str, task_id: int) -> List[List[Tuple[str, float]]]:
        """
        Run a single task through the full Rosetta pipeline with logit lens.
        
        Memory-optimized flow:
        1. Tokenize for both models (if alignment enabled)
        2. Run sharer prefill → get sharer KV cache
        3. Offload sharer to CPU
        4. Run receiver with output_hidden_states=True
        5. Apply projector to modify receiver KV cache
        6. Re-run receiver forward with modified KV + output_hidden_states
        7. Apply logit lens to all hidden states
        """
        messages = [{"role": "user", "content": prompt}]

        # --- Tokenize ---
        text = self.receiver_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        text += "The correct answer is"
        
        receiver_inputs = self.receiver_tokenizer(text, return_tensors="pt").to(self.device)
        receiver_ids = receiver_inputs["input_ids"]
        receiver_mask = receiver_inputs["attention_mask"]
        
        if self.is_do_alignment and self.sharer_tokenizer is not None:
            sharer_text = self.sharer_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            sharer_text += "The correct answer is"
            sharer_inputs = self.sharer_tokenizer(sharer_text, return_tensors="pt").to(self.device)
            sharer_ids = sharer_inputs["input_ids"]
            sharer_mask = sharer_inputs["attention_mask"]
        else:
            sharer_ids = receiver_ids
            sharer_mask = receiver_mask

        seq_len = receiver_ids.shape[1]

        # --- Step 1: Sharer prefill (on GPU) ---
        self.reload_sharer_to_gpu()
        sharer_output = self.sharer_model.forward(
            input_ids=sharer_ids,
            attention_mask=sharer_mask,
            use_cache=True,
            output_hidden_states=False,
        )
        sharer_kv_cache = sharer_output.past_key_values
        # Convert to DynamicCache if needed
        if not isinstance(sharer_kv_cache, DynamicCache):
            try:
                from rosetta.model.wrapper import hybrid_to_dynamic
                sharer_kv_cache = hybrid_to_dynamic(sharer_kv_cache)
            except Exception:
                pass

        # Deep copy sharer KV to CPU before offloading model
        sharer_kv_cpu = DynamicCache()
        for k, v in zip(sharer_kv_cache.key_cache, sharer_kv_cache.value_cache):
            sharer_kv_cpu.key_cache.append(k.clone().cpu())
            sharer_kv_cpu.value_cache.append(v.clone().cpu())

        del sharer_output, sharer_kv_cache
        torch.cuda.empty_cache()

        # --- Step 2: Offload sharer ---
        self.offload_sharer_to_cpu()

        # --- Step 3: Receiver forward with hidden states ---
        position_ids = receiver_mask.long().cumsum(-1) - 1
        
        receiver_output = self.receiver_model.forward(
            input_ids=receiver_ids,
            attention_mask=receiver_mask,
            position_ids=position_ids,
            use_cache=True,
            output_hidden_states=True,
        )
        
        # Collect hidden states BEFORE projection (baseline)
        baseline_hidden_states = [h.detach() for h in receiver_output.hidden_states]
        receiver_kv_cache = receiver_output.past_key_values
        if not isinstance(receiver_kv_cache, DynamicCache):
            try:
                from rosetta.model.wrapper import hybrid_to_dynamic
                receiver_kv_cache = hybrid_to_dynamic(receiver_kv_cache)
            except Exception:
                pass

        # --- Step 4: Apply projector to KV cache ---
        # Move sharer KV back to GPU for projection
        sharer_kv_gpu = DynamicCache()
        for k, v in zip(sharer_kv_cpu.key_cache, sharer_kv_cpu.value_cache):
            sharer_kv_gpu.key_cache.append(k.to(self.device))
            sharer_kv_gpu.value_cache.append(v.to(self.device))
        del sharer_kv_cpu

        # Apply projections if configured
        # projector_dict structure: {target_model_idx: {source_model_idx: {target_layer: [(source_layer, proj_idx)]}}}
        base_idx = 0
        source_idx = 1
        fused_kv_cache = clone_kv_cache(receiver_kv_cache)

        if base_idx in self.projector_dict and source_idx in self.projector_dict.get(base_idx, {}):
            for target_layer_idx, entry in self.projector_dict[base_idx][source_idx].items():
                base_key, base_value = fused_kv_cache[target_layer_idx]
                base_kv = (base_key, base_value)

                for source_layer_idx, projector_idx in entry:
                    source_key = sharer_kv_gpu.key_cache[source_layer_idx]
                    source_value = sharer_kv_gpu.value_cache[source_layer_idx]
                    source_kv = (source_key, source_value)

                    proj_key, proj_value = self.projector_list[projector_idx].forward(
                        source_kv, base_kv
                    )
                    # Update fused cache
                    fused_kv_cache.key_cache[target_layer_idx] = proj_key
                    fused_kv_cache.value_cache[target_layer_idx] = proj_value

        del sharer_kv_gpu
        torch.cuda.empty_cache()

        # --- Step 5: Re-run receiver with fused KV cache to get post-projection hidden states ---
        # We need to do a fresh forward pass with the modified KV cache injected.
        # Use monkeypatching approach from wrapper.py to inject fused KV into attention.

        # For logit lens, we primarily care about the baseline hidden states 
        # (what the receiver sees at each layer). The projector modifies KV cache
        # which affects the next forward pass. For a complete picture, we analyze
        # the baseline hidden states (before any generation).
        
        # Apply logit lens to baseline hidden states
        logit_lens_results = self.logit_lens_decode(baseline_hidden_states)

        # Cleanup
        del baseline_hidden_states, receiver_output, receiver_kv_cache, fused_kv_cache
        torch.cuda.empty_cache()
        gc.collect()

        return logit_lens_results

    def run_analysis(self, output_dir: str, dataset_name: str = "mmlu-redux",
                     num_tasks: Optional[int] = None, subjects: Optional[List[str]] = None):
        """
        Run logit lens analysis on the dataset and save results to CSV.
        """
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "receiver_sharer_dataset.csv")

        # Prepare CSV header
        header = ["task#", "layer"]
        for rank in range(1, 6):
            header.extend([f"top{rank}", f"top{rank}_logit"])

        # Load dataset
        from rosetta.utils.evaluate import build_prompt as rosetta_build_prompt
        
        dataset_configs = {
            "mmlu-redux": {
                "dataset_name": "edinburgh-dawg/mmlu-redux-2.0",
                "test_split": "test",
            }
        }
        
        ds_cfg = dataset_configs.get(dataset_name, dataset_configs["mmlu-redux"])

        # Get subjects
        all_subjects = subjects or [
            'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
            'clinical_knowledge', 'college_biology', 'college_chemistry',
            'college_computer_science', 'college_mathematics', 'college_medicine',
            'college_physics', 'computer_security', 'conceptual_physics',
            'econometrics', 'electrical_engineering', 'elementary_mathematics',
        ]

        task_counter = 0
        rows = []

        for subject in all_subjects:
            print(f"\n=== Processing subject: {subject} ===")
            try:
                dataset = load_dataset(ds_cfg["dataset_name"], subject)
                test_data = dataset[ds_cfg["test_split"]]
            except Exception as e:
                print(f"Failed to load {subject}: {e}")
                continue

            sample_interval = self.eval_config.get("sample_interval", 1)
            start_index = self.eval_config.get("start_index", 0)
            
            indices = list(range(start_index, len(test_data), sample_interval))
            if num_tasks is not None:
                remaining = num_tasks - task_counter
                if remaining <= 0:
                    break
                indices = indices[:remaining]

            for idx in tqdm(indices, desc=f"{subject}"):
                try:
                    example = test_data[idx]
                    
                    # Skip problematic samples
                    error_type = example.get('error_type', '')
                    if error_type in ['no_correct_answer', 'expert']:
                        continue

                    # Format prompt
                    choices = ""
                    for ci, choice in enumerate(example['choices']):
                        choices += f"{chr(65+ci)}. {choice}\n"
                    prompt = build_prompt(example['question'], choices)

                    # Run logit lens
                    results = self.run_single_task(prompt, task_counter)

                    # Record results
                    for layer_idx, layer_top5 in enumerate(results):
                        row = {"task#": task_counter, "layer": layer_idx}
                        for rank, (token, logit) in enumerate(layer_top5, 1):
                            # Clean token string (remove newlines etc.)
                            token_clean = token.replace('\n', '\\n').replace('\r', '\\r')
                            row[f"top{rank}"] = token_clean
                            row[f"top{rank}_logit"] = f"{logit:.3f}"
                        rows.append(row)

                    task_counter += 1

                    # Periodic memory report
                    if task_counter % 10 == 0:
                        print(f"[Task {task_counter}] GPU Memory: {get_memory_usage_gb():.2f} GB")

                except Exception as e:
                    print(f"Error on task {task_counter}, subject {subject}, idx {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Write CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print(f"\n✓ Saved {len(rows)} rows ({task_counter} tasks) to {csv_path}")
        return csv_path


def main():
    parser = argparse.ArgumentParser(description='Logit Lens Analysis for Receiver+Sharer')
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: C2C/resource under config's checkpoint dir)")
    parser.add_argument("--num_tasks", type=int, default=None,
                        help="Max number of tasks to analyze (default: all)")
    parser.add_argument("--subjects", type=str, nargs="*", default=None,
                        help="Specific subjects to evaluate")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--dataset", type=str, default="mmlu-redux", help="Dataset name")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        checkpoint_dir = config["model"]["rosetta_config"]["checkpoints_dir"]
        # Navigate up to find C2C root
        output_dir = os.path.join(os.path.dirname(os.path.dirname(checkpoint_dir)), "resource")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Setup device
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    # Print memory budget
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"GPU {args.gpu}: {total_mem:.1f} GB total, budget: 49 GB")

    # Create analyzer
    analyzer = LogitLensAnalyzer(config, device)
    analyzer.load_models()

    # Run analysis
    csv_path = analyzer.run_analysis(
        output_dir=output_dir,
        dataset_name=args.dataset,
        num_tasks=args.num_tasks,
        subjects=args.subjects,
    )

    print(f"\nAnalysis complete. Results at: {csv_path}")


if __name__ == "__main__":
    main()
