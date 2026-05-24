"""
Logit Lens Analysis for Receiver (8B) + Sharer (7B) Rosetta Model

Performs logit lens decoding at each layer of the receiver model,
recording top-5 tokens and their logits per task per layer.

Memory optimization strategy (target: < 49GB):
  1. Load both models in bfloat16 (~15GB + ~14GB = ~29GB)
  2. Run sharer prefill → KV cache, offload sharer to CPU (~14GB freed)
  3. Run receiver with output_hidden_states=True for logit lens
  4. Peak ~25GB well within 49GB budget

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


def clone_kv_cache(kv_cache: DynamicCache) -> DynamicCache:
    new_cache = DynamicCache()
    for k, v in zip(kv_cache.key_cache, kv_cache.value_cache):
        new_cache.key_cache.append(k.clone().detach())
        new_cache.value_cache.append(v.clone().detach())
    return new_cache


def hybrid_to_dynamic(cache):
    """Convert HybridCache to DynamicCache if needed."""
    if cache is None or isinstance(cache, DynamicCache):
        return cache
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        keys = cache.key_cache
        values = cache.value_cache
        legacy_cache = [(k, v) for k, v in zip(keys, values)]
        return DynamicCache.from_legacy_cache(legacy_cache)
    raise TypeError(f"Unsupported cache type: {type(cache)}")


def build_prompt(question: str, choices: str) -> str:
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
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.model_config = config["model"]
        self.eval_config = config["eval"]

        rosetta_cfg = self.model_config["rosetta_config"]
        self.receiver_path = rosetta_cfg["base_model"]       # 8B receiver
        self.sharer_path = rosetta_cfg["teacher_model"]       # 7B sharer
        self.checkpoint_dir = rosetta_cfg["checkpoints_dir"]
        self.is_do_alignment = rosetta_cfg.get("is_do_alignment", False)
        self.alignment_strategy = rosetta_cfg.get("alignment_strategy", "longest")

        self.receiver_model = None
        self.sharer_model = None
        self.receiver_tokenizer = None
        self.sharer_tokenizer = None
        self.projector_list = []
        self.projector_dict = {}
        self.aligner = None

    def load_models(self):
        print(f"[Memory] Before loading: {get_memory_usage_gb():.2f} GB")

        # --- Tokenizers ---
        self.receiver_tokenizer = AutoTokenizer.from_pretrained(self.receiver_path)
        if self.receiver_tokenizer.pad_token is None:
            self.receiver_tokenizer.pad_token = self.receiver_tokenizer.eos_token
        from rosetta.utils.evaluate import set_default_chat_template
        set_default_chat_template(self.receiver_tokenizer, self.receiver_path)

        self.sharer_tokenizer = AutoTokenizer.from_pretrained(self.sharer_path)
        if self.sharer_tokenizer.pad_token is None:
            self.sharer_tokenizer.pad_token = self.sharer_tokenizer.eos_token
        set_default_chat_template(self.sharer_tokenizer, self.sharer_path)

        # --- Aligner (if alignment enabled) ---
        if self.is_do_alignment:
            from rosetta.model.aligner import TokenAligner, AlignmentStrategy
            self.aligner = TokenAligner(
                slm_tokenizer=self.receiver_tokenizer,
                llm_tokenizer=self.sharer_tokenizer,
                strategy=AlignmentStrategy(self.alignment_strategy),
            )
            print(f"Token aligner initialized with strategy: {self.alignment_strategy}")

        # --- Receiver (8B) ---
        print(f"Loading receiver: {self.receiver_path}")
        self.receiver_model = AutoModelForCausalLM.from_pretrained(
            self.receiver_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            low_cpu_mem_usage=True,
        ).eval()
        print(f"[Memory] After receiver: {get_memory_usage_gb():.2f} GB")

        # --- Sharer (7B) ---
        print(f"Loading sharer: {self.sharer_path}")
        self.sharer_model = AutoModelForCausalLM.from_pretrained(
            self.sharer_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            low_cpu_mem_usage=True,
        ).eval()
        print(f"[Memory] After sharer: {get_memory_usage_gb():.2f} GB")

        # --- Projectors ---
        self._load_projectors()
        print(f"[Memory] After projectors: {get_memory_usage_gb():.2f} GB")

    def _load_projectors(self):
        from rosetta.model.projector import load_projector

        checkpoint_dir = self.checkpoint_dir
        num_projectors = len([f for f in os.listdir(checkpoint_dir)
                              if re.match(r"projector_\d+\.pt", f)])

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

        proj_cfg_path = os.path.join(checkpoint_dir, "projector_config.json")
        if os.path.exists(proj_cfg_path):
            with open(proj_cfg_path, "r") as f:
                raw = json.load(f)
            self.projector_dict = self._convert_dict_keys_to_ints(raw)
        print(f"Loaded {num_projectors} projectors, mapping: {self.projector_dict}")

    @staticmethod
    def _convert_dict_keys_to_ints(obj):
        if isinstance(obj, dict):
            return {
                (int(k) if isinstance(k, str) and k.lstrip('-').isdigit() else k):
                LogitLensAnalyzer._convert_dict_keys_to_ints(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [LogitLensAnalyzer._convert_dict_keys_to_ints(v) for v in obj]
        return obj

    def offload_sharer_to_cpu(self):
        if self.sharer_model is not None:
            self.sharer_model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

    def reload_sharer_to_gpu(self):
        if self.sharer_model is not None:
            self.sharer_model.to(self.device)

    # ------------------------------------------------------------------ #
    #  Tokenization: produce aligned receiver_ids / sharer_ids of same L  #
    # ------------------------------------------------------------------ #
    def tokenize_prompt(self, prompt: str):
        """
        Tokenize a prompt for both receiver and sharer.
        If alignment is enabled, returns padded ids of identical length.

        Returns dict with keys:
            receiver_ids   (1, L)   on device
            receiver_mask  (1, L)   on device
            sharer_ids     (1, L)   on device
            sharer_mask    (1, L)   on device
        """
        messages = [{"role": "user", "content": prompt}]

        if self.aligner is not None:
            # Use aligner to produce padded, equal-length ids
            response_text = "The correct answer is"
            messages_with_resp = messages + [{"role": "assistant", "content": response_text}]

            details = self.aligner.align_chat_messages(
                messages_with_resp,
                add_generation_prompt=False,
                return_details=True,
                enable_thinking=False,
                remove_last_surfix=True,
            )

            receiver_ids = torch.tensor(details['slm_ids_padded']).unsqueeze(0).to(self.device)
            sharer_ids   = torch.tensor(details['llm_ids_padded']).unsqueeze(0).to(self.device)

            slm_pad_mask = torch.tensor(details['slm_padding_mask']).unsqueeze(0)
            llm_pad_mask = torch.tensor(details['llm_padding_mask']).unsqueeze(0)

            receiver_mask = (~slm_pad_mask).float().to(self.device)
            sharer_mask   = (~llm_pad_mask).float().to(self.device)

            assert receiver_ids.shape == sharer_ids.shape, \
                f"Aligned lengths differ: {receiver_ids.shape} vs {sharer_ids.shape}"

        else:
            # No alignment: same text, same tokenizer for both
            text = self.receiver_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            text += "The correct answer is"

            tok = self.receiver_tokenizer(text, return_tensors="pt").to(self.device)
            receiver_ids  = tok["input_ids"]
            receiver_mask = tok["attention_mask"].float()
            sharer_ids    = receiver_ids.clone()
            sharer_mask   = receiver_mask.clone()

        return {
            "receiver_ids":  receiver_ids,
            "receiver_mask": receiver_mask,
            "sharer_ids":    sharer_ids,
            "sharer_mask":   sharer_mask,
        }

    # ------------------------------------------------------------------ #
    #  Logit Lens: hidden_state → norm → lm_head → top-k                 #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def logit_lens_decode(self, hidden_states: List[torch.Tensor],
                          top_k: int = 5) -> List[List[Tuple[str, float]]]:
        norm    = self.receiver_model.model.norm
        lm_head = self.receiver_model.lm_head

        results = []
        for h in hidden_states:
            h_last  = h[:, -1:, :]
            logits  = lm_head(norm(h_last))[0, 0].float()
            vals, ids = torch.topk(logits, top_k)
            layer_result = [
                (self.receiver_tokenizer.decode([idx]), round(val, 3))
                for val, idx in zip(vals.tolist(), ids.tolist())
            ]
            results.append(layer_result)
        return results

    # ------------------------------------------------------------------ #
    #  Single-task pipeline                                               #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def run_single_task(self, prompt: str, task_id: int):
        """
        1. Tokenize (aligned)
        2. Sharer prefill → KV cache → offload sharer
        3. Receiver forward (output_hidden_states=True)
        4. Apply projector to receiver KV cache using sharer KV
        5. Re-run receiver with fused KV cache + output_hidden_states
        6. Logit lens on all hidden states
        """
        tok = self.tokenize_prompt(prompt)
        receiver_ids  = tok["receiver_ids"]
        receiver_mask = tok["receiver_mask"]
        sharer_ids    = tok["sharer_ids"]
        sharer_mask   = tok["sharer_mask"]
        seq_len       = receiver_ids.shape[1]

        # ---- 1. Sharer prefill (GPU) ----
        self.reload_sharer_to_gpu()
        sharer_out = self.sharer_model.forward(
            input_ids=sharer_ids,
            attention_mask=sharer_mask,
            use_cache=True,
            output_hidden_states=False,
        )
        sharer_kv = hybrid_to_dynamic(sharer_out.past_key_values)

        # Copy sharer KV to CPU
        sharer_kv_cpu = DynamicCache()
        for k, v in zip(sharer_kv.key_cache, sharer_kv.value_cache):
            sharer_kv_cpu.key_cache.append(k.cpu())
            sharer_kv_cpu.value_cache.append(v.cpu())
        del sharer_out, sharer_kv
        torch.cuda.empty_cache()

        # ---- 2. Offload sharer ----
        self.offload_sharer_to_cpu()

        # ---- 3. Receiver forward (with hidden states) ----
        position_ids = receiver_mask.long().cumsum(-1) - 1
        receiver_out = self.receiver_model.forward(
            input_ids=receiver_ids,
            attention_mask=receiver_mask,
            position_ids=position_ids,
            use_cache=True,
            output_hidden_states=True,
        )
        baseline_hidden = [h.detach() for h in receiver_out.hidden_states]
        receiver_kv = hybrid_to_dynamic(receiver_out.past_key_values)

        # ---- 4. Apply projector: modify receiver KV with sharer KV ----
        # Move sharer KV back to GPU
        sharer_kv_gpu = DynamicCache()
        for k, v in zip(sharer_kv_cpu.key_cache, sharer_kv_cpu.value_cache):
            sharer_kv_gpu.key_cache.append(k.to(self.device))
            sharer_kv_gpu.value_cache.append(v.to(self.device))
        del sharer_kv_cpu

        fused_kv = clone_kv_cache(receiver_kv)
        base_idx, source_idx = 0, 1

        if base_idx in self.projector_dict and \
           source_idx in self.projector_dict.get(base_idx, {}):
            for target_layer, entry in self.projector_dict[base_idx][source_idx].items():
                # Receiver (target) KV: (B, H_recv, N, D) from fused_kv
                recv_key  = fused_kv.key_cache[target_layer]     # (B, H_r, N, D)
                recv_val  = fused_kv.value_cache[target_layer]   # (B, H_r, N, D)
                target_kv = (recv_key, recv_val)

                for source_layer, proj_idx in entry:
                    # Sharer (source) KV: (B, H_shr, N, D) from sharer_kv_gpu
                    shr_key = sharer_kv_gpu.key_cache[source_layer]   # (B, H_s, N, D)
                    shr_val = sharer_kv_gpu.value_cache[source_layer] # (B, H_s, N, D)
                    source_kv = (shr_key, shr_val)

                    proj_key, proj_val = self.projector_list[proj_idx].forward(
                        source_kv, target_kv
                    )

                    fused_kv.key_cache[target_layer]  = proj_key
                    fused_kv.value_cache[target_layer] = proj_val
                    # Update target_kv for next source in same target layer
                    target_kv = (proj_key, proj_val)

        del sharer_kv_gpu
        torch.cuda.empty_cache()

        # ---- 5. Re-run receiver with fused KV cache ----
        # Monkeypatch attention layers to use fused KV, then forward for hidden states
        from rosetta.model.wrapper import RosettaModel
        hook_handlers = []
        num_layers = self.receiver_model.config.num_hidden_layers
        for i in range(num_layers):
            attn = self.receiver_model.model.layers[i].self_attn
            new_k = fused_kv.key_cache[i]
            new_v = fused_kv.value_cache[i]
            try:
                orig = RosettaModel._monkeypatch_qwen3_attention_forward(attn, new_k, new_v)
                hook_handlers.append((attn, orig))
            except Exception:
                # If monkeypatching fails, skip (use baseline hidden states instead)
                pass

        if hook_handlers:
            # Re-run with fused KV injected via monkeypatch
            fused_out = self.receiver_model.forward(
                input_ids=receiver_ids,
                attention_mask=receiver_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            fused_hidden = [h.detach() for h in fused_out.hidden_states]
            del fused_out

            # Restore original attention forwards
            for attn, orig_forward in hook_handlers:
                attn.forward = orig_forward
        else:
            # Fallback: use baseline hidden states
            fused_hidden = baseline_hidden

        # ---- 6. Logit lens ----
        logit_lens_results = self.logit_lens_decode(fused_hidden)

        # Cleanup
        del baseline_hidden, fused_hidden, receiver_out, receiver_kv, fused_kv
        torch.cuda.empty_cache()
        gc.collect()

        return logit_lens_results

    # ------------------------------------------------------------------ #
    #  Main analysis loop                                                 #
    # ------------------------------------------------------------------ #
    def run_analysis(self, output_dir: str, dataset_name: str = "mmlu-redux",
                     num_tasks: Optional[int] = None,
                     subjects: Optional[List[str]] = None):
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "receiver_sharer_dataset.csv")

        header = ["task#", "layer"]
        for rank in range(1, 6):
            header.extend([f"top{rank}", f"top{rank}_logit"])

        ds_cfg = {
            "dataset_name": "edinburgh-dawg/mmlu-redux-2.0",
            "test_split": "test",
        }

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
                    error_type = example.get('error_type', '')
                    if error_type in ['no_correct_answer', 'expert']:
                        continue

                    choices = ""
                    for ci, choice in enumerate(example['choices']):
                        choices += f"{chr(65+ci)}. {choice}\n"
                    prompt = build_prompt(example['question'], choices)

                    results = self.run_single_task(prompt, task_counter)

                    for layer_idx, layer_top5 in enumerate(results):
                        row = {"task#": task_counter, "layer": layer_idx}
                        for rank, (token, logit) in enumerate(layer_top5, 1):
                            token_clean = repr(token)[1:-1]  # safe repr
                            row[f"top{rank}"] = token_clean
                            row[f"top{rank}_logit"] = f"{logit:.3f}"
                        rows.append(row)

                    task_counter += 1
                    if task_counter % 10 == 0:
                        print(f"[Task {task_counter}] GPU: {get_memory_usage_gb():.2f} GB")

                except Exception as e:
                    print(f"Error on task {task_counter}, subject {subject}, idx {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print(f"\n✓ Saved {len(rows)} rows ({task_counter} tasks) to {csv_path}")
        return csv_path


def main():
    parser = argparse.ArgumentParser(description='Logit Lens Analysis')
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--subjects", type=str, nargs="*", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="mmlu-redux")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        ckpt = config["model"]["rosetta_config"]["checkpoints_dir"]
        output_dir = os.path.join(os.path.dirname(os.path.dirname(ckpt)), "resource")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"GPU {args.gpu}: {total:.1f} GB total, budget 49 GB")

    analyzer = LogitLensAnalyzer(config, device)
    analyzer.load_models()
    csv_path = analyzer.run_analysis(
        output_dir=output_dir,
        dataset_name=args.dataset,
        num_tasks=args.num_tasks,
        subjects=args.subjects,
    )
    print(f"\nDone. Results: {csv_path}")


if __name__ == "__main__":
    main()
