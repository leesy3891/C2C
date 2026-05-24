"""
Logit Lens Analysis for Receiver + Sharer Rosetta Model

Supports all datasets from unified_evaluator (mmlu-redux, mmmlu, gpqa,
math-500, gsm8k, openbookqa, ai2-arc, mmlu-pro, ceval) and CoT toggle,
all controlled via YAML config.

Memory optimization: sharer offloaded to CPU after prefill (~25GB peak for 7B+8B).

Output: CSV at <output_dir>/receiver_sharer_dataset.csv
"""

import argparse
import csv
import gc
import os
import re
import json
import random
import hashlib
import torch
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset


# ===================================================================== #
#  Dataset configs (mirrored from unified_evaluator.py)                  #
# ===================================================================== #
DATASET_CONFIGS = {
    "mmlu-redux": {
        "hf_name": "edinburgh-dawg/mmlu-redux-2.0",
        "split": "test",
        "subjects": [
            'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
            'clinical_knowledge', 'college_biology', 'college_chemistry',
            'college_computer_science', 'college_mathematics', 'college_medicine',
            'college_physics', 'computer_security', 'conceptual_physics',
            'econometrics', 'electrical_engineering', 'elementary_mathematics',
            'formal_logic', 'global_facts', 'high_school_biology',
            'high_school_chemistry', 'high_school_computer_science',
            'high_school_european_history', 'high_school_geography',
            'high_school_government_and_politics', 'high_school_macroeconomics',
            'high_school_mathematics', 'high_school_microeconomics',
            'high_school_physics', 'high_school_psychology',
            'high_school_statistics', 'high_school_us_history',
            'high_school_world_history', 'human_aging', 'human_sexuality',
            'international_law', 'jurisprudence', 'logical_fallacies',
            'machine_learning', 'management', 'marketing', 'medical_genetics',
            'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
            'philosophy', 'prehistory', 'professional_accounting',
            'professional_law', 'professional_medicine',
            'professional_psychology', 'public_relations', 'security_studies',
            'sociology', 'us_foreign_policy', 'virology', 'world_religions',
        ],
        "type": "mcq",       # multiple-choice question
    },
    "mmmlu": {
        "hf_name": "openai/MMMLU",
        "split": "test",
        "subjects": [
            'AR_XY', 'BN_BD', 'DE_DE', 'ES_LA', 'FR_FR', 'HI_IN',
            'ID_ID', 'IT_IT', 'JA_JP', 'KO_KR', 'PT_BR', 'SW_KE',
            'YO_NG', 'ZH_CN',
        ],
        "type": "mcq",
    },
    "gpqa": {
        "hf_name": "Idavidrein/gpqa",
        "split": "train",
        "subjects": ["gpqa_diamond"],
        "type": "mcq",
    },
    "math-500": {
        "hf_name": "HuggingFaceH4/MATH-500",
        "split": "test",
        "subjects": ["all"],
        "type": "open",       # open-ended generation
    },
    "gsm8k": {
        "hf_name": "openai/gsm8k",
        "split": "test",
        "subjects": ["main"],
        "type": "open",
    },
    "openbookqa": {
        "hf_name": "openbookqa",
        "split": "test",
        "subjects": ["main"],
        "type": "mcq",
    },
    "ai2-arc": {
        "hf_name": "allenai/ai2_arc",
        "split": "test",
        "subjects": ["ARC-Challenge"],
        "type": "mcq",
    },
    "mmlu-pro": {
        "hf_name": "TIGER-Lab/MMLU-Pro",
        "split": "test",
        "subjects": ["main"],
        "type": "mcq",
    },
    "ceval": {
        "hf_name": "ceval/ceval-exam",
        "split": "test",
        "subjects": [
            "accountant", "advanced_mathematics", "art_studies",
            "basic_medicine", "business_administration",
            "chinese_language_and_literature", "civil_servant",
            "clinical_medicine", "college_chemistry", "college_economics",
            "college_physics", "college_programming",
            "computer_architecture", "computer_network",
            "discrete_mathematics", "education_science",
            "electrical_engineer",
            "environmental_impact_assessment_engineer", "fire_engineer",
            "high_school_biology", "high_school_chemistry",
            "high_school_chinese", "high_school_geography",
            "high_school_history", "high_school_mathematics",
            "high_school_physics", "high_school_politics",
            "ideological_and_moral_cultivation", "law",
            "legal_professional", "logic", "mao_zedong_thought",
            "marxism", "metrology_engineer", "middle_school_biology",
            "middle_school_chemistry", "middle_school_geography",
            "middle_school_history", "middle_school_mathematics",
            "middle_school_physics", "middle_school_politics",
            "modern_chinese_history", "operating_system", "physician",
            "plant_protection", "probability_and_statistics",
            "professional_tour_guide", "sports_science",
            "tax_accountant", "teacher_qualification",
            "urban_and_rural_planner", "veterinary_medicine",
        ],
        "type": "mcq",
    },
}


# ===================================================================== #
#  Prompt builders                                                       #
# ===================================================================== #

def build_mcq_prompt(question: str, choices: str, use_cot: bool) -> str:
    """MCQ prompt for MMLU-family / GPQA / ARC / OpenBookQA / ceval."""
    if use_cot:
        tpl = (
            "Accurately answer the following question:\n\n"
            "{{question}}\n\n"
            "Choices:\n{{choices}}\n"
            "Instructions:\n"
            "- Carefully read the question and all options.\n"
            "- Let's think step by step and explain your reasoning briefly.\n"
            "- Then give the final answer starting with The correct answer is"
        )
    else:
        tpl = (
            "Accurately answer the following question:\n\n"
            "{{question}}\n\n"
            "Choices:\n{{choices}}\n"
            "Instructions:\n"
            "- Carefully read the question and all options.\n"
            "- Select the single most correct answer.\n"
            '- Respond ONLY in the following format: "The correct answer is A/B/C/D".\n'
            "- Do not include any explanations, additional text, or punctuation besides the answer.\n\n"
            "The correct answer is"
        )
    return tpl.replace("{{question}}", question).replace("{{choices}}", choices)


def build_math_prompt(question: str, use_cot: bool) -> str:
    """Open-ended math prompt for MATH-500 / GSM8K."""
    if use_cot:
        return (
            "Solve the following math problem step by step. The last line of "
            "your response should be of the form Answer: $ANSWER (without quotes) "
            "where $ANSWER is the answer to the problem.\n\n"
            f"{question}\n\n"
            "Please think step by step and explain your reasoning. Remember to "
            'put your answer on its own line after "Answer:", and you do not '
            "need to use a \\boxed command."
        )
    else:
        return (
            "Solve the following math problem.\n\n"
            f"{question}\n\n"
            "Answer:"
        )


# ===================================================================== #
#  Example formatting per dataset                                        #
# ===================================================================== #

def _prepare_gpqa_item(example: Dict) -> Dict:
    """Deterministic shuffle for GPQA options."""
    def pick(pk, rk):
        rv = example.get(rk)
        return str(rv) if rv is not None and str(rv).strip() else str(example.get(pk, ""))

    q = pick("Question", "Extra Revised Question")
    correct = pick("Correct Answer", "Extra Revised Correct Answer")
    inc = [pick(f"Incorrect Answer {i}", f"Extra Revised Incorrect Answer {i}") for i in range(1, 4)]
    all_c = [correct] + inc
    seed = int(hashlib.md5("||".join([q] + all_c).encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    idx = list(range(4)); rng.shuffle(idx)
    shuffled = [all_c[i] for i in idx]
    return {"question": q, "choices": shuffled, "answer": shuffled.index(correct)}


def format_example(dataset_name: str, example: Dict, use_cot: bool,
                   subject: str = "") -> Optional[str]:
    """
    Format a single example into a prompt string.
    Returns None if the example should be skipped.
    """
    if dataset_name == "mmlu-redux":
        err = example.get("error_type", "")
        if err in ("no_correct_answer", "expert"):
            return None
        choices = "".join(f"{chr(65+i)}. {c}\n" for i, c in enumerate(example["choices"]))
        return build_mcq_prompt(example["question"], choices, use_cot)

    elif dataset_name == "mmmlu":
        q = example["Question"]
        choices = "".join(f"{k}. {example[k]}\n" for k in ("A", "B", "C", "D") if k in example)
        return build_mcq_prompt(q, choices, use_cot)

    elif dataset_name == "gpqa":
        prep = _prepare_gpqa_item(example)
        choices = "".join(f"{chr(65+i)}. {c}\n" for i, c in enumerate(prep["choices"]))
        return build_mcq_prompt(prep["question"], choices, use_cot)

    elif dataset_name in ("math-500",):
        return build_math_prompt(example.get("problem", ""), use_cot)

    elif dataset_name == "gsm8k":
        return build_math_prompt(example.get("question", ""), use_cot)

    elif dataset_name == "openbookqa":
        q = example.get("question_stem", "")
        raw = example.get("choices", {})
        texts = list(raw.get("text", [])) if isinstance(raw, dict) else [
            str(x.get("text", x) if isinstance(x, dict) else x) for x in raw
        ]
        choices = "".join(f"{chr(65+i)}. {t}\n" for i, t in enumerate(texts))
        return build_mcq_prompt(q, choices, use_cot)

    elif dataset_name == "ai2-arc":
        q = example.get("question", "")
        raw = example.get("choices", {})
        texts = list(raw.get("text", [])) if isinstance(raw, dict) else [
            str(x.get("text", x) if isinstance(x, dict) else x) for x in raw
        ]
        choices = "".join(f"{chr(65+i)}. {t}\n" for i, t in enumerate(texts))
        return build_mcq_prompt(q, choices, use_cot)

    elif dataset_name == "mmlu-pro":
        q = example.get("question", "")
        opts = example.get("options", [])
        choices = "".join(f"{chr(65+i)}. {o}\n" for i, o in enumerate(opts[:10]))
        return build_mcq_prompt(q, choices, use_cot)

    elif dataset_name == "ceval":
        q = example.get("question", "")
        choices = "".join(f"{k}. {example.get(k, '')}\n"
                          for k in ("A", "B", "C", "D") if example.get(k))
        return build_mcq_prompt(q, choices, use_cot)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_ground_truth(dataset_name: str, example: Dict, subject: str = "") -> Optional[str]:
    """Extract the ground-truth answer string from an example."""
    if dataset_name == "mmlu-redux":
        err = example.get("error_type", "")
        if err == "wrong_groundtruth" and example.get("correct_answer") is not None:
            a = example["correct_answer"]
            return chr(65 + int(a)) if a in "0123" else a
        return chr(65 + int(example["answer"]))

    elif dataset_name == "mmmlu":
        a = example.get("Answer")
        if isinstance(a, int): return chr(65 + a)
        if isinstance(a, str) and a in "0123": return chr(65 + int(a))
        if isinstance(a, str) and a in "ABCD": return a
        return None

    elif dataset_name == "gpqa":
        prep = _prepare_gpqa_item(example)
        return chr(65 + int(prep["answer"]))

    elif dataset_name == "math-500":
        return str(example.get("answer", "")).strip() or None

    elif dataset_name == "gsm8k":
        full = str(example.get("answer", ""))
        if "####" in full:
            tail = full.split("####")[-1].strip()
            m = re.search(r"[-+]?\d+(?:\.\d+)?", tail)
            return m.group(0) if m else tail
        return None

    elif dataset_name == "openbookqa":
        return example.get("answerKey")

    elif dataset_name == "ai2-arc":
        ak = example.get("answerKey", "")
        if ak in "12345":
            return chr(64 + int(ak))
        return ak if ak in "ABCDE" else None

    elif dataset_name == "mmlu-pro":
        return example.get("answer")

    elif dataset_name == "ceval":
        return example.get("answer")

    return None


def extract_prediction(dataset_name: str, text: str) -> Optional[str]:
    """
    Extract predicted answer from generated text.
    MCQ datasets → look for A/B/C/D letter.
    Math datasets → look for numerical answer.
    """
    text = text.strip()
    if not text:
        return None

    ds_type = DATASET_CONFIGS.get(dataset_name, {}).get("type", "mcq")

    if ds_type == "mcq":
        # Try "correct answer is X" patterns first
        for pat in [
            r'(?:correct answer|answer)\s*(?:is|:)\s*([A-J])',
            r'\b([A-J])(?:\s*[.,!?:)]?\s*$)',
            r'(?:^|\s)([A-J])\s*$',
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).upper()
        # Fallback: last A-D letter
        letters = re.findall(r'\b([A-D])\b', text)
        return letters[-1].upper() if letters else None

    else:  # open (math)
        # Look for "Answer: <value>"
        m = re.search(r'[Aa]nswer\s*:\s*(.+)', text)
        if m:
            val = m.group(1).strip().rstrip(".")
            num = re.search(r"[-+]?\d+(?:,\d+)*(?:\.\d+)?", val)
            return num.group(0).replace(",", "") if num else val
        # Fallback: last number in text
        nums = re.findall(r"[-+]?\d+(?:,\d+)*(?:\.\d+)?", text)
        return nums[-1].replace(",", "") if nums else None


def judge_correct(dataset_name: str, pred: Optional[str],
                  gt: Optional[str]) -> Optional[bool]:
    """Compare prediction with ground truth. Returns None if either is missing."""
    if pred is None or gt is None:
        return None
    ds_type = DATASET_CONFIGS.get(dataset_name, {}).get("type", "mcq")
    if ds_type == "mcq":
        return pred.strip().upper() == gt.strip().upper()
    else:
        # Numeric comparison for math
        try:
            return abs(float(pred) - float(gt)) < 1e-5
        except (ValueError, TypeError):
            return pred.strip() == gt.strip()


# ===================================================================== #
#  Utilities                                                             #
# ===================================================================== #

def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)

def clone_kv_cache(kv: DynamicCache) -> DynamicCache:
    c = DynamicCache()
    for k, v in zip(kv.key_cache, kv.value_cache):
        c.key_cache.append(k.clone().detach())
        c.value_cache.append(v.clone().detach())
    return c

def hybrid_to_dynamic(cache):
    if cache is None or isinstance(cache, DynamicCache):
        return cache
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return DynamicCache.from_legacy_cache(list(zip(cache.key_cache, cache.value_cache)))
    raise TypeError(f"Unsupported cache type: {type(cache)}")

def gpu_gb():
    return torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0


# ===================================================================== #
#  Analyzer                                                              #
# ===================================================================== #

class LogitLensAnalyzer:
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.model_config = config["model"]
        self.eval_config  = config["eval"]

        rcfg = self.model_config["rosetta_config"]
        self.receiver_path  = rcfg["base_model"]
        self.sharer_path    = rcfg["teacher_model"]
        self.checkpoint_dir = rcfg["checkpoints_dir"]
        self.is_do_alignment    = rcfg.get("is_do_alignment", False)
        self.alignment_strategy = rcfg.get("alignment_strategy", "longest")

        # Dataset / CoT from eval config
        self.dataset_name = self.eval_config.get("dataset", "mmlu-redux")
        self.use_cot      = self.eval_config.get("use_cot", False)

        self.receiver_model = self.sharer_model = None
        self.receiver_tokenizer = self.sharer_tokenizer = None
        self.projector_list = []
        self.projector_dict = {}
        self.aligner = None

    # ---- model loading ----
    def load_models(self):
        print(f"[Mem] start: {gpu_gb():.2f} GB")
        from rosetta.utils.evaluate import set_default_chat_template

        self.receiver_tokenizer = AutoTokenizer.from_pretrained(self.receiver_path)
        if self.receiver_tokenizer.pad_token is None:
            self.receiver_tokenizer.pad_token = self.receiver_tokenizer.eos_token
        set_default_chat_template(self.receiver_tokenizer, self.receiver_path)

        self.sharer_tokenizer = AutoTokenizer.from_pretrained(self.sharer_path)
        if self.sharer_tokenizer.pad_token is None:
            self.sharer_tokenizer.pad_token = self.sharer_tokenizer.eos_token
        set_default_chat_template(self.sharer_tokenizer, self.sharer_path)

        if self.is_do_alignment:
            from rosetta.model.aligner import TokenAligner, AlignmentStrategy
            self.aligner = TokenAligner(
                slm_tokenizer=self.receiver_tokenizer,
                llm_tokenizer=self.sharer_tokenizer,
                strategy=AlignmentStrategy(self.alignment_strategy),
            )

        print(f"Loading receiver: {self.receiver_path}")
        self.receiver_model = AutoModelForCausalLM.from_pretrained(
            self.receiver_path, torch_dtype=torch.bfloat16,
            device_map={"": self.device}, low_cpu_mem_usage=True).eval()
        print(f"[Mem] +receiver: {gpu_gb():.2f} GB")

        print(f"Loading sharer: {self.sharer_path}")
        self.sharer_model = AutoModelForCausalLM.from_pretrained(
            self.sharer_path, torch_dtype=torch.bfloat16,
            device_map={"": self.device}, low_cpu_mem_usage=True).eval()
        print(f"[Mem] +sharer: {gpu_gb():.2f} GB")

        self._load_projectors()
        print(f"[Mem] +proj: {gpu_gb():.2f} GB")

    def _load_projectors(self):
        from rosetta.model.projector import load_projector
        d = self.checkpoint_dir
        n = len([f for f in os.listdir(d) if re.match(r"projector_\d+\.pt", f)])
        self.projector_list = []
        for t in range(n):
            p = load_projector(os.path.join(d, f"projector_{t}.json"))
            p = p.to(device=self.device, dtype=torch.bfloat16)
            pt = os.path.join(d, f"projector_{t}.pt")
            if os.path.exists(pt):
                p.load_state_dict(torch.load(pt, map_location=self.device), strict=False)
            p.eval(); self.projector_list.append(p)
        cfg = os.path.join(d, "projector_config.json")
        if os.path.exists(cfg):
            with open(cfg) as f: raw = json.load(f)
            self.projector_dict = self._ints(raw)
        print(f"Loaded {n} projectors")

    @staticmethod
    def _ints(obj):
        if isinstance(obj, dict):
            return {(int(k) if isinstance(k, str) and k.lstrip('-').isdigit() else k):
                    LogitLensAnalyzer._ints(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [LogitLensAnalyzer._ints(v) for v in obj]
        return obj

    def _offload_sharer(self):
        if self.sharer_model is not None:
            self.sharer_model.to("cpu"); torch.cuda.empty_cache(); gc.collect()

    def _reload_sharer(self):
        if self.sharer_model is not None:
            self.sharer_model.to(self.device)

    # ---- tokenization (alignment-aware) ----
    def tokenize_prompt(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]

        # Determine response_text based on dataset type and CoT
        ds_type = DATASET_CONFIGS.get(self.dataset_name, {}).get("type", "mcq")
        if self.use_cot:
            # CoT: use generate mode, no suffix appended
            response_text = None
        elif ds_type == "mcq":
            response_text = "The correct answer is"
        else:
            response_text = "Answer:"

        if self.aligner is not None:
            if response_text is not None:
                msgs = messages + [{"role": "assistant", "content": response_text}]
                add_gen = False; remove_last = True
            else:
                msgs = messages
                add_gen = True; remove_last = False

            details = self.aligner.align_chat_messages(
                msgs, add_generation_prompt=add_gen, return_details=True,
                enable_thinking=False, remove_last_surfix=remove_last)

            r_ids = torch.tensor(details['slm_ids_padded']).unsqueeze(0).to(self.device)
            s_ids = torch.tensor(details['llm_ids_padded']).unsqueeze(0).to(self.device)
            r_mask = (~torch.tensor(details['slm_padding_mask']).unsqueeze(0)).float().to(self.device)
            s_mask = (~torch.tensor(details['llm_padding_mask']).unsqueeze(0)).float().to(self.device)
            assert r_ids.shape == s_ids.shape
        else:
            text = self.receiver_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if response_text is not None:
                text += response_text
            tok = self.receiver_tokenizer(text, return_tensors="pt").to(self.device)
            r_ids = tok["input_ids"]; r_mask = tok["attention_mask"].float()
            s_ids = r_ids.clone(); s_mask = r_mask.clone()

        return {"receiver_ids": r_ids, "receiver_mask": r_mask,
                "sharer_ids": s_ids, "sharer_mask": s_mask}

    # ---- logit lens ----
    @torch.no_grad()
    def logit_lens_decode(self, hidden_states, top_k=5):
        norm = self.receiver_model.model.norm
        lm_head = self.receiver_model.lm_head
        results = []
        for h in hidden_states:
            logits = lm_head(norm(h[:, -1:, :]))[0, 0].float()
            vals, ids = torch.topk(logits, top_k)
            results.append([
                (self.receiver_tokenizer.decode([i]), round(v, 3))
                for v, i in zip(vals.tolist(), ids.tolist())])
        return results

    # ---- single task ----
    @torch.no_grad()
    def run_single_task(self, prompt: str, task_id: int):
        """
        Returns:
            logit_lens_results: list of top-5 per layer
            generated_text: str — model's greedy-decoded answer
        """
        tok = self.tokenize_prompt(prompt)
        r_ids, r_mask = tok["receiver_ids"], tok["receiver_mask"]
        s_ids, s_mask = tok["sharer_ids"],   tok["sharer_mask"]

        # 1) Sharer prefill
        self._reload_sharer()
        s_out = self.sharer_model(input_ids=s_ids, attention_mask=s_mask,
                                  use_cache=True, output_hidden_states=False)
        s_kv = hybrid_to_dynamic(s_out.past_key_values)
        s_kv_cpu = DynamicCache()
        for k, v in zip(s_kv.key_cache, s_kv.value_cache):
            s_kv_cpu.key_cache.append(k.cpu()); s_kv_cpu.value_cache.append(v.cpu())
        del s_out, s_kv; torch.cuda.empty_cache()

        # 2) Offload sharer
        self._offload_sharer()

        # 3) Receiver forward
        pos = r_mask.long().cumsum(-1) - 1
        r_out = self.receiver_model(input_ids=r_ids, attention_mask=r_mask,
                                    position_ids=pos, use_cache=True,
                                    output_hidden_states=True)
        baseline = [h.detach() for h in r_out.hidden_states]
        r_kv = hybrid_to_dynamic(r_out.past_key_values)

        # 4) Projector
        s_kv_gpu = DynamicCache()
        for k, v in zip(s_kv_cpu.key_cache, s_kv_cpu.value_cache):
            s_kv_gpu.key_cache.append(k.to(self.device))
            s_kv_gpu.value_cache.append(v.to(self.device))
        del s_kv_cpu

        fused = clone_kv_cache(r_kv)
        if 0 in self.projector_dict and 1 in self.projector_dict.get(0, {}):
            for tl, entry in self.projector_dict[0][1].items():
                tgt_kv = (fused.key_cache[tl], fused.value_cache[tl])
                for sl, pi in entry:
                    src_kv = (s_kv_gpu.key_cache[sl], s_kv_gpu.value_cache[sl])
                    pk, pv = self.projector_list[pi](src_kv, tgt_kv)
                    fused.key_cache[tl] = pk; fused.value_cache[tl] = pv
                    tgt_kv = (pk, pv)
        del s_kv_gpu; torch.cuda.empty_cache()

        # 5) Re-run with fused KV for logit lens
        from rosetta.model.wrapper import RosettaModel
        mdtype = next(self.receiver_model.parameters()).dtype
        hooks = []
        for i in range(self.receiver_model.config.num_hidden_layers):
            attn = self.receiver_model.model.layers[i].self_attn
            try:
                orig = RosettaModel._monkeypatch_qwen3_attention_forward(
                    attn, fused.key_cache[i].to(dtype=mdtype),
                    fused.value_cache[i].to(dtype=mdtype))
                hooks.append((attn, orig))
            except Exception:
                pass

        if hooks:
            f_out = self.receiver_model(input_ids=r_ids, attention_mask=r_mask,
                                        position_ids=pos, use_cache=True,
                                        output_hidden_states=True)
            hidden = [h.detach() for h in f_out.hidden_states]
            fused_past = hybrid_to_dynamic(f_out.past_key_values)
            last_logits = f_out.logits[:, -1, :]
            del f_out
            for attn, orig in hooks: attn.forward = orig
        else:
            hidden = baseline
            fused_past = r_kv
            last_logits = r_out.logits[:, -1, :]

        # 6) Logit lens
        logit_lens_results = self.logit_lens_decode(hidden)

        # 7) Greedy auto-regressive generation from the fused state
        max_new = self.model_config.get("generation_config", {}).get("max_new_tokens", 64)
        eos_id = self.receiver_model.config.eos_token_id
        if isinstance(eos_id, list):
            eos_set = set(eos_id)
        elif eos_id is not None:
            eos_set = {eos_id}
        else:
            eos_set = set()

        gen_ids = []
        cur_logits = last_logits
        cur_past = fused_past
        cur_mask = r_mask.clone()

        for _ in range(max_new):
            next_id = cur_logits.argmax(dim=-1)          # (B,)
            token_id = next_id.item()
            if token_id in eos_set:
                break
            gen_ids.append(token_id)

            next_input = next_id.unsqueeze(1)             # (1, 1)
            cur_mask = torch.cat([cur_mask,
                torch.ones((1, 1), device=self.device, dtype=cur_mask.dtype)], dim=1)

            step_out = self.receiver_model(
                input_ids=next_input, attention_mask=cur_mask,
                past_key_values=cur_past, use_cache=True)
            cur_logits = step_out.logits[:, -1, :]
            cur_past = step_out.past_key_values

        generated_text = self.receiver_tokenizer.decode(gen_ids, skip_special_tokens=True)

        del baseline, hidden, r_out, r_kv, fused, fused_past, cur_past
        torch.cuda.empty_cache(); gc.collect()
        return logit_lens_results, generated_text.strip()

    # ---- main loop ----
    def run_analysis(self, output_dir: str, num_tasks: Optional[int] = None,
                     subjects: Optional[List[str]] = None):
        os.makedirs(output_dir, exist_ok=True)
        csv_path  = os.path.join(output_dir, "receiver_sharer_dataset.csv")
        jsonl_path = os.path.join(output_dir, "receiver_sharer_answers.jsonl")

        header = ["task#", "layer"]
        for r in range(1, 6):
            header += [f"top{r}", f"top{r}_logit"]

        ds_name = self.dataset_name
        ds_cfg  = DATASET_CONFIGS[ds_name]

        cfg_subjects = self.eval_config.get("subjects", None)
        all_subjects = subjects or cfg_subjects or ds_cfg["subjects"]

        task_counter = 0
        rows = []
        answer_records = []   # for JSONL

        for subject in all_subjects:
            print(f"\n=== {ds_name} / {subject} ===")
            try:
                if ds_name in ("math-500",):
                    dataset = load_dataset(ds_cfg["hf_name"])
                elif ds_name == "gsm8k":
                    dataset = load_dataset(ds_cfg["hf_name"], "main")
                elif ds_name == "openbookqa":
                    dataset = load_dataset(ds_cfg["hf_name"])
                elif ds_name == "ai2-arc":
                    dataset = load_dataset(ds_cfg["hf_name"], "ARC-Challenge")
                elif ds_name == "mmlu-pro":
                    dataset = load_dataset(ds_cfg["hf_name"])
                else:
                    dataset = load_dataset(ds_cfg["hf_name"], subject)
                test_data = dataset[ds_cfg["split"]]
            except Exception as e:
                print(f"  Failed to load: {e}"); continue

            interval = self.eval_config.get("sample_interval", 1)
            start    = self.eval_config.get("start_index", 0)
            indices  = list(range(start, len(test_data), interval))

            if num_tasks is not None:
                rem = num_tasks - task_counter
                if rem <= 0: break
                indices = indices[:rem]

            for idx in tqdm(indices, desc=f"  {subject}"):
                try:
                    example = test_data[idx]
                    prompt = format_example(ds_name, example, self.use_cot, subject)
                    if prompt is None:
                        continue

                    # Ground truth
                    gt = get_ground_truth(ds_name, example, subject)

                    # Run logit lens + generation
                    logit_lens, gen_text = self.run_single_task(prompt, task_counter)

                    # Extract prediction and judge
                    pred = extract_prediction(ds_name, gen_text)
                    correct = judge_correct(ds_name, pred, gt)

                    # ---- CSV rows (logit lens) ----
                    for li, top5 in enumerate(logit_lens):
                        row = {"task#": task_counter, "layer": li}
                        for rk, (tok, logit) in enumerate(top5, 1):
                            row[f"top{rk}"] = repr(tok)[1:-1]
                            row[f"top{rk}_logit"] = f"{logit:.3f}"
                        rows.append(row)

                    # ---- JSONL record (answer) ----
                    answer_records.append({
                        "task_id": task_counter,
                        "dataset": ds_name,
                        "subject": subject,
                        "example_index": idx,
                        "prompt": prompt,
                        "generated_text": gen_text,
                        "prediction": pred,
                        "ground_truth": gt,
                        "correct": correct,
                    })

                    task_counter += 1
                    if task_counter % 10 == 0:
                        print(f"  [Task {task_counter}] GPU: {gpu_gb():.2f} GB")
                except Exception as e:
                    print(f"  Error task {task_counter}, idx {idx}: {e}")
                    import traceback; traceback.print_exc()
                    continue

        # ---- Write CSV (logit lens) ----
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=header); w.writeheader()
            for row in rows: w.writerow(row)
        print(f"\n✓ Logit lens: {len(rows)} rows ({task_counter} tasks) → {csv_path}")

        # ---- Write JSONL (answers) ----
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for rec in answer_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_correct = sum(1 for r in answer_records if r["correct"] is True)
        n_judged  = sum(1 for r in answer_records if r["correct"] is not None)
        acc = n_correct / n_judged * 100 if n_judged > 0 else 0.0
        print(f"✓ Answers: {len(answer_records)} tasks → {jsonl_path}")
        print(f"  Accuracy: {n_correct}/{n_judged} = {acc:.2f}%")

        return csv_path, jsonl_path


# ===================================================================== #
#  Main                                                                  #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description='Logit Lens Analysis')
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--subjects", type=str, nargs="*", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(
            config["model"]["rosetta_config"]["checkpoints_dir"])), "resource")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    ds_name = config["eval"].get("dataset", "mmlu-redux")
    use_cot = config["eval"].get("use_cot", False)
    print(f"Dataset: {ds_name}  |  CoT: {use_cot}  |  Output: {output_dir}")

    # num_tasks: CLI > yaml > None (전체)
    num_tasks = args.num_tasks if args.num_tasks is not None \
        else config["eval"].get("num_of_tasks", None)

    analyzer = LogitLensAnalyzer(config, device)
    analyzer.load_models()
    analyzer.run_analysis(output_dir, num_tasks=num_tasks, subjects=args.subjects)


if __name__ == "__main__":
    main()
