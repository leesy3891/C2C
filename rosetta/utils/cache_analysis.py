"""
Cache Analysis Module for C2C / Rosetta / TwoStage evaluation.

Provides:
  - CacheAnalysisCollector: records layerwise key-cache tensors during eval
  - Shannon entropy via logit-lens projection (handles GQA head expansion)
  - KL divergence between consecutive layers (raw key-cache distribution)
  - PCA scatter-plots per layer (4x4 grid per PNG)
  - Layerwise line-plots for entropy and KL

Output structure (under output_dir):
  C2C/resource/cache_analysis/
      {dataset}_{N}cases_cache_metrics.csv
      {dataset}_{N}cases_cache_analysis_summary.json
      {dataset}_{N}cases_entropy_layerwise.png
      {dataset}_{N}cases_kl_prev_layerwise.png
      pca/
          {dataset}_{N}cases_pca_layer{start}-{end}.png   (4x4 grid)
      raw_cache/   (only if save_raw_cache=true)
"""

from __future__ import annotations

import csv
import json
import math
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
C2C_COLORS = {"receiver": "tab:blue", "sharer": "tab:orange", "fused": "tab:green"}
TWO_STAGE_COLORS = {"stage1": "tab:orange", "stage2": "tab:blue"}


class CacheAnalysisCollector:
    """Collects key-cache snapshots during evaluation and computes analysis."""

    def __init__(self, config: Dict[str, Any], output_dir: str | Path, rank: int = 0):
        self.enabled: bool = config.get("enabled", False)
        if not self.enabled:
            return

        self.target: str = config.get("target", "key")
        self.save_raw_cache: bool = config.get("save_raw_cache", False)
        self.max_pca_points: int = config.get("max_pca_points_per_layer", 2000)
        self.pca_seed: int = config.get("pca_sample_seed", 42)

        self.entropy_space: str = config.get("entropy_space", "logit_lens")
        self.logit_lens_temperature: float = config.get("logit_lens_temperature", 1.0)
        self.entropy_eps: float = config.get("entropy_eps", 1e-12)

        self.kl_space: str = config.get("kl_space", "raw_key_cache")
        self.kl_vector_align: str = config.get("kl_vector_align", "last")
        self.kl_eps: float = config.get("kl_eps", 1e-12)

        self.rank = rank
        self.output_dir = Path(output_dir)
        # All outputs under C2C/resource/cache_analysis
        self.ca_dir = self.output_dir / "C2C" / "resource" / "cache_analysis"
        self.ca_dir.mkdir(parents=True, exist_ok=True)

        self._records: List[Dict[str, Any]] = []
        self._raw_caches: Dict[tuple, torch.Tensor] = {}
        self._file_prefix: str = ""  # set in finalize

        # lm_head references + GQA topology
        self._lm_heads: Dict[str, torch.nn.Module] = {}
        self._hidden_sizes: Dict[str, int] = {}
        self._vocab_sizes: Dict[str, int] = {}
        self._num_attention_heads: Dict[str, int] = {}
        self._num_kv_heads: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # External setters
    # ------------------------------------------------------------------
    def set_lm_head(self, name: str, lm_head: torch.nn.Module,
                    hidden_size: int, vocab_size: int,
                    num_attention_heads: int = 0,
                    num_key_value_heads: int = 0):
        """Register an lm_head for logit-lens entropy.

        For GQA models (num_key_value_heads < num_attention_heads), the key cache
        has H = num_key_value_heads heads. We expand by repeating to match
        num_attention_heads * head_dim = hidden_size before applying lm_head.
        """
        if not self.enabled:
            return
        self._lm_heads[name] = lm_head
        self._hidden_sizes[name] = hidden_size
        self._vocab_sizes[name] = vocab_size
        self._num_attention_heads[name] = num_attention_heads
        self._num_kv_heads[name] = num_key_value_heads

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record(self, model_type: str, subject: str, question_id: int,
               stage: str, cache_name: str, layer_idx: int,
               key_cache: torch.Tensor, metadata: Optional[Dict[str, Any]] = None):
        if not self.enabled:
            return
        kc = key_cache.detach().float().cpu()
        self._records.append({
            "model_type": model_type, "subject": subject,
            "question_id": question_id, "stage": stage,
            "cache_name": cache_name, "layer_idx": layer_idx,
            "key_cache": kc, "metadata": metadata or {},
        })
        if self.save_raw_cache:
            self._raw_caches[(subject, question_id, model_type, stage, cache_name, layer_idx)] = kc

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    def finalize(self, dataset: str, model_name: str, answer_method: str,
                 start_index: int = 0, num_of_tasks: Optional[int] = None):
        if not self.enabled or not self._records:
            return

        n_cases = len({(r["subject"], r["question_id"]) for r in self._records})
        self._file_prefix = f"{dataset}_{n_cases}cases"
        rank_suffix = f"_rank{self.rank}" if self.rank > 0 else ""

        metric_rows = self._compute_metrics(dataset)

        csv_path = self.ca_dir / f"{self._file_prefix}_cache_metrics{rank_suffix}.csv"
        self._write_metrics_csv(metric_rows, csv_path)

        if self.save_raw_cache:
            self._save_raw_caches()

        self._generate_pca_plots()
        self._generate_line_plots(metric_rows)

        summary = {
            "dataset": dataset, "model_name": model_name,
            "answer_method": answer_method, "start_index": start_index,
            "num_of_tasks": num_of_tasks, "n_cases": n_cases,
            "cache_names": sorted({r["cache_name"] for r in self._records}),
            "num_records": len(self._records), "rank": self.rank,
            "pca_config": {"max_pca_points_per_layer": self.max_pca_points,
                           "pca_sample_seed": self.pca_seed},
            "lm_head_info": {
                name: {"hidden_size": self._hidden_sizes[name],
                       "vocab_size": self._vocab_sizes[name],
                       "num_attention_heads": self._num_attention_heads.get(name, 0),
                       "num_kv_heads": self._num_kv_heads.get(name, 0)}
                for name in self._lm_heads
            },
            "entropy_definition": (
                "Logit-lens entropy: key cache [B,H_kv,T,D] expanded via GQA head "
                "repetition to [B,T,hidden_size], projected through lm_head, "
                "softmax(logits/T), Shannon entropy. Normalized by log(vocab_size)."),
            "kl_definition": (
                "KL(l||l-1): magnitude-normalized distribution of "
                "abs(key_cache).flatten(), aligned by kl_vector_align."),
        }
        sp = self.ca_dir / f"{self._file_prefix}_cache_analysis_summary{rank_suffix}.json"
        with open(sp, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[CacheAnalysis] Summary saved to {sp}")

    # ------------------------------------------------------------------
    # GQA-aware expansion  key_cache -> hidden_size vector
    # ------------------------------------------------------------------
    def _expand_kv_to_hidden(self, key_cache: torch.Tensor,
                              lm_head_name: str) -> Optional[torch.Tensor]:
        """Expand GQA key cache [B, H_kv, T, D_head] -> [B*T, hidden_size].

        For GQA: num_key_value_heads < num_attention_heads.
        Each KV head is repeated (num_attn_heads // num_kv_heads) times.
        """
        hidden_size = self._hidden_sizes[lm_head_name]
        n_attn = self._num_attention_heads.get(lm_head_name, 0)
        n_kv = self._num_kv_heads.get(lm_head_name, 0)
        B, H_kv, T, D_head = key_cache.shape

        # Case 1: MHA — H_kv * D_head already equals hidden_size
        if H_kv * D_head == hidden_size:
            return key_cache.transpose(1, 2).reshape(B * T, hidden_size)

        # Case 2: GQA — expand KV heads
        if n_attn > 0 and n_kv > 0 and H_kv == n_kv and n_attn % n_kv == 0:
            rep = n_attn // n_kv
            # [B, H_kv, T, D] -> [B, H_kv, rep, T, D] -> [B, n_attn, T, D]
            expanded = key_cache.unsqueeze(2).expand(B, H_kv, rep, T, D_head)
            expanded = expanded.reshape(B, n_attn, T, D_head)
            full_dim = n_attn * D_head
            if full_dim != hidden_size:
                warnings.warn(
                    f"[CacheAnalysis] After GQA expansion {full_dim} != hidden_size "
                    f"{hidden_size}. Skipping logit-lens for '{lm_head_name}'.")
                return None
            return expanded.transpose(1, 2).reshape(B * T, hidden_size)

        # Case 3: mismatch
        warnings.warn(
            f"[CacheAnalysis] Cannot map H_kv={H_kv} D={D_head} to hidden={hidden_size} "
            f"(n_attn={n_attn}, n_kv={n_kv}). Skipping logit-lens for '{lm_head_name}'.")
        return None

    # ------------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------------
    def _compute_logit_lens_entropy(self, key_cache: torch.Tensor,
                                    lm_head_name: str) -> Optional[Tuple[float, float, int]]:
        if lm_head_name not in self._lm_heads:
            return None
        lm_head = self._lm_heads[lm_head_name]
        vocab_size = self._vocab_sizes[lm_head_name]

        z_flat = self._expand_kv_to_hidden(key_cache, lm_head_name)
        if z_flat is None:
            return None

        device = next(lm_head.parameters()).device
        dtype = next(lm_head.parameters()).dtype
        z_flat = z_flat.to(device=device, dtype=dtype)

        with torch.no_grad():
            logits = lm_head(z_flat)
            p = F.softmax(logits / self.logit_lens_temperature, dim=-1)
            log_p = torch.log(p + self.entropy_eps)
            ent = -(p * log_p).sum(dim=-1).mean().item()
            ent_norm = ent / math.log(vocab_size) if vocab_size > 1 else 0.0
        return ent, ent_norm, vocab_size

    # ------------------------------------------------------------------
    # KL divergence
    # ------------------------------------------------------------------
    @staticmethod
    def _make_magnitude_dist(kc: torch.Tensor, eps: float) -> torch.Tensor:
        x = kc.detach().float().abs().flatten()
        return x / (x.sum() + eps)

    def _compute_kl(self, p: torch.Tensor, q: torch.Tensor) -> float:
        min_len = min(len(p), len(q))
        if self.kl_vector_align == "last":
            p, q = p[-min_len:], q[-min_len:]
        else:
            p, q = p[:min_len], q[:min_len]
        p = p / (p.sum() + self.kl_eps)
        q = q / (q.sum() + self.kl_eps)
        return (p * (torch.log(p + self.kl_eps) - torch.log(q + self.kl_eps))).sum().item()

    # ------------------------------------------------------------------
    # Compute all metrics
    # ------------------------------------------------------------------
    def _compute_metrics(self, dataset: str) -> List[Dict[str, Any]]:
        grouped: Dict[tuple, List[Dict]] = defaultdict(list)
        for r in self._records:
            key = (r["model_type"], r["subject"], r["question_id"], r["stage"], r["cache_name"])
            grouped[key].append(r)

        rows = []
        for (mt, subj, qid, stage, cn), recs in grouped.items():
            recs.sort(key=lambda x: x["layer_idx"])
            lm_name = self._resolve_lm_head_name(mt, cn)
            prev_dist = None
            for rec in recs:
                kc = rec["key_cache"]
                ent_r = self._compute_logit_lens_entropy(kc, lm_name)
                e_raw, e_norm, vs = ent_r if ent_r else (float("nan"), float("nan"), 0)
                cd = self._make_magnitude_dist(kc, self.kl_eps)
                kl = self._compute_kl(cd, prev_dist) if prev_dist is not None else float("nan")
                prev_dist = cd
                rows.append({
                    "dataset": dataset, "subject": subj, "question_id": qid,
                    "model_type": mt, "stage": stage, "cache_name": cn,
                    "layer_idx": rec["layer_idx"],
                    "entropy_raw": e_raw, "entropy_normalized": e_norm,
                    "logit_lens_temperature": self.logit_lens_temperature,
                    "vocab_size": vs, "kl_prev_layer": kl,
                    "kl_vector_align": self.kl_vector_align,
                })
        return rows

    def _resolve_lm_head_name(self, model_type: str, cache_name: str) -> str:
        if model_type in ("rosetta", "two_stage_rosetta"):
            if cache_name in ("receiver", "fused", "sharer"):
                return "receiver"
        if model_type in ("two_stage", "two_stage_rosetta"):
            if cache_name == "stage1":
                return "stage1"
            if cache_name == "stage2":
                return "stage2"
        return cache_name

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    @staticmethod
    def _write_metrics_csv(rows: List[Dict[str, Any]], path: Path):
        if not rows:
            return
        fieldnames = [
            "dataset", "subject", "question_id", "model_type", "stage", "cache_name",
            "layer_idx", "entropy_raw", "entropy_normalized", "logit_lens_temperature",
            "vocab_size", "kl_prev_layer", "kl_vector_align",
        ]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"[CacheAnalysis] Metrics CSV -> {path}")

    # ------------------------------------------------------------------
    # Raw cache saving
    # ------------------------------------------------------------------
    def _save_raw_caches(self):
        raw_dir = self.ca_dir / "raw_cache"
        raw_dir.mkdir(parents=True, exist_ok=True)
        rs = f"_rank{self.rank}" if self.rank > 0 else ""
        for (subj, qid, mt, stg, cn, li), kc in self._raw_caches.items():
            torch.save(kc, raw_dir / f"{self._file_prefix}_{subj}_{qid}_{mt}_{stg}_{cn}_layer{li}{rs}.pt")
        print(f"[CacheAnalysis] Raw caches saved ({len(self._raw_caches)} files)")

    # ------------------------------------------------------------------
    # PCA — 4x4 grid
    # ------------------------------------------------------------------
    def _generate_pca_plots(self):
        try:
            from sklearn.decomposition import PCA
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("[CacheAnalysis] sklearn/matplotlib not found – skipping PCA.")
            return

        pca_dir = self.ca_dir / "pca"
        pca_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate per layer
        layer_data: Dict[int, Dict[str, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
        mt_global = None
        for r in self._records:
            layer_data[r["layer_idx"]][r["cache_name"]].append(r["key_cache"])
            if mt_global is None:
                mt_global = r["model_type"]

        sorted_layers = sorted(layer_data.keys())
        if not sorted_layers:
            return

        colors = C2C_COLORS if mt_global in ("rosetta", "two_stage_rosetta") else TWO_STAGE_COLORS
        rng = np.random.RandomState(self.pca_seed)

        grid = 16  # 4x4
        for gs in range(0, len(sorted_layers), grid):
            grp = sorted_layers[gs:gs + grid]
            n = len(grp)
            nr = min(4, (n + 3) // 4)
            nc = 4 if nr > 1 else min(4, n)
            fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3.5 * nr))
            axes = np.atleast_2d(axes)
            if axes.ndim == 1:
                axes = axes[np.newaxis, :]

            for pi, li in enumerate(grp):
                r, c = divmod(pi, nc)
                ax = axes[r, c]
                cc = layer_data[li]

                # dim compat check
                dims = {cn: ts[0].shape[1] * ts[0].shape[3] for cn, ts in cc.items()}
                primary = max(set(dims.values()), key=list(dims.values()).count)
                comp = {cn: ts for cn, ts in cc.items() if dims[cn] == primary}

                pts_all, names = [], []
                for cn, ts in comp.items():
                    flat = [t.permute(0, 2, 1, 3).reshape(-1, primary).numpy() for t in ts]
                    pts = np.concatenate(flat)
                    if len(pts) > self.max_pca_points:
                        pts = pts[rng.choice(len(pts), self.max_pca_points, replace=False)]
                    pts_all.append(pts)
                    names.append(cn)

                if not pts_all:
                    ax.set_visible(False)
                    continue
                combined = np.concatenate(pts_all)
                if combined.shape[0] < 2:
                    ax.set_visible(False)
                    continue

                pca = PCA(n_components=2)
                tf = pca.fit_transform(combined)
                off = 0
                for cn, p in zip(names, pts_all):
                    nl = len(p)
                    ax.scatter(tf[off:off + nl, 0], tf[off:off + nl, 1],
                               c=colors.get(cn, "gray"), label=cn, alpha=0.4, s=3)
                    off += nl
                ax.set_title(f"Layer {li}", fontsize=9)
                ax.tick_params(labelsize=6)
                if pi == 0:
                    ax.legend(fontsize=6, loc="upper right")

            for pi in range(n, nr * nc):
                r, c = divmod(pi, nc)
                axes[r, c].set_visible(False)

            fl, ll = grp[0], grp[-1]
            fig.suptitle(f"Key Cache PCA — Layers {fl}-{ll}", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(pca_dir / f"{self._file_prefix}_pca_layer{fl}-{ll}.png", dpi=150)
            plt.close(fig)
        print(f"[CacheAnalysis] PCA grid plots -> {pca_dir}")

    # ------------------------------------------------------------------
    # Line-plots
    # ------------------------------------------------------------------
    def _generate_line_plots(self, rows: List[Dict[str, Any]]):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        mts = {r["model_type"] for r in rows}
        colors = C2C_COLORS if any(m in ("rosetta", "two_stage_rosetta") for m in mts) else TWO_STAGE_COLORS

        self._lineplot(rows, "layer_idx", "entropy_raw", colors,
                       self.ca_dir / f"{self._file_prefix}_entropy_layerwise.png",
                       title=f"Layerwise Entropy — {self._file_prefix}", ylabel="Entropy (raw)")
        self._lineplot(rows, "layer_idx", "kl_prev_layer", colors,
                       self.ca_dir / f"{self._file_prefix}_kl_prev_layerwise.png",
                       title=f"Layerwise KL(l||l-1) — {self._file_prefix}",
                       ylabel="KL divergence", skip_nan=True)
        print(f"[CacheAnalysis] Line-plots -> {self.ca_dir}")

    @staticmethod
    def _lineplot(rows, x_key, y_key, cmap, save_path, title="", ylabel="", skip_nan=False):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        by_c: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
        for r in rows:
            v = r[y_key]
            if skip_nan and v != v:
                continue
            by_c[r["cache_name"]][r[x_key]].append(v)
        if not by_c:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        for cn, lv in sorted(by_c.items()):
            ls = sorted(lv.keys())
            ms = [np.mean(lv[l]) for l in ls]
            ss = [np.std(lv[l]) for l in ls]
            co = cmap.get(cn, "gray")
            ax.plot(ls, ms, label=cn, color=co, linewidth=1.5)
            ax.fill_between(ls, [m - s for m, s in zip(ms, ss)],
                            [m + s for m, s in zip(ms, ss)], alpha=0.15, color=co)
        ax.set_xlabel("Layer index"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(save_path, dpi=150); plt.close(fig)
