"""
Cache Analysis Module for C2C / Rosetta / TwoStage evaluation.

Provides:
  - CacheAnalysisCollector: records layerwise key-cache tensors during eval
  - Shannon entropy via logit-lens projection
  - KL divergence between consecutive layers (raw key-cache distribution)
  - PCA scatter-plots per layer
  - Layerwise line-plots for entropy and KL

Usage:
    collector = CacheAnalysisCollector(config, output_dir, rank=0)
    collector.record(model_type, subject, question_id, stage, cache_name,
                     layer_idx, key_cache_tensor, metadata)
    ...
    collector.finalize(dataset="mmlu-redux", model_name="Rosetta", answer_method="generate")
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
# C2C / Rosetta 3-way
C2C_COLORS = {"receiver": "tab:blue", "sharer": "tab:orange", "fused": "tab:green"}
# TwoStage 2-way
TWO_STAGE_COLORS = {"stage1": "tab:orange", "stage2": "tab:blue"}


# ---------------------------------------------------------------------------
# CacheAnalysisCollector
# ---------------------------------------------------------------------------

class CacheAnalysisCollector:
    """Collects key-cache snapshots during evaluation and computes analysis."""

    def __init__(self, config: Dict[str, Any], output_dir: str | Path, rank: int = 0):
        """
        Args:
            config: cache_analysis sub-dict from the eval YAML.
            output_dir: root output directory for this evaluation run.
            rank: GPU rank for multi-process safety.
        """
        self.enabled: bool = config.get("enabled", False)
        if not self.enabled:
            return

        self.target: str = config.get("target", "key")  # currently only "key"
        self.save_raw_cache: bool = config.get("save_raw_cache", False)
        self.max_pca_points: int = config.get("max_pca_points_per_layer", 2000)
        self.pca_seed: int = config.get("pca_sample_seed", 42)

        # Entropy config
        self.entropy_space: str = config.get("entropy_space", "logit_lens")
        self.logit_lens_temperature: float = config.get("logit_lens_temperature", 1.0)
        self.entropy_eps: float = config.get("entropy_eps", 1e-12)

        # KL config
        self.kl_space: str = config.get("kl_space", "raw_key_cache")
        self.kl_vector_align: str = config.get("kl_vector_align", "last")
        self.kl_eps: float = config.get("kl_eps", 1e-12)

        self.rank = rank
        self.output_dir = Path(output_dir)
        self.ca_dir = self.output_dir / "cache_analysis"
        self.ca_dir.mkdir(parents=True, exist_ok=True)

        # Internal storage: list of record dicts
        self._records: List[Dict[str, Any]] = []
        # Raw cache storage: keyed by (subject, question_id, model_type, stage, cache_name, layer_idx)
        self._raw_caches: Dict[tuple, torch.Tensor] = {}

        # lm_head references — set externally via set_lm_head()
        self._lm_heads: Dict[str, torch.nn.Module] = {}
        self._hidden_sizes: Dict[str, int] = {}
        self._vocab_sizes: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # External setters
    # ------------------------------------------------------------------
    def set_lm_head(self, name: str, lm_head: torch.nn.Module, hidden_size: int, vocab_size: int):
        """Register an lm_head for logit-lens entropy.

        Args:
            name: identifier — e.g. "receiver", "stage1", "stage2"
            lm_head: nn.Linear or equivalent
            hidden_size: model hidden_size (H*D_head must match this)
            vocab_size: model vocab_size
        """
        if not self.enabled:
            return
        self._lm_heads[name] = lm_head
        self._hidden_sizes[name] = hidden_size
        self._vocab_sizes[name] = vocab_size

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record(
        self,
        model_type: str,
        subject: str,
        question_id: int,
        stage: str,
        cache_name: str,
        layer_idx: int,
        key_cache: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Store one layer's key cache snapshot.

        Args:
            model_type: "rosetta", "two_stage", "two_stage_rosetta", etc.
            subject: dataset subject or split identifier
            question_id: sample index
            stage: "prefill", "generate_prefill", etc.
            cache_name: "receiver", "sharer", "fused", "stage1", "stage2"
            layer_idx: transformer layer index
            key_cache: tensor of shape [B, H, T, D] (will be detached/cpu'd)
            metadata: optional extra info
        """
        if not self.enabled:
            return

        kc = key_cache.detach().float().cpu()
        rec = {
            "model_type": model_type,
            "subject": subject,
            "question_id": question_id,
            "stage": stage,
            "cache_name": cache_name,
            "layer_idx": layer_idx,
            "key_cache": kc,
            "metadata": metadata or {},
        }
        self._records.append(rec)

        if self.save_raw_cache:
            raw_key = (subject, question_id, model_type, stage, cache_name, layer_idx)
            self._raw_caches[raw_key] = kc

    # ------------------------------------------------------------------
    # Finalize: compute metrics + generate plots
    # ------------------------------------------------------------------
    def finalize(self, dataset: str, model_name: str, answer_method: str,
                 start_index: int = 0, num_of_tasks: Optional[int] = None):
        """Compute all metrics, write CSVs, generate plots, save summary."""
        if not self.enabled or not self._records:
            return

        # 1. Compute entropy & KL metrics
        metric_rows = self._compute_metrics(dataset)

        # 2. Write metrics CSV (rank-safe filename)
        csv_path = self.ca_dir / f"cache_metrics_rank{self.rank}.csv"
        self._write_metrics_csv(metric_rows, csv_path)

        # 3. Save raw caches if requested
        if self.save_raw_cache:
            self._save_raw_caches()

        # 4. PCA scatter-plots
        self._generate_pca_plots()

        # 5. Entropy & KL line-plots
        self._generate_line_plots(metric_rows)

        # 6. Summary JSON
        summary = {
            "dataset": dataset,
            "model_name": model_name,
            "answer_method": answer_method,
            "start_index": start_index,
            "num_of_tasks": num_of_tasks,
            "cache_names": sorted({r["cache_name"] for r in self._records}),
            "num_records": len(self._records),
            "rank": self.rank,
            "pca_config": {
                "max_pca_points_per_layer": self.max_pca_points,
                "pca_sample_seed": self.pca_seed,
            },
            "entropy_definition": (
                "Logit-lens entropy: key cache reshaped to (B,T,H*D_head), projected "
                "through lm_head, softmax(logits/temperature), then Shannon entropy. "
                "Normalized by log(vocab_size)."
            ),
            "kl_definition": (
                "KL(layer_l || layer_{l-1}): magnitude-normalized distribution of "
                "abs(key_cache).flatten(), aligned by kl_vector_align setting."
            ),
        }
        summary_path = self.ca_dir / f"cache_analysis_summary_rank{self.rank}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[CacheAnalysis] Summary saved to {summary_path}")

    # ------------------------------------------------------------------
    # Entropy (logit lens)
    # ------------------------------------------------------------------
    def _compute_logit_lens_entropy(self, key_cache: torch.Tensor,
                                    lm_head_name: str) -> Optional[Tuple[float, float, int]]:
        """Return (entropy_raw, entropy_normalized, vocab_size) or None."""
        if lm_head_name not in self._lm_heads:
            return None
        lm_head = self._lm_heads[lm_head_name]
        hidden_size = self._hidden_sizes[lm_head_name]
        vocab_size = self._vocab_sizes[lm_head_name]

        # key_cache shape: [B, H, T, D_head]
        B, H, T, D_head = key_cache.shape
        if H * D_head != hidden_size:
            warnings.warn(
                f"[CacheAnalysis] H*D_head={H * D_head} != hidden_size={hidden_size} "
                f"for lm_head '{lm_head_name}'. Skipping logit-lens entropy."
            )
            return None

        # Merge heads: [B, T, H*D_head]
        z = key_cache.transpose(1, 2).reshape(B, T, H * D_head)
        z_flat = z.reshape(-1, hidden_size)  # [B*T, hidden_size]

        device = next(lm_head.parameters()).device
        dtype = next(lm_head.parameters()).dtype
        z_flat = z_flat.to(device=device, dtype=dtype)

        with torch.no_grad():
            logits = lm_head(z_flat)  # [B*T, vocab_size]
            p = F.softmax(logits / self.logit_lens_temperature, dim=-1)
            # Shannon entropy per token
            log_p = torch.log(p + self.entropy_eps)
            ent_per_token = -(p * log_p).sum(dim=-1)  # [B*T]
            entropy_raw = ent_per_token.mean().item()
            entropy_normalized = entropy_raw / math.log(vocab_size) if vocab_size > 1 else 0.0

        return entropy_raw, entropy_normalized, vocab_size

    # ------------------------------------------------------------------
    # KL divergence (raw key cache)
    # ------------------------------------------------------------------
    @staticmethod
    def _make_magnitude_dist(kc: torch.Tensor, eps: float) -> torch.Tensor:
        x = kc.detach().float().abs().flatten()
        return x / (x.sum() + eps)

    def _compute_kl(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """KL(p || q)."""
        min_len = min(len(p), len(q))
        if self.kl_vector_align == "last":
            p = p[-min_len:]
            q = q[-min_len:]
        else:
            p = p[:min_len]
            q = q[:min_len]
        # Re-normalize after slicing
        p = p / (p.sum() + self.kl_eps)
        q = q / (q.sum() + self.kl_eps)
        kl = (p * (torch.log(p + self.kl_eps) - torch.log(q + self.kl_eps))).sum().item()
        return kl

    # ------------------------------------------------------------------
    # Compute all metrics
    # ------------------------------------------------------------------
    def _compute_metrics(self, dataset: str) -> List[Dict[str, Any]]:
        """Iterate over records, compute entropy and KL, return list of metric rows."""
        # Group records by (model_type, subject, question_id, stage, cache_name)
        grouped: Dict[tuple, List[Dict]] = defaultdict(list)
        for r in self._records:
            key = (r["model_type"], r["subject"], r["question_id"], r["stage"], r["cache_name"])
            grouped[key].append(r)

        rows = []
        for (model_type, subject, qid, stage, cache_name), recs in grouped.items():
            # Sort by layer_idx
            recs.sort(key=lambda x: x["layer_idx"])

            # Determine lm_head name for entropy
            lm_head_name = self._resolve_lm_head_name(model_type, cache_name)

            prev_dist = None
            for rec in recs:
                layer_idx = rec["layer_idx"]
                kc = rec["key_cache"]

                # Entropy
                ent_result = self._compute_logit_lens_entropy(kc, lm_head_name)
                if ent_result is not None:
                    entropy_raw, entropy_norm, vs = ent_result
                else:
                    entropy_raw, entropy_norm, vs = float("nan"), float("nan"), 0

                # KL
                curr_dist = self._make_magnitude_dist(kc, self.kl_eps)
                if prev_dist is not None:
                    kl_val = self._compute_kl(curr_dist, prev_dist)
                else:
                    kl_val = float("nan")
                prev_dist = curr_dist

                rows.append({
                    "dataset": dataset,
                    "subject": subject,
                    "question_id": qid,
                    "model_type": model_type,
                    "stage": stage,
                    "cache_name": cache_name,
                    "layer_idx": layer_idx,
                    "entropy_raw": entropy_raw,
                    "entropy_normalized": entropy_norm,
                    "logit_lens_temperature": self.logit_lens_temperature,
                    "vocab_size": vs,
                    "kl_prev_layer": kl_val,
                    "kl_vector_align": self.kl_vector_align,
                })
        return rows

    def _resolve_lm_head_name(self, model_type: str, cache_name: str) -> str:
        """Map (model_type, cache_name) → registered lm_head name."""
        if model_type in ("rosetta", "two_stage_rosetta"):
            if cache_name in ("receiver", "fused", "sharer"):
                return "receiver"  # target/base model lm_head
        if model_type in ("two_stage", "two_stage_rosetta"):
            if cache_name == "stage1":
                return "stage1"
            if cache_name == "stage2":
                return "stage2"
        # Default fallback
        return cache_name

    # ------------------------------------------------------------------
    # CSV writer
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
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[CacheAnalysis] Metrics CSV saved to {path}")

    # ------------------------------------------------------------------
    # Raw cache saving
    # ------------------------------------------------------------------
    def _save_raw_caches(self):
        raw_dir = self.ca_dir / "raw_cache"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for (subject, qid, mtype, stage, cname, lidx), kc in self._raw_caches.items():
            fname = f"{subject}_{qid}_{mtype}_{stage}_{cname}_layer{lidx}_rank{self.rank}.pt"
            torch.save(kc, raw_dir / fname)
        print(f"[CacheAnalysis] Raw caches saved ({len(self._raw_caches)} files)")

    # ------------------------------------------------------------------
    # PCA scatter-plots
    # ------------------------------------------------------------------
    def _generate_pca_plots(self):
        """Generate per-layer PCA scatter-plots grouped by model_type and subject."""
        try:
            from sklearn.decomposition import PCA
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("[CacheAnalysis] sklearn or matplotlib not found – skipping PCA plots.")
            return

        # Group records by (model_type, subject, layer_idx)
        grouped: Dict[Tuple[str, str, int], Dict[str, List[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for r in self._records:
            key = (r["model_type"], r["subject"], r["layer_idx"])
            kc = r["key_cache"]  # [B, H, T, D]
            grouped[key][r["cache_name"]].append(kc)

        rng = np.random.RandomState(self.pca_seed)

        for (model_type, subject, layer_idx), cname_caches in grouped.items():
            pca_dir = self.ca_dir / "pca" / model_type / subject
            pca_dir.mkdir(parents=True, exist_ok=True)

            # Check dim compatibility
            dims = {}
            for cname, tensors in cname_caches.items():
                d = tensors[0].shape[-1] * tensors[0].shape[1]  # H * D_head
                dims[cname] = d
            unique_dims = set(dims.values())
            if len(unique_dims) > 1:
                warnings.warn(
                    f"[CacheAnalysis] PCA dim mismatch at layer {layer_idx} for {model_type}/{subject}: "
                    f"{dims}. Plotting only comparable groups."
                )

            # Group by dimension
            dim_groups: Dict[int, Dict[str, List[torch.Tensor]]] = defaultdict(dict)
            for cname, tensors in cname_caches.items():
                dim_groups[dims[cname]][cname] = tensors

            colors = C2C_COLORS if model_type in ("rosetta", "two_stage_rosetta") else TWO_STAGE_COLORS

            for dim_val, group in dim_groups.items():
                fig, ax = plt.subplots(figsize=(8, 6))
                all_points = []
                all_labels = []

                for cname, tensors in group.items():
                    # Reshape each tensor to [N, D] where D = H*D_head
                    pts_list = []
                    for t in tensors:
                        B, H, T, D = t.shape
                        pts_list.append(t.permute(0, 2, 1, 3).reshape(-1, H * D).numpy())
                    pts = np.concatenate(pts_list, axis=0)

                    # Subsample
                    if len(pts) > self.max_pca_points:
                        idx = rng.choice(len(pts), self.max_pca_points, replace=False)
                        pts = pts[idx]

                    all_points.append(pts)
                    all_labels.extend([cname] * len(pts))

                combined = np.concatenate(all_points, axis=0)
                if combined.shape[0] < 2 or combined.shape[1] < 2:
                    plt.close(fig)
                    continue

                pca = PCA(n_components=2)
                transformed = pca.fit_transform(combined)

                offset = 0
                for cname, pts in zip(group.keys(), all_points):
                    n = len(pts)
                    color = colors.get(cname, "gray")
                    ax.scatter(
                        transformed[offset:offset + n, 0],
                        transformed[offset:offset + n, 1],
                        c=color, label=cname, alpha=0.4, s=4,
                    )
                    offset += n

                ax.set_title(f"{model_type} / {subject} / Layer {layer_idx}")
                ax.legend(fontsize=8)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                fig.tight_layout()
                fig.savefig(pca_dir / f"layer_{layer_idx:02d}.png", dpi=120)
                plt.close(fig)

        print(f"[CacheAnalysis] PCA plots saved under {self.ca_dir / 'pca'}")

    # ------------------------------------------------------------------
    # Line-plots (entropy / KL)
    # ------------------------------------------------------------------
    def _generate_line_plots(self, metric_rows: List[Dict[str, Any]]):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("[CacheAnalysis] matplotlib not found – skipping line-plots.")
            return

        # Group by model_type
        by_model: Dict[str, List[Dict]] = defaultdict(list)
        for row in metric_rows:
            by_model[row["model_type"]].append(row)

        for model_type, rows in by_model.items():
            plot_dir = self.ca_dir / "plots" / model_type
            plot_dir.mkdir(parents=True, exist_ok=True)

            colors = C2C_COLORS if model_type in ("rosetta", "two_stage_rosetta") else TWO_STAGE_COLORS

            # --- Entropy lineplot ---
            self._lineplot(
                rows, "layer_idx", "entropy_raw",
                colors, plot_dir / "entropy_layerwise.png",
                title=f"{model_type} — Layerwise Entropy (raw)",
                ylabel="Entropy (raw)",
            )

            # --- KL lineplot ---
            self._lineplot(
                rows, "layer_idx", "kl_prev_layer",
                colors, plot_dir / "kl_prev_layerwise.png",
                title=f"{model_type} — Layerwise KL(l || l-1)",
                ylabel="KL divergence",
                skip_nan=True,
            )

        print(f"[CacheAnalysis] Line-plots saved under {self.ca_dir / 'plots'}")

    @staticmethod
    def _lineplot(rows, x_key, y_key, color_map, save_path,
                  title="", ylabel="", skip_nan=False):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Group by cache_name → list of (layer, value)
        by_cache: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
        for r in rows:
            val = r[y_key]
            if skip_nan and (val != val):  # NaN check
                continue
            by_cache[r["cache_name"]][r[x_key]].append(val)

        fig, ax = plt.subplots(figsize=(10, 5))
        for cname, layer_vals in sorted(by_cache.items()):
            layers = sorted(layer_vals.keys())
            means = [np.mean(layer_vals[l]) for l in layers]
            stds = [np.std(layer_vals[l]) for l in layers]
            color = color_map.get(cname, "gray")
            ax.plot(layers, means, label=cname, color=color)
            ax.fill_between(
                layers,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.15, color=color,
            )

        ax.set_xlabel("Layer index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
