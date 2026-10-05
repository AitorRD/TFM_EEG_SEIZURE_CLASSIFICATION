"""
Experiment 2 — fine-tune the best foundation model of experiment 1
(loso_preictal.py) and repeat the same LOSO seizure-prediction protocol.

Same data, labels, folds, validation patients, in-context rows, threshold
rule and metrics as loso_preictal.py; the only change is that the model's
weights are adapted on each fold's training patients before predicting:

  --method lora  Low-Rank Adaptation: every attention / MLP nn.Linear of the
                 transformer gets a trainable low-rank update W + (alpha/r)·B·A
                 (B initialised to zero, so training starts exactly from the
                 pretrained model); all pretrained weights stay frozen. ~1-2%
                 of the parameters are trained, which limits overfitting to
                 the ~20 training patients.
  --method full  Full fine-tuning of every weight (the libraries' default).

Each library's own fine-tuning loop is used (episodic: random context/query
splits of the training rows, cross-entropy on the queries) and LoRA is
injected into the model before the optimizer is built. Early stopping runs
on the fold's held-out validation patients — the same ones that pick the
decision threshold — so the test patient never influences training.

Supported: TabPFNv3 / TabPFNv2 (FinetunedTabPFNClassifier; on v2 only the MLPs
are nn.Linear, so LoRA covers MLPs only) and Mitra (MitraClassifier with
fine_tune=True). TabICL and TabFM ship no training loop.

Usage:
    python loso_preictal_finetune.py --best-from images/results/loso_preictal_chbmit_efficient_h60_sph0
    python loso_preictal_finetune.py --model tabpfnv3 --method lora --lora-r 8
    python loso_preictal_finetune.py --model mitra --method full
"""

from __future__ import annotations

import math
import re
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

import loso_foundation_report as L
import loso_preictal as P
from loso_foundation_report import print  # noqa: A004 - timestamped, flushed print

FINETUNABLE = {"tabpfnv3", "tabpfnv2", "mitra"}

# Attention projections and MLP layers of TabPFN v3 (q/k/v/out_projection,
# mlp.0/mlp.2), TabPFN v2 (mlp.linear1/linear2) and Mitra (attention*.q/k/v/o,
# linear1-4). Embedders, decoders and TabPFN v3's softmax-scaling MLPs stay
# frozen.
DEFAULT_LORA_TARGET = r"(^|\.)(q|k|v|o|q_projection|k_projection|v_projection|out_projection|linear[1-4])$|\.mlp\.[02]$"
DEFAULT_LORA_EXCLUDE = r"softmax_scaling_layer|decoder|embed"


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """y = base(x) + (alpha / r) · dropout(x) Aᵀ Bᵀ, with base frozen."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # Some model code reads these off the layer directly (init, dtype checks).
    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        update = (self.dropout(x).to(self.lora_A.dtype) @ self.lora_A.t()) @ self.lora_B.t()
        return out + (update * self.scaling).to(out.dtype)


def inject_lora(model: nn.Module, r: int, alpha: float, dropout: float,
                target: str = DEFAULT_LORA_TARGET, exclude: str = DEFAULT_LORA_EXCLUDE) -> Dict[str, int]:
    """Freeze `model` and wrap every matching nn.Linear in a LoRALinear."""
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise RuntimeError("model already has LoRA adapters")
    for p in model.parameters():
        p.requires_grad_(False)
    target_re, exclude_re = re.compile(target), re.compile(exclude)
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and target_re.search(n) and not exclude_re.search(n)]
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(getattr(parent, child), r, alpha, dropout))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if not names:
        raise RuntimeError("no nn.Linear matched the LoRA target pattern")
    print(f"    LoRA: {len(names)} layers adapted, trainable {trainable:,}/{total:,} "
          f"({100 * trainable / total:.2f}%)")
    return {"layers": len(names), "trainable": trainable, "total": total}


# ---------------------------------------------------------------------------
# TabPFN
# ---------------------------------------------------------------------------

if L.HAS_TABPFN:
    from tabpfn.finetuning import FinetunedTabPFNClassifier

    class TabPFNFinetuner(FinetunedTabPFNClassifier):
        """FinetunedTabPFNClassifier that (a) builds the requested model version
        (upstream always uses the v2.5 defaults) and (b) optionally injects LoRA
        right after the pretrained weights load, before the optimizer exists.

        `_tabpfn_version` / `_lora_cfg` are plain attributes, not __init__
        params, because sklearn's get_params() (called inside fit) only accepts
        explicit constructor arguments."""

        _tabpfn_version = "v3"
        _lora_cfg: Optional[dict] = None

        def _create_estimator(self, config):
            from tabpfn import TabPFNClassifier
            return TabPFNClassifier.create_default_for_version(
                self._tabpfn_version, **config, fit_mode="batched", differentiable_input=False,
            )

        def _setup_estimator(self) -> None:
            super()._setup_estimator()
            if self._lora_cfg is None:
                return
            est = self.finetuned_estimator_
            init = est._initialize_model_variables
            lora_cfg = self._lora_cfg

            def init_with_lora():
                out = init()
                for m in est.models_:
                    if not any(isinstance(x, LoRALinear) for x in m.modules()):
                        inject_lora(m, **lora_cfg)
                return out

            est._initialize_model_variables = init_with_lora


def build_tabpfn(n_rows: int, args, version: str):
    lora = args.method == "lora"
    model = TabPFNFinetuner(
        device="cuda" if torch.cuda.is_available() else "cpu",
        epochs=args.ft_epochs,
        time_limit=args.ft_time_limit,
        learning_rate=args.ft_lr if args.ft_lr is not None else (1e-4 if lora else 1e-5),
        n_finetune_ctx_plus_query_samples=args.ft_chunk_rows,
        early_stopping_patience=args.ft_patience,
        random_state=args.seed,
        save_checkpoint_interval=None,
        eval_metric="roc_auc",
    )
    model._tabpfn_version = version
    model._lora_cfg = lora_cfg(args) if lora else None
    return model


# ---------------------------------------------------------------------------
# Mitra
# ---------------------------------------------------------------------------

def build_mitra(n_rows: int, args):
    lora = args.method == "lora"
    model = L.MitraClassifier(
        # Mitra's default warmup is 1000 *epochs* (stepped once per epoch), so
        # with tens of steps the LR would never leave ~lr/1000.
        fine_tune=True, fine_tune_steps=args.ft_epochs, patience=args.ft_patience,
        warmup_steps=args.mitra_warmup_steps,
        lr=args.ft_lr if args.ft_lr is not None else (1e-4 if lora else 1e-5),
        seed=args.seed, verbose=True,
    )
    create_config = model._create_config
    cfg_lora = lora_cfg(args) if lora else None

    def _create_config(task, dim_output, time_limit=None):
        cfg, model_cls = create_config(task, dim_output, time_limit)
        # Backward pass needs far more memory than inference: start at Mitra's
        # own default (8192) instead of the whole context; Mitra halves it on OOM.
        cfg.hyperparams["max_samples_support"] = min(n_rows, args.mitra_support_rows)
        if cfg_lora is None:
            return cfg, model_cls

        class LoRAModel:
            @staticmethod
            def from_pretrained(*a, **k):
                m = model_cls.from_pretrained(*a, **k)
                inject_lora(m, **cfg_lora)
                return m

        return cfg, LoRAModel

    model._create_config = _create_config
    return model


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def lora_cfg(args) -> dict:
    return {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
            "target": args.lora_target, "exclude": args.lora_exclude}


def pick_best(best_from: Path, metric: str) -> str:
    df = pd.read_csv(best_from / "loso_fold_metrics.csv")
    ranking = df.groupby("model")[metric].mean().sort_values(ascending=False)
    print(f"Experiment 1 ranking by mean {metric} ({best_from}):")
    for name, v in ranking.items():
        print(f"  {name:<10} {v:.4f}")
    name_to_key = {v: k for k, v in L.MODEL_DISPLAY_NAMES.items()}
    best = name_to_key[ranking.index[0]]
    if best not in FINETUNABLE:
        raise SystemExit(
            f"Best model {ranking.index[0]} has no fine-tuning loop here (supported: "
            f"{', '.join(sorted(FINETUNABLE))}). Pass --model explicitly, e.g. the best supported one."
        )
    return best


def comparison_section(all_results: Dict[str, List[Dict]], baseline_csv: Path,
                       baseline_name: str, ft_name: str) -> List[str]:
    """Paired per-patient comparison against the zero-shot (in-context only)
    run of the same model from experiment 1."""
    base = pd.read_csv(baseline_csv)
    base = base[base["model"] == baseline_name].set_index("patient")
    ft = pd.DataFrame([{"patient": r["patient"], **r["metrics"]} for r in all_results[ft_name]]).set_index("patient")
    common = base.index.intersection(ft.index)
    if len(common) == 0:
        return []
    rows = []
    for k in ["ROC AUC", "PR AUC", "Recall", "Specificity", "F1", "Seizure Sensitivity", "FA/h"]:
        if k not in base or k not in ft:
            continue
        d = (ft.loc[common, k] - base.loc[common, k]).dropna()
        better = (d < 0) if k == "FA/h" else (d > 0)
        rows.append([k, f"{base.loc[common, k].mean():.4f}", f"{ft.loc[common, k].mean():.4f}",
                     f"{d.mean():+.4f}", f"{int(better.sum())}/{len(d)}"])
    return [f"## Fine-tuned vs. in-context only ({baseline_name}, {len(common)} paired folds)", "",
            L._md_table(["Metric", "In-context", "Fine-tuned", "Mean Δ", "Folds improved"], rows), ""]


def main() -> None:
    p = P.build_parser("LOSO seizure prediction with a fine-tuned (LoRA / full) foundation model")
    g = p.add_argument_group("fine-tuning")
    g.add_argument("--model", choices=sorted(FINETUNABLE), default=None)
    g.add_argument("--best-from", type=str, default=None,
                   help="loso_preictal.py output dir: fine-tune its best model and compare against it")
    g.add_argument("--select-metric", default="PR AUC", help="Metric used with --best-from")
    g.add_argument("--method", choices=["lora", "full"], default="lora")
    g.add_argument("--lora-r", type=int, default=8)
    g.add_argument("--lora-alpha", type=float, default=16.0)
    g.add_argument("--lora-dropout", type=float, default=0.05)
    g.add_argument("--lora-target", default=DEFAULT_LORA_TARGET)
    g.add_argument("--lora-exclude", default=DEFAULT_LORA_EXCLUDE)
    g.add_argument("--ft-epochs", type=int, default=20,
                   help="TabPFN: epochs over the context; Mitra: fine-tuning steps")
    g.add_argument("--ft-lr", type=float, default=None, help="Default 1e-4 (LoRA) / 1e-5 (full)")
    g.add_argument("--ft-patience", type=int, default=5, help="Early-stopping patience (val patients)")
    g.add_argument("--ft-chunk-rows", type=int, default=2048,
                   help="TabPFN: rows (context + query) per fine-tuning episode")
    g.add_argument("--ft-time-limit", type=int, default=None, help="TabPFN: seconds per fold")
    g.add_argument("--mitra-support-rows", type=int, default=8192)
    g.add_argument("--mitra-warmup-steps", type=int, default=2,
                   help="Linear LR warmup in fine-tuning steps (Mitra's own default, 1000, "
                        "keeps the LR near zero for short runs)")
    args = p.parse_args()

    if args.best_from:
        args.model = args.model or pick_best(Path(args.best_from), args.select_metric)
    if args.model is None:
        raise SystemExit("pass --model or --best-from")
    if args.model == "mitra" and not L.HAS_MITRA or args.model != "mitra" and not L.HAS_TABPFN:
        raise SystemExit(f"{args.model} is not installed")

    tag = f"{args.method}" + (f"_r{args.lora_r}" if args.method == "lora" else "")
    P.resolve_paths(args, out_prefix=f"loso_preictal_ft_{args.model}_{tag}")

    base_name = L.MODEL_DISPLAY_NAMES[args.model]
    ft_name = f"{base_name}-{'LoRA' if args.method == 'lora' else 'FullFT'}"
    if args.model == "mitra":
        build = partial(build_mitra, args=args)
    else:
        build = partial(build_tabpfn, args=args, version=args.model.replace("tabpfn", ""))
    spec = P.ModelSpec(key=f"{args.model}_{tag}", name=ft_name, build=build, uses_val=True)

    lr = args.ft_lr if args.ft_lr is not None else (1e-4 if args.method == "lora" else 1e-5)
    extra_config = [
        ["Fine-tuned model", base_name],
        ["Method", "LoRA" if args.method == "lora" else "full fine-tuning"],
        ["LoRA", f"r={args.lora_r}, alpha={args.lora_alpha:g}, dropout={args.lora_dropout:g}, "
                 f"targets `{args.lora_target}` (excluding `{args.lora_exclude}`)"
         if args.method == "lora" else "—"],
        ["Optimisation", f"AdamW lr={lr:g}, {args.ft_epochs} "
                         f"{'steps' if args.model == 'mitra' else 'epochs'}, early stopping on the "
                         f"validation patients (patience {args.ft_patience})"],
    ]
    if args.model != "mitra":
        extra_config.append(["Episode size", f"{args.ft_chunk_rows} rows (context + query)"])

    sections = None
    if args.best_from:
        baseline_csv = Path(args.best_from) / "loso_fold_metrics.csv"
        sections = partial(comparison_section, baseline_csv=baseline_csv,
                           baseline_name=base_name, ft_name=ft_name)

    P.run_loso(args, [spec], f"LOSO seizure prediction — {ft_name}",
               extra_config=extra_config, extra_sections_fn=sections)


if __name__ == "__main__":
    main()
