"""
run_pipeline.py — One-command reproducible pipeline
====================================================
Runs the entire issue-triage-agent pipeline from cleaned data
through training and evaluation. Produces:

  models/nn_weighted_model.pt    — trained model weights
  models/scaler.pkl              — fitted scaler
  models/encoders.pkl            — one-hot vocabularies + feature order
  models/author_history_lookup.parquet
  models/metrics.json            — all evaluation metrics
  data/processed/X_train.parquet, X_test.parquet, y_train.parquet, y_test.parquet

Usage:
  python run_pipeline.py              # full pipeline (clean → features → train → eval)
  python run_pipeline.py --skip-clean # skip cleaning, use existing cleaned parquets

Note: Ingestion (01_ingest.py) is NOT included. It hits the live GitHub
API, takes ~30 min, and requires GITHUB_TOKEN. Run it separately first:
  python src/01_ingest.py

The pipeline starts from 02_clean.py, which reads data/raw/*.json.
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# ── Config ───────────────────────────────────────────────────────────
SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"


# ── Seed locking ─────────────────────────────────────────────────────
def lock_seeds(seed):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Deterministic mode for PyTorch (slight speed cost, full reproducibility)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(f"  Seeds locked: random/numpy/torch all set to {seed}")
    print(f"  torch.use_deterministic_algorithms(True)")


# ── Module loader (handles numeric-prefix filenames) ─────────────────
def load_module(name, filename):
    """Import a script by file path, bypassing Python's module naming rules."""
    path = SRC / filename
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Pipeline steps ───────────────────────────────────────────────────
def step_clean():
    print("\n" + "=" * 60)
    print("STEP 1/4 — CLEAN (02_clean.py)")
    print("=" * 60 + "\n")
    mod = load_module("clean", "02_clean.py")
    mod.main()


def step_features(seed):
    print("\n" + "=" * 60)
    print("STEP 2/4 — FEATURES (03_features.py)")
    print("=" * 60 + "\n")
    mod = load_module("features", "03_features.py")
    mod.main(seed=seed)


def step_train(seed):
    print("\n" + "=" * 60)
    print("STEP 3/4 — TRAIN (06b_nn_weighted.py)")
    print("=" * 60 + "\n")
    mod = load_module("train", "06b_nn_weighted.py")
    return mod.main(seed=seed)


def step_eval():
    print("\n" + "=" * 60)
    print("STEP 4a/4 — PRECISION@K (07_precision_at_k.py)")
    print("=" * 60 + "\n")
    mod_pk = load_module("precision_at_k", "07_precision_at_k.py")
    pk_metrics = mod_pk.main()

    print("\n" + "=" * 60)
    print("STEP 4b/4 — OOD EVALUATION (08_ood_eval.py)")
    print("=" * 60 + "\n")
    mod_ood = load_module("ood_eval", "08_ood_eval.py")
    ood_metrics = mod_ood.main()

    return pk_metrics, ood_metrics


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Issue Triage Agent — reproducible pipeline")
    parser.add_argument("--skip-clean", action="store_true",
                        help="Skip cleaning step; use existing cleaned parquets")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed for reproducibility (default: {SEED})")
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 60)
    print("  ISSUE TRIAGE AGENT — REPRODUCIBLE PIPELINE")
    print("=" * 60)
    print(f"  Started:  {datetime.now(timezone.utc).isoformat()}")
    print(f"  Seed:     {args.seed}")
    print(f"  Project:  {PROJECT_ROOT}")

    # Lock seeds BEFORE any step
    lock_seeds(args.seed)

    # Verify raw data exists
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_files = list(raw_dir.glob("*.json"))
    metadata_file = raw_dir / "_metadata.json"
    if not metadata_file.exists() or len(raw_files) < 2:
        print("\n❌  No raw data found in data/raw/.")
        print("    Run ingestion first:  python 01_ingest.py")
        sys.exit(1)
    print(f"  Raw data: {len(raw_files)} files in {raw_dir}")

    # Run pipeline
    if args.skip_clean:
        processed = PROJECT_ROOT / "data" / "processed" / "cleaned_train.parquet"
        if not processed.exists():
            print("\n❌  --skip-clean used but cleaned_train.parquet not found.")
            sys.exit(1)
        print("\n  Skipping clean step (--skip-clean)")
    else:
        step_clean()

    step_features(seed=args.seed)
    train_metrics = step_train(seed=args.seed)
    pk_metrics, ood_metrics = step_eval()

    # ── Save metrics.json ────────────────────────────────────────────
    MODELS_DIR.mkdir(exist_ok=True)
    elapsed = time.time() - t0

    all_metrics = {
        "pipeline_version": "v0.1",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "runtime_seconds": round(elapsed, 1),
        "training": train_metrics,
        "precision_at_k": pk_metrics,
        "ood": ood_metrics,
        "success_criteria": {
            "in_domain_f1_pass": train_metrics["f1"] >= 0.5237,
            "in_domain_p_at_5_pass": pk_metrics["in_domain_p_at_5_pass"],
            "ood_p_at_5_pass": ood_metrics["ood_p_at_5_pass"],
        },
    }

    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Runtime:        {elapsed:.1f}s")
    print(f"  Seed:           {args.seed}")
    print(f"  Model F1:       {train_metrics['f1']}")
    print(f"  In-domain P@5:  {pk_metrics['in_domain_p_at_5']}  "
          f"{'✅' if pk_metrics['in_domain_p_at_5_pass'] else '❌'}")
    print(f"  OOD P@5:        {ood_metrics['ood_p_at_5']}  "
          f"{'✅' if ood_metrics['ood_p_at_5_pass'] else '❌'}")
    print(f"\n  Artifacts saved to: {MODELS_DIR}/")
    print(f"  Metrics:            {metrics_path}")
    print()


if __name__ == "__main__":
    main()