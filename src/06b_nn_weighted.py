"""
06b_nn_weighted.py — NN with repo-based sample weighting
=========================================================
Same 2-layer NN as 06_nn.py, but adds per-sample weights
so that underrepresented repos (langchain) get more
influence during training.

Combines TWO reweighting strategies:
  1. pos_weight: class imbalance (YES vs NO)
  2. sample_weights: repo imbalance (pytorch vs langchain)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"


# ── Network architecture (module-level so other scripts can import) ──
class IssueTriage_NN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def to_tensors(X, y):
    return (
        torch.tensor(X.values, dtype=torch.float32),
        torch.tensor(y.values, dtype=torch.float32).unsqueeze(1),
    )


def per_repo_f1(y_true, y_pred, X):
    repo_cols = [c for c in X.columns if c.startswith("repo_")]
    results = {}
    for col in repo_cols:
        name = col.replace("repo_", "")
        mask = X[col].values == 1
        if mask.sum() == 0:
            continue
        f1 = f1_score(y_true[mask], y_pred[mask], zero_division=0)
        n = mask.sum()
        results[name] = (f1, n)
    return results


# ── Main ─────────────────────────────────────────────────────────────
def main(seed=42):
    MODELS_DIR.mkdir(exist_ok=True)

    # Load data
    X_train_full = pd.read_parquet(PROCESSED / "X_train.parquet")
    X_test       = pd.read_parquet(PROCESSED / "X_test.parquet")
    y_train_full = pd.read_parquet(PROCESSED / "y_train.parquet").squeeze()
    y_test       = pd.read_parquet(PROCESSED / "y_test.parquet").squeeze()

    # Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2, random_state=seed, stratify=y_train_full
    )
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Compute repo-based sample weights
    REPO_COLS = [c for c in X_train.columns if c.startswith("repo_")]

    print("\nRepo distribution in training set:")
    repo_counts = {}
    for col in REPO_COLS:
        name = col.replace("repo_", "")
        count = (X_train[col] == 1).sum()
        repo_counts[col] = count
        print(f"  {name:30s}  {count:,} rows")

    n_repos = len(REPO_COLS)
    total_rows = len(X_train)

    sample_weights = np.ones(total_rows, dtype=np.float32)
    for col in REPO_COLS:
        mask = X_train[col].values == 1
        weight = total_rows / (n_repos * repo_counts[col])
        sample_weights[mask] = weight
        name = col.replace("repo_", "")
        print(f"  {name:30s}  weight = {weight:.2f}")

    sample_weights_t = torch.tensor(sample_weights, dtype=torch.float32)

    # Convert to tensors
    X_tr_t, y_tr_t = to_tensors(X_train, y_train)
    X_vl_t, y_vl_t = to_tensors(X_val, y_val)
    X_te_t, y_te_t = to_tensors(X_test, y_test)

    train_dataset = TensorDataset(X_tr_t, y_tr_t, sample_weights_t)
    train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Model
    n_features = X_train.shape[1]
    model = IssueTriage_NN(n_features)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
    print(f"pos_weight (class) = {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    val_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training loop
    EPOCHS = 100
    PATIENCE = 10

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE})...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, w_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            per_sample_loss = criterion(logits, y_batch)
            weighted_loss = (per_sample_loss * w_batch.unsqueeze(1)).mean()
            weighted_loss.backward()
            optimizer.step()
            train_loss += weighted_loss.item() * len(X_batch)
        train_loss /= len(X_tr_t)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_vl_t)
            val_loss = val_criterion(val_logits, y_vl_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == PATIENCE:
            print(f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  patience={patience_counter}/{PATIENCE}")

        if patience_counter == PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}. Best val_loss={best_val_loss:.4f}")
            break
    else:
        print(f"\nCompleted all {EPOCHS} epochs. Best val_loss={best_val_loss:.4f}")

    # Load best weights and save
    model.load_state_dict(best_state)
    model_path = MODELS_DIR / "nn_weighted_model.pt"
    torch.save(best_state, model_path)
    print(f"Model saved → {model_path}\n")

    # Predict on test set
    model.eval()
    with torch.no_grad():
        test_logits = model(X_te_t)
        test_probs  = torch.sigmoid(test_logits).squeeze()
        y_pred_test = (test_probs >= 0.5).int().numpy()

    y_test_np = y_test.values

    # Evaluation
    acc  = accuracy_score(y_test_np, y_pred_test)
    prec = precision_score(y_test_np, y_pred_test, zero_division=0)
    rec  = recall_score(y_test_np, y_pred_test, zero_division=0)
    f1   = f1_score(y_test_np, y_pred_test, zero_division=0)
    repo_f1s = per_repo_f1(y_test_np, y_pred_test, X_test)

    print("─" * 60)
    print("  Neural Network (repo-weighted) — Test set")
    print("─" * 60)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 (pos):  {f1:.4f}")
    for repo, (rf1, n) in repo_f1s.items():
        print(f"    └─ {repo:30s} F1 = {rf1:.4f}  (n={n})")

    # Return metrics for run_pipeline.py to collect
    metrics = {
        "model": "nn_repo_weighted",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "per_repo_f1": {repo: round(rf1, 4) for repo, (rf1, _) in repo_f1s.items()},
    }
    return metrics


if __name__ == "__main__":
    main()