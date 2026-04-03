"""
run_experiments.py
==================
Runs Experiment 1: Communication Efficiency.

Compares A2 (block-coordinate minimisation) against A4 (gradient descent
with partial participation) across K in {1, 2, 3}.

Results are saved to results/ as numpy .npz files so the notebook can
load them without rerunning the algorithms.

Usage (from project root):
    python src/run_experiments.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


import os
import numpy as np
import pandas as pd

from src.graph import get_station_index, build_knn_graph
from src.models import build_local_datasets, init_weights, mse
from src.fl_algorithm import run_a2, run_a4

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_PATH    = "data/daily_records.csv"
RESULTS_DIR  = "results"

K_VALUES     = [1, 2, 3]
ALPHA        = 1.0
LR           = 0.001
N_ITER_A2    = 100
N_ITER_A4    = 1000
SEED         = 0


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_test_loss(history: dict, data: dict) -> float:
    """Compute mean test MSE from the final weights in a history dict."""
    weights = history["weights"]
    losses = [
        mse(data[i]["X_test"], data[i]["y_test"], weights[i])
        for i in weights
    ]
    return float(np.mean(losses))


def save_history(history: dict, data: dict, name: str, results_dir: str) -> None:
    """
    Save a run's history to a .npz file.

    Saved arrays:
        train_loss      : mean train MSE per iteration
        val_loss        : mean val   MSE per iteration
        messages_sent   : messages sent per iteration
        cum_messages    : cumulative messages sent
        test_loss       : scalar, final test MSE (stored as 0-d array)
    """
    path = os.path.join(results_dir, f"{name}.npz")
    np.savez(
        path,
        train_loss    = np.array(history["train_loss"]),
        val_loss      = np.array(history["val_loss"]),
        messages_sent = np.array(history["messages_sent"]),
        cum_messages  = np.array(history["cum_messages"]),
        test_loss     = np.array(compute_test_loss(history, data)),
    )
    print(f"  saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data...")
    df       = pd.read_csv(DATA_PATH)
    stations = get_station_index(df)
    A, _     = build_knn_graph(stations, k=3, weight="distance")
    data     = build_local_datasets(df, stations)
    w0       = init_weights(data, seed=SEED)
    print(f"  {len(stations)} stations, graph built\n")

    # ── A2 ─────────────────────────────────────────────────────────────────────
    print(f"Running A2 ({N_ITER_A2} iterations)...")
    h_a2 = run_a2(data, w0, A, alpha=ALPHA, n_iterations=N_ITER_A2)
    print(f"  train: {h_a2['train_loss'][-1]:.4f}  "
          f"val: {h_a2['val_loss'][-1]:.4f}  "
          f"test: {compute_test_loss(h_a2, data):.4f}  "
          f"messages: {h_a2['cum_messages'][-1]:,}")
    save_history(h_a2, data, "a2", RESULTS_DIR)
    print()

    # ── A4 across K values ─────────────────────────────────────────────────────
    for K in K_VALUES:
        print(f"Running A4 K={K} ({N_ITER_A4} iterations)...")
        h_a4 = run_a4(
            data, w0, A, stations,
            alpha=ALPHA, lr=LR, K=K,
            n_iterations=N_ITER_A4,
        )
        print(f"  train: {h_a4['train_loss'][-1]:.4f}  "
              f"val: {h_a4['val_loss'][-1]:.4f}  "
              f"test: {compute_test_loss(h_a4, data):.4f}  "
              f"messages: {h_a4['cum_messages'][-1]:,}")
        save_history(h_a4, data, f"a4_k{K}", RESULTS_DIR)
        print()

    # ── Summary table ──────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"{'Algorithm':<12} {'Test MSE':>10} {'Total messages':>16}")
    print("-" * 60)

    results = {}
    for name in ["a2", "a4_k1", "a4_k2", "a4_k3"]:
        r = np.load(os.path.join(RESULTS_DIR, f"{name}.npz"))
        results[name] = r
        print(f"{name:<12} {float(r['test_loss']):>10.4f} {int(r['cum_messages'][-1]):>16,}")

    print("=" * 60)


if __name__ == "__main__":
    main()