"""Reproducible experiment script for federated wind speed forecasting.

Run this script to reproduce all results:

    python src/run_experiments.py

The script:
  1. Loads (or fetches) hourly FMI data for full-year 2024.
  2. Filters stations with >= 80% hourly coverage and builds per-node datasets.
  3. Constructs two FL network graphs:
       - System 1: Geographic k-NN (distance-weighted)
       - System 2: Correlation k-NN (Pearson-weighted)
  4. Runs an α-sweep (grid search) for both systems to select the best
     regularisation strength via validation loss.
  5. Runs the full FedGD with the selected α for both systems and a
     baseline (α=0, geographic graph).
  6. Saves results to the ``results/`` directory as .npz files.

Requirements: numpy, pandas, scipy, scikit-learn, requests  (no plotly).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Allow running as  python src/run_experiments.py  from the repo root
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data import (
    SEED,
    build_node_datasets,
    fetch_wind_data,
    get_station_index,
    init_weights,
)
from fedgd import FedGDConfig, FedGDHistory, run_fedgd
from graph import GraphData, correlation_graph, geographic_graph, graph_summary

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

K: int = 5                  # k-NN degree for both graph systems
LR: float = 0.01            # FedGD learning rate
N_ITERS_SWEEP: int = 300    # iterations for the α-sweep
N_ITERS_FULL: int = 750     # iterations for the full runs

ALPHA_GRID: list[float] = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

DATA_PATH: Path = Path("data/wind_2024.csv")
RESULTS_DIR: Path = Path("results")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_history(history: FedGDHistory, path: Path, system_name: str) -> None:
    """Save a FedGDHistory to a .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        train_loss=np.array(history["train_loss"]),
        val_loss=np.array(history["val_loss"]),
        messages_sent=np.array(history["messages_sent"]),
        cum_messages=np.array(history["cum_messages"]),
        test_loss=float(history["test_loss"]),
        alpha=float(history["alpha"]),
        system_name=system_name,
        weights=np.array(history["weights"], dtype=object),
    )
    print(f"  Saved → {path}  (test MSE={history['test_loss']:.4f})")


def _alpha_sweep(
    datasets: dict,
    graph: GraphData,
    init_w: dict,
    alpha_grid: list[float],
    n_iters: int,
    lr: float,
) -> dict[float, float]:
    """Run FedGD for each α in the grid; return {alpha: final_val_loss}."""
    results: dict[float, float] = {}
    for alpha in alpha_grid:
        cfg = FedGDConfig(alpha=alpha, lr=lr, n_iters=n_iters)
        history = run_fedgd(datasets, graph, cfg, init_w)
        results[alpha] = history["val_loss"][-1]
        print(f"    α={alpha:.3f}  val_loss={results[alpha]:.4f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ---- 1. Data ----
    print("=" * 60)
    print("Step 1: Loading data")
    print("=" * 60)
    df = fetch_wind_data(cache_path=DATA_PATH)
    stations = get_station_index(df)
    datasets = build_node_datasets(df, stations)

    # Initialise weights once — shared across all runs for fair comparison
    init_w = init_weights(datasets, seed=SEED)

    n_stations = len(datasets)
    print(f"Using {n_stations} stations for federated learning.")

    # ---- 2. Graphs ----
    print("\n" + "=" * 60)
    print("Step 2: Building graphs")
    print("=" * 60)
    geo_graph = geographic_graph(stations, k=K)
    corr_graph = correlation_graph(df, stations, k=K)

    graph_summary(stations, geo_graph)
    graph_summary(stations, corr_graph)

    # ---- 3. α sweep ----
    print("\n" + "=" * 60)
    print("Step 3: α sweep (validation loss)")
    print("=" * 60)

    print(f"\nSystem 1 — {geo_graph.name}")
    geo_sweep = _alpha_sweep(datasets, geo_graph, init_w, ALPHA_GRID, N_ITERS_SWEEP, LR)

    print(f"\nSystem 2 — {corr_graph.name}")
    corr_sweep = _alpha_sweep(datasets, corr_graph, init_w, ALPHA_GRID, N_ITERS_SWEEP, LR)

    # Select best α (excluding α=0 from the selection — that is the baseline)
    geo_best_alpha = min(
        (a for a in geo_sweep if a > 0),
        key=lambda a: geo_sweep[a],
    )
    corr_best_alpha = min(
        (a for a in corr_sweep if a > 0),
        key=lambda a: corr_sweep[a],
    )

    print(f"\n  Best α — Geographic : {geo_best_alpha}")
    print(f"  Best α — Correlation: {corr_best_alpha}")

    # Save sweep results
    sweep_path = RESULTS_DIR / "alpha_sweep.npz"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        sweep_path,
        alpha_grid=np.array(ALPHA_GRID),
        geo_val_loss=np.array([geo_sweep[a] for a in ALPHA_GRID]),
        corr_val_loss=np.array([corr_sweep[a] for a in ALPHA_GRID]),
        geo_best_alpha=geo_best_alpha,
        corr_best_alpha=corr_best_alpha,
    )
    print(f"  Saved sweep → {sweep_path}")

    # ---- 4. Full runs ----
    print("\n" + "=" * 60)
    print("Step 4: Full FedGD runs")
    print("=" * 60)

    # Baseline: α=0 (no collaboration) on the geographic graph
    print(f"\nBaseline (α=0.0, geographic graph) — {N_ITERS_FULL} iters")
    baseline_cfg = FedGDConfig(alpha=0.0, lr=LR, n_iters=N_ITERS_FULL)
    baseline_history = run_fedgd(datasets, geo_graph, baseline_cfg, init_w)
    _save_history(baseline_history, RESULTS_DIR / "baseline.npz", "Baseline (α=0)")

    # System 1: Geographic, best α
    print(f"\nSystem 1 — {geo_graph.name}, α={geo_best_alpha} — {N_ITERS_FULL} iters")
    geo_cfg = FedGDConfig(alpha=geo_best_alpha, lr=LR, n_iters=N_ITERS_FULL)
    geo_history = run_fedgd(datasets, geo_graph, geo_cfg, init_w)
    _save_history(geo_history, RESULTS_DIR / "geo.npz", geo_graph.name)

    # System 2: Correlation, best α
    print(f"\nSystem 2 — {corr_graph.name}, α={corr_best_alpha} — {N_ITERS_FULL} iters")
    corr_cfg = FedGDConfig(alpha=corr_best_alpha, lr=LR, n_iters=N_ITERS_FULL)
    corr_history = run_fedgd(datasets, corr_graph, corr_cfg, init_w)
    _save_history(corr_history, RESULTS_DIR / "corr.npz", corr_graph.name)

    # ---- 5. Summary ----
    print("\n" + "=" * 60)
    print("Summary — Test MSE (mean across all nodes)")
    print("=" * 60)
    print(f"  Baseline (α=0)            : {baseline_history['test_loss']:.4f}")
    print(f"  System 1 (Geographic)     : {geo_history['test_loss']:.4f}")
    print(f"  System 2 (Correlation)    : {corr_history['test_loss']:.4f}")

    if geo_history["test_loss"] < baseline_history["test_loss"]:
        print("\n  ✓ System 1 outperforms baseline (FL helps).")
    else:
        print("\n  ✗ System 1 does NOT outperform baseline — check α or lr.")

    print("\nAll results saved to:", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()
