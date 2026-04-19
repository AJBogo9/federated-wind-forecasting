"""Generate a LaTeX results table from experiment .npz files.

Usage:
    python src/generate_table.py

Reads results/baseline.npz, results/geo.npz, results/corr.npz and writes
report/results_table.tex, which is \\input-ed by FLReport.tex.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC_DIR = Path(__file__).parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

RESULTS_DIR = Path("results")
OUTPUT_PATH = Path("report/results_table.tex")

SYSTEMS = [
    ("Baseline ($\\alpha=0$)", "baseline.npz"),
    ("System 1 — Geographic",  "geo.npz"),
    ("System 2 — Correlation", "corr.npz"),
]


def _load(filename: str) -> dict:
    d = np.load(RESULTS_DIR / filename, allow_pickle=True)
    return {
        "alpha":      float(d["alpha"]),
        "train_loss": float(d["train_loss"][-1]),
        "val_loss":   float(d["val_loss"][-1]),
        "test_loss":  float(d["test_loss"]),
    }


def main() -> None:
    rows = [(label, _load(fname)) for label, fname in SYSTEMS]

    best_test = min(r["test_loss"] for _, r in rows)

    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Mean MSE across all stations (normalised labels). "
                 r"Bold marks the best test MSE.}")
    lines.append(r"  \label{tab:results}")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \hline")
    lines.append(r"    \textbf{System} & $\bm{\alpha}$ & \textbf{Train MSE} "
                 r"& \textbf{Val MSE} & \textbf{Test MSE} \\")
    lines.append(r"    \hline")

    for label, r in rows:
        test_str = f"{r['test_loss']:.4f}"
        if r["test_loss"] == best_test:
            test_str = r"\textbf{" + test_str + "}"
        lines.append(
            f"    {label} & {r['alpha']:.2f} & "
            f"{r['train_loss']:.4f} & {r['val_loss']:.4f} & {test_str} \\\\"
        )

    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Written → {OUTPUT_PATH}")
    for label, r in rows:
        print(f"  {label:30s}  test={r['test_loss']:.4f}")


if __name__ == "__main__":
    main()
