"""Graph construction for the federated learning network.

Two graph topologies are implemented:
  - System 1 (Geographic): k-NN by haversine distance, Gaussian edge weights.
  - System 2 (Correlation): k-NN by Pearson wind-speed correlation, non-negative weights.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from data import TRAIN_FRAC

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class GraphData(NamedTuple):
    """Immutable representation of a weighted undirected graph.

    Attributes
    ----------
    adjacency:
        Weighted adjacency matrix of shape (N, N).  Symmetric, zero diagonal.
        Entry [i, j] is the edge weight A_{ij} (0 if no edge).
    edge_index:
        Array of shape (E, 2) where each row (i, j) satisfies i < j.
        Only the upper-triangle edges are listed.
    name:
        Human-readable label for this graph (used in plots and tables).
    """

    adjacency: np.ndarray
    edge_index: np.ndarray
    name: str


# ---------------------------------------------------------------------------
# Geographic graph (System 1)
# ---------------------------------------------------------------------------


def geographic_graph(
    stations: pd.DataFrame,
    k: int = 5,
) -> GraphData:
    """Build a k-NN graph using haversine (great-circle) distance.

    Edge weight formula:
        A_{ij} = exp(−d_{ij}² / σ²)
    where d_{ij} is the haversine distance in kilometres and
    σ² is the median of all pairwise squared distances (data-adaptive scale).

    Parameters
    ----------
    stations:
        DataFrame indexed by station name with columns ``lat`` and ``lon``.
    k:
        Number of nearest neighbours per node.

    Returns
    -------
    GraphData with ``name = "Geographic (k={k})"``
    """
    station_names = list(stations.index)
    n = len(station_names)

    # BallTree expects radians
    coords_rad = np.deg2rad(stations[["lat", "lon"]].to_numpy(dtype=np.float64))

    tree = BallTree(coords_rad, metric="haversine")
    # Query k+1 because the nearest neighbour of a point is itself (distance 0)
    dists_rad, indices = tree.query(coords_rad, k=k + 1)

    # Convert from radians to kilometres (Earth radius ≈ 6371 km)
    earth_radius_km = 6371.0
    dists_km = dists_rad * earth_radius_km

    # σ² = median pairwise squared distance (excluding self-distances)
    all_dists = dists_km[:, 1:].ravel()
    sigma_sq = float(np.median(all_dists ** 2))
    if sigma_sq == 0.0:
        sigma_sq = 1.0

    adjacency = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for rank, (j, d) in enumerate(zip(indices[i], dists_km[i])):
            if rank == 0:
                continue  # skip self
            weight = float(np.exp(-(d ** 2) / sigma_sq))
            adjacency[i, j] = weight
            adjacency[j, i] = weight  # symmetric

    edge_index = np.array(
        [[i, j] for i in range(n) for j in range(i + 1, n) if adjacency[i, j] > 0],
        dtype=np.int64,
    )

    return GraphData(adjacency=adjacency, edge_index=edge_index, name=f"Geographic (k={k})")


# ---------------------------------------------------------------------------
# Correlation graph (System 2)
# ---------------------------------------------------------------------------


def correlation_graph(
    df: pd.DataFrame,
    stations: pd.DataFrame,
    k: int = 5,
    train_frac: float = TRAIN_FRAC,
) -> GraphData:
    """Build a k-NN graph by Pearson correlation of wind speed time series.

    Edge weight formula:
        A_{ij} = max(0, corr(ws_i, ws_j))
    Negative correlations receive weight 0 (no disagreement incentive for
    anticorrelated stations). Correlation is computed on the training period
    only to avoid data leakage.

    Parameters
    ----------
    df:
        Raw DataFrame from ``fetch_wind_data`` with columns station, time, ws.
    stations:
        DataFrame indexed by station name (qualified stations only).
    k:
        Number of highest-correlation neighbours per node.
    train_frac:
        Fraction of the timeline that constitutes the training period.

    Returns
    -------
    GraphData with ``name = "Correlation (k={k})"``
    """
    station_names = list(stations.index)
    n = len(station_names)

    # Build a (time × station) pivot of wind speed
    ws_pivot = (
        df[df["station"].isin(station_names)]
        .sort_values("time")
        .pivot(index="time", columns="station", values="ws")
        .reindex(columns=station_names)
    )

    # Restrict to training period
    n_time = len(ws_pivot)
    n_train = int(n_time * train_frac)
    ws_train = ws_pivot.iloc[:n_train]

    # Pearson correlation matrix — fill NaN with row mean before correlating
    ws_filled = ws_train.fillna(ws_train.mean())
    corr_matrix = ws_filled.corr(method="pearson").to_numpy(dtype=np.float64).copy()
    np.fill_diagonal(corr_matrix, 0.0)  # zero out self-correlation

    adjacency = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        # Find k stations with highest (non-negative) correlation
        row = corr_matrix[i].copy()
        row[row < 0] = 0.0  # clip negatives
        top_k_indices = np.argsort(row)[::-1][:k]
        for j in top_k_indices:
            if row[j] > 0:
                adjacency[i, j] = row[j]
                adjacency[j, i] = row[j]  # enforce symmetry (take max later)

    # Symmetrise: take max of A[i,j] and A[j,i]
    adjacency = np.maximum(adjacency, adjacency.T)

    edge_index = np.array(
        [[i, j] for i in range(n) for j in range(i + 1, n) if adjacency[i, j] > 0],
        dtype=np.int64,
    )

    return GraphData(adjacency=adjacency, edge_index=edge_index, name=f"Correlation (k={k})")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def graph_summary(stations: pd.DataFrame, graph: GraphData) -> None:
    """Print a summary of graph statistics.

    Parameters
    ----------
    stations:
        Station index DataFrame (for node count label).
    graph:
        GraphData to summarise.
    """
    n = len(stations)
    e = len(graph.edge_index)
    degrees = (graph.adjacency > 0).sum(axis=1)
    weights = graph.adjacency[graph.adjacency > 0]

    print(f"\n{graph.name}")
    print(f"  Nodes : {n}")
    print(f"  Edges : {e}")
    print(f"  Degree: mean={degrees.mean():.1f}, min={degrees.min()}, max={degrees.max()}")
    print(
        f"  Weight: mean={weights.mean():.4f}, "
        f"min={weights.min():.4f}, max={weights.max():.4f}"
    )
