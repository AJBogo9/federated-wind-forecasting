import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def get_station_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract one row per station with its coordinates.

    Returns a DataFrame with columns [station, lat, lon],
    indexed 0..N-1. This integer index is what the rest of
    the graph functions use to identify nodes.
    """
    stations = (
        df.drop_duplicates("station")[["station", "lat", "lon"]]
        .reset_index(drop=True)
    )
    return stations


def build_knn_graph(
    stations: pd.DataFrame,
    k: int = 3,
    weight: str = "distance",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a k-nearest-neighbour graph over FMI stations.

    Parameters
    ----------
    stations : DataFrame
        Output of get_station_index(). Must have columns lat, lon.
    k : int
        Number of neighbours per node (excluding the node itself).
    weight : {"distance", "binary"}
        How to set edge weights:
        - "distance"  : w_{ij} = exp(-d_{ij} / sigma), where sigma is
                        the median pairwise distance among all edges.
                        Closer stations get higher weights.
        - "binary"    : w_{ij} = 1 for all edges.

    Returns
    -------
    adjacency : np.ndarray, shape (N, N)
        Symmetric weighted adjacency matrix. adjacency[i, j] > 0
        iff stations i and j are connected.
    edge_index : np.ndarray, shape (E, 2)
        Array of undirected edges as (i, j) pairs with i < j.
    """
    coords = stations[["lat", "lon"]].values
    N = len(coords)

    # sklearn's kneighbors includes the node itself, so ask for k+1
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # distances[:, 0] is always 0 (self), skip it
    distances = distances[:, 1:]   # shape (N, k)
    indices = indices[:, 1:]       # shape (N, k)

    # Compute edge weights
    if weight == "distance":
        all_dists = distances.ravel()
        sigma = np.median(all_dists)
        weights = np.exp(-distances / sigma)  # shape (N, k)
    elif weight == "binary":
        weights = np.ones_like(distances)
    else:
        raise ValueError(f"weight must be 'distance' or 'binary', got '{weight}'")

    # Fill adjacency matrix symmetrically
    adjacency = np.zeros((N, N))
    for i in range(N):
        for j_idx in range(k):
            j = indices[i, j_idx]
            w = weights[i, j_idx]
            # Use the average weight if both directions differ slightly
            # (they can when i is in j's kNN but j is not in i's kNN)
            adjacency[i, j] = max(adjacency[i, j], w)
            adjacency[j, i] = max(adjacency[j, i], w)

    # Collect undirected edges (i < j only, to avoid duplicates)
    edge_index = np.array(
        [[i, j] for i in range(N) for j in range(i + 1, N) if adjacency[i, j] > 0]
    )

    return adjacency, edge_index


def graph_summary(
    stations: pd.DataFrame,
    adjacency: np.ndarray,
    edge_index: np.ndarray,
) -> None:
    """
    Print a short summary of the graph for sanity checking.
    """
    N = len(stations)
    E = len(edge_index)
    degrees = (adjacency > 0).sum(axis=1)

    print(f"Nodes : {N}")
    print(f"Edges : {E}")
    print(f"Degree — min: {degrees.min():.0f}  "
          f"mean: {degrees.mean():.1f}  "
          f"max: {degrees.max():.0f}")
    nonzero_weights = adjacency[adjacency > 0]
    print(f"Weight — min: {nonzero_weights.min():.4f}  "
          f"mean: {nonzero_weights.mean():.4f}  "
          f"max: {nonzero_weights.max():.4f}")