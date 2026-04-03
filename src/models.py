import numpy as np
import pandas as pd


# ── Split sizes (41 days total) ────────────────────────────────────────────────
N_TRAIN = 28
N_VAL   = 7
N_TEST  = 6


def build_local_datasets(
    df: pd.DataFrame,
    stations: pd.DataFrame,
) -> dict:
    """
    Build local (X, y) datasets for each station node.

    Feature : tmin  (scalar, so X has shape (n_samples, 1) with bias column)
    Label   : tmax

    A bias term (column of ones) is appended to X so that the local model
    w = [w_1, w_0]^T represents  tmax_hat = w_1 * tmin + w_0,
    and the gradient is simply  X^T (Xw - y).

    The 41 days are split chronologically:
        train : days  0-27  (28 days)
        val   : days 28-34  ( 7 days)
        test  : days 35-40  ( 6 days)

    Parameters
    ----------
    df : DataFrame
        Full dataset (output of fetch_data).
    stations : DataFrame
        Output of get_station_index(). Provides the integer node index.

    Returns
    -------
    data : dict  { node_index -> {"X_train", "y_train",
                                  "X_val",   "y_val",
                                  "X_test",  "y_test"} }
    """
    data = {}

    for node_i, row in stations.iterrows():
        station_name = row["station"]

        # Select this station's rows, sorted by date
        sub = (
            df[df["station"] == station_name]
            .sort_values("day")
            .reset_index(drop=True)
        )

        # Raw feature and label vectors
        tmin = sub["tmin"].values.astype(float)   # shape (41,)
        tmax = sub["tmax"].values.astype(float)   # shape (41,)

        # Add bias column: X[:, 0] = tmin, X[:, 1] = 1
        X = np.column_stack([tmin, np.ones(len(tmin))])  # shape (41, 2)
        y = tmax                                          # shape (41,)

        # Chronological split
        X_train, y_train = X[:N_TRAIN],               y[:N_TRAIN]
        X_val,   y_val   = X[N_TRAIN:N_TRAIN+N_VAL],  y[N_TRAIN:N_TRAIN+N_VAL]
        X_test,  y_test  = X[N_TRAIN+N_VAL:],         y[N_TRAIN+N_VAL:]

        data[node_i] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val":   X_val,
            "y_val":   y_val,
            "X_test":  X_test,
            "y_test":  y_test,
        }

    return data


def mse(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """
    Mean squared error:  (1/n) * ||Xw - y||^2
    """
    residuals = X @ w - y
    return float(np.mean(residuals ** 2))


def local_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE loss at node i:
        grad = (2/n) * X^T (Xw - y)

    Parameters
    ----------
    X : shape (n, d)
    y : shape (n,)
    w : shape (d,)

    Returns
    -------
    grad : shape (d,)
    """
    n = len(y)
    residuals = X @ w - y
    return (2.0 / n) * (X.T @ residuals)


def init_weights(data: dict, seed: int = 0) -> dict:
    """
    Initialise model parameters for all nodes.

    Each w^(i) is drawn from N(0, 0.01) so all nodes start close to
    zero, giving a consistent baseline regardless of random seed.

    Parameters
    ----------
    data : output of build_local_datasets()
    seed : random seed for reproducibility

    Returns
    -------
    weights : dict { node_index -> w^(i) of shape (d,) }
    """
    rng = np.random.default_rng(seed)
    weights = {}
    for node_i, node_data in data.items():
        d = node_data["X_train"].shape[1]
        weights[node_i] = rng.normal(0, 0.01, size=d)
    return weights