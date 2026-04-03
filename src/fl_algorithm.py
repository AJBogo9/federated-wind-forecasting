import numpy as np
import pandas as pd
from .models import local_gradient, mse


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _neighbour_penalty_gradient(
    weights: dict,
    adjacency: np.ndarray,
    node_i: int,
    alpha: float,
) -> np.ndarray:
    """
    Gradient of the GTV penalty term at node i:

        grad_penalty^(i) = 2 * alpha * sum_{i' in N(i)} A_{i,i'} (w^(i) - w^(i'))

    This is the coupling term that pulls node i's parameters
    towards its neighbours.
    """
    w_i = weights[node_i]
    grad = np.zeros_like(w_i)
    for j, a_ij in enumerate(adjacency[node_i]):
        if a_ij > 0 and j != node_i:
            grad += 2.0 * alpha * a_ij * (w_i - weights[j])
    return grad


def _mean_loss(weights: dict, data: dict, split: str) -> float:
    """Average MSE across all nodes for a given split ('train'/'val'/'test')."""
    losses = [
        mse(data[i][f"X_{split}"], data[i][f"y_{split}"], weights[i])
        for i in weights
    ]
    return float(np.mean(losses))


def _count_messages(adjacency: np.ndarray, active_nodes: list[int]) -> int:
    """
    Count messages sent in one iteration.

    Each active node sends its parameters to each of its neighbours
    (active or not) once.  An undirected edge (i, j) where both i and j
    are active is counted as 2 messages (one each way).
    """
    total = 0
    active_set = set(active_nodes)
    N = adjacency.shape[0]
    for i in active_set:
        for j in range(N):
            if adjacency[i, j] > 0 and j != i:
                total += 1  # node i sends to j
    return total


# ── Algorithm A2: Block-coordinate minimisation ────────────────────────────────

def _exact_local_solve(
    data: dict,
    weights: dict,
    adjacency: np.ndarray,
    node_i: int,
    alpha: float,
) -> np.ndarray:
    """
    Exact minimiser of the local subproblem for node i (A2 update):

        min_{w} f_i(w) + alpha * sum_{i'} A_{i,i'} ||w - w^(i')||^2

    For MSE loss f_i(w) = (1/n)||Xw - y||^2 this is a ridge-regression
    problem with a closed-form solution:

        w* = (X^T X / n + alpha * D_i * I)^{-1} (X^T y / n + alpha * D_i_weighted * w_bar)

    where D_i = sum_{i'} A_{i,i'} and
          w_bar = (sum_{i'} A_{i,i'} w^(i')) / D_i
    """
    X = data[node_i]["X_train"]
    y = data[node_i]["y_train"]
    n, d = X.shape

    # Weighted sum of neighbour parameters
    D_i = 0.0
    weighted_neighbour_sum = np.zeros(d)
    for j, a_ij in enumerate(adjacency[node_i]):
        if a_ij > 0 and j != node_i:
            D_i += a_ij
            weighted_neighbour_sum += a_ij * weights[j]

    # Build and solve the linear system
    XtX = (X.T @ X) / n
    Xty = (X.T @ y) / n
    A_mat = XtX + alpha * D_i * np.eye(d)
    b_vec = Xty + alpha * weighted_neighbour_sum

    return np.linalg.solve(A_mat, b_vec)


def run_a2(
    data: dict,
    weights_init: dict,
    adjacency: np.ndarray,
    alpha: float = 1.0,
    n_iterations: int = 100,
) -> dict:
    """
    A2: Block-coordinate minimisation (synchronous exact local solve).

    Each iteration every node solves its local subproblem exactly in
    closed form, using the *previous* iteration's neighbour parameters
    (Gauss-Seidel-style synchronous update).

    Communication cost per iteration: 2|E| messages
    (every node sends to all neighbours, every edge traversed twice).

    Parameters
    ----------
    data            : output of build_local_datasets()
    weights_init    : output of init_weights()
    adjacency       : (N, N) weighted adjacency matrix
    alpha           : GTV regularisation strength
    n_iterations    : number of iterations to run

    Returns
    -------
    history : dict with keys
        "train_loss"    : list of mean train MSE per iteration
        "val_loss"      : list of mean val   MSE per iteration
        "weights"       : final weights dict
        "messages_sent" : list of messages sent per iteration
        "cum_messages"  : list of cumulative messages sent
    """
    weights = {i: w.copy() for i, w in weights_init.items()}
    nodes = list(weights.keys())
    N = adjacency.shape[0]
    messages_per_iter = 2 * int((adjacency > 0).sum() // 2)  # 2|E|

    history = {
        "train_loss":    [],
        "val_loss":      [],
        "weights":       None,
        "messages_sent": [],
        "cum_messages":  [],
    }
    cum = 0

    for t in range(n_iterations):
        # All nodes solve their subproblem using weights from iteration t
        new_weights = {}
        for i in nodes:
            new_weights[i] = _exact_local_solve(data, weights, adjacency, i, alpha)
        weights = new_weights

        # Record losses and communication cost
        history["train_loss"].append(_mean_loss(weights, data, "train"))
        history["val_loss"].append(_mean_loss(weights, data, "val"))
        history["messages_sent"].append(messages_per_iter)
        cum += messages_per_iter
        history["cum_messages"].append(cum)

    history["weights"] = weights
    return history


# ── Algorithm A4: Gradient descent with partial participation ──────────────────

def _rank_schedule(stations: pd.DataFrame, K: int) -> list[list[int]]:
    """
    Deterministic participation schedule for A4.

    Nodes are ranked by latitude, southernmost first (ascending lat).
    In each period of K iterations, 1/K of nodes participate per round
    in round-robin order.

    Parameters
    ----------
    stations : DataFrame with a 'lat' column, indexed 0..N-1
    K        : participation period (1, 2, or 3)

    Returns
    -------
    schedule : list of K lists, where schedule[t % K] gives the
               node indices that are active at iteration t.
    """
    sorted_indices = stations["lat"].sort_values().index.tolist()  # ascending lat
    N = len(sorted_indices)
    schedule = [[] for _ in range(K)]
    for rank, node_i in enumerate(sorted_indices):
        schedule[rank % K].append(node_i)
    return schedule


def run_a4(
    data: dict,
    weights_init: dict,
    adjacency: np.ndarray,
    stations: pd.DataFrame,
    alpha: float = 1.0,
    lr: float = 0.001,
    K: int = 1,
    n_iterations: int = 100,
) -> dict:
    """
    A4: Gradient descent with partial participation.

    Active nodes perform one gradient step (local gradient + neighbour
    penalty gradient). Inactive nodes keep their parameters frozen.

    Participation is deterministic: nodes are ranked by latitude
    (southernmost first) and scheduled in round-robin across K slots.

    Communication cost per iteration: sum of degrees of active nodes
    (each active node sends to all its neighbours).

    Parameters
    ----------
    data            : output of build_local_datasets()
    weights_init    : output of init_weights()
    adjacency       : (N, N) weighted adjacency matrix
    stations        : DataFrame from get_station_index()
    alpha           : GTV regularisation strength
    lr              : gradient descent step size (eta)
    K               : participation period (1 = all nodes active every iter)
    n_iterations    : number of iterations to run

    Returns
    -------
    history : dict with keys
        "train_loss"    : list of mean train MSE per iteration
        "val_loss"      : list of mean val   MSE per iteration
        "weights"       : final weights dict
        "messages_sent" : list of messages sent per iteration
        "cum_messages"  : list of cumulative messages sent
    """
    weights = {i: w.copy() for i, w in weights_init.items()}
    schedule = _rank_schedule(stations, K)

    history = {
        "train_loss":    [],
        "val_loss":      [],
        "weights":       None,
        "messages_sent": [],
        "cum_messages":  [],
    }
    cum = 0

    for t in range(n_iterations):
        active_nodes = schedule[t % K]

        # Compute new parameters for active nodes only (using current weights)
        new_weights = {}
        for i in active_nodes:
            X_i = data[i]["X_train"]
            y_i = data[i]["y_train"]
            grad = (
                local_gradient(X_i, y_i, weights[i])
                + _neighbour_penalty_gradient(weights, adjacency, i, alpha)
            )
            new_weights[i] = weights[i] - lr * grad

        # Apply updates; inactive nodes stay frozen
        for i in active_nodes:
            weights[i] = new_weights[i]

        # Record losses and communication cost
        history["train_loss"].append(_mean_loss(weights, data, "train"))
        history["val_loss"].append(_mean_loss(weights, data, "val"))
        msgs = _count_messages(adjacency, active_nodes)
        history["messages_sent"].append(msgs)
        cum += msgs
        history["cum_messages"].append(cum)

    history["weights"] = weights
    return history
