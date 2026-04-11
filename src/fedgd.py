"""Synchronous Federated Gradient Descent (FedGD) for GTVMin.

Solves the Graph Total Variation Minimisation objective:

    min_{w^(1),...,w^(N)}  Σ_i L_i(w^(i))
                         + α Σ_{(i,j)∈E} A_{ij} ||w^(i) - w^(j)||²

where L_i(w) = (1/n_i) ||X_i w - y_i||² is the per-node MSE loss.

The FedGD update rule is synchronous (all nodes update simultaneously):

    w^(i) ← w^(i) − η [ ∇L_i(w^(i)) + 2α Σ_{j∈N(i)} A_{ij} (w^(i) − w^(j)) ]

Communication cost: 2|E| messages per iteration (each undirected edge
carries one message in each direction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np

from data import NodeDataset
from graph import GraphData

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FedGDConfig:
    """Hyper-parameters for a single FedGD run.

    Attributes
    ----------
    alpha:
        Graph regularisation strength.  alpha=0 reduces to independent
        local training (the baseline).
    lr:
        Gradient descent step size (learning rate).
    n_iters:
        Number of synchronous FedGD iterations to run.
    """

    alpha: float = 1.0
    lr: float = 0.01
    n_iters: int = 500


class FedGDHistory(TypedDict):
    """Recorded metrics from a FedGD run.

    Attributes
    ----------
    train_loss:
        Mean MSE across all nodes on the training set, recorded after each
        iteration.  Length = n_iters.
    val_loss:
        Mean MSE on the validation set, recorded after each iteration.
        Length = n_iters.
    weights:
        Final per-node weight vectors after all iterations.
    messages_sent:
        Number of messages exchanged in each iteration.
        For synchronous FedGD: 2 * |E| every iteration.
    cum_messages:
        Cumulative sum of messages_sent.  Length = n_iters.
    test_loss:
        Mean MSE on the held-out test set using the final weights.
    alpha:
        The regularisation strength used in this run.
    """

    train_loss: list[float]
    val_loss: list[float]
    weights: dict[str, np.ndarray]
    messages_sent: list[int]
    cum_messages: list[int]
    test_loss: float
    alpha: float


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _mse(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Compute MSE = (1/n) ||Xw - y||².

    Parameters
    ----------
    X: shape (n, d)
    y: shape (n,)
    w: shape (d,)
    """
    residuals = X @ w - y
    return float(np.dot(residuals, residuals) / len(y))


def _local_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Gradient of MSE w.r.t. w: (2/n) Xᵀ(Xw − y).

    Parameters
    ----------
    X: shape (n, d)
    y: shape (n,)
    w: shape (d,)

    Returns
    -------
    Gradient vector of shape (d,).
    """
    n = len(y)
    return (2.0 / n) * (X.T @ (X @ w - y))


def _penalty_gradient(
    weights: dict[str, np.ndarray],
    node: str,
    graph: GraphData,
    node_order: list[str],
    alpha: float,
) -> np.ndarray:
    """Graph TV regularisation gradient for node i.

    Computes: 2α Σ_{j∈N(i)} A_{ij} (w_i − w_j)

    Parameters
    ----------
    weights:
        Current weight dict (uses weights *before* the step for synchrony).
    node:
        Station name of node i.
    graph:
        GraphData with adjacency matrix.
    node_order:
        Ordered list of station names (defines row/column indexing).
    alpha:
        Regularisation strength.

    Returns
    -------
    Penalty gradient vector of shape (d,).
    """
    i = node_order.index(node)
    w_i = weights[node]
    grad = np.zeros_like(w_i)

    for j, node_j in enumerate(node_order):
        a_ij = graph.adjacency[i, j]
        if a_ij > 0:
            grad += a_ij * (w_i - weights[node_j])

    return 2.0 * alpha * grad


def _mean_mse(
    weights: dict[str, np.ndarray],
    datasets: dict[str, NodeDataset],
    split: Literal["train", "val", "test"],
) -> float:
    """Compute the mean MSE across all nodes for the given split.

    Parameters
    ----------
    weights:
        Per-node weight vectors.
    datasets:
        Per-node NodeDataset dicts.
    split:
        One of ``"train"``, ``"val"``, or ``"test"``.

    Returns
    -------
    Mean MSE (scalar).
    """
    x_key = f"X_{split}"
    y_key = f"y_{split}"
    losses = [
        _mse(datasets[node][x_key], datasets[node][y_key], weights[node])  # type: ignore[literal-required]
        for node in weights
        if node in datasets
    ]
    return float(np.mean(losses))


def _fedgd_step(
    weights: dict[str, np.ndarray],
    datasets: dict[str, NodeDataset],
    graph: GraphData,
    node_order: list[str],
    alpha: float,
    lr: float,
) -> dict[str, np.ndarray]:
    """One synchronous FedGD step.

    All nodes read the *current* weights (before this step) and compute
    their update independently, ensuring synchronous updates.

    Parameters
    ----------
    weights:
        Per-node weights at the start of this step.
    datasets:
        Per-node NodeDataset dicts.
    graph:
        GraphData for penalty gradient computation.
    node_order:
        Consistent ordering of node names.
    alpha:
        Regularisation strength.
    lr:
        Learning rate.

    Returns
    -------
    New weights dict (does not mutate the input).
    """
    new_weights: dict[str, np.ndarray] = {}
    for node in node_order:
        if node not in datasets:
            new_weights[node] = weights[node].copy()
            continue

        X = datasets[node]["X_train"]
        y = datasets[node]["y_train"]
        w = weights[node]

        grad_local = _local_gradient(X, y, w)
        grad_penalty = _penalty_gradient(weights, node, graph, node_order, alpha)

        new_weights[node] = w - lr * (grad_local + grad_penalty)

    return new_weights


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_fedgd(
    datasets: dict[str, NodeDataset],
    graph: GraphData,
    config: FedGDConfig,
    init_w: dict[str, np.ndarray],
) -> FedGDHistory:
    """Run synchronous FedGD for ``config.n_iters`` iterations.

    Parameters
    ----------
    datasets:
        Per-node NodeDataset dicts.
    graph:
        GraphData describing the FL network topology.
    config:
        FedGDConfig with alpha, lr, and n_iters.
    init_w:
        Initial per-node weight vectors (will not be mutated).

    Returns
    -------
    FedGDHistory with full convergence curves and final test loss.

    Notes
    -----
    Communication cost per iteration is 2 * |E| messages because every
    undirected edge is traversed in both directions (node i sends w_i to j
    and node j sends w_j to i each round).
    """
    node_order = sorted(init_w.keys())
    n_edges = len(graph.edge_index)
    messages_per_iter: int = 2 * n_edges

    weights = {node: w.copy() for node, w in init_w.items()}

    train_loss: list[float] = []
    val_loss: list[float] = []
    messages_sent: list[int] = []

    for it in range(config.n_iters):
        weights = _fedgd_step(
            weights, datasets, graph, node_order, config.alpha, config.lr
        )

        train_loss.append(_mean_mse(weights, datasets, "train"))
        val_loss.append(_mean_mse(weights, datasets, "val"))
        messages_sent.append(messages_per_iter)

        if (it + 1) % 50 == 0:
            print(
                f"  iter {it + 1:4d}/{config.n_iters}  "
                f"train={train_loss[-1]:.4f}  val={val_loss[-1]:.4f}"
            )

    test_loss = _mean_mse(weights, datasets, "test")
    cum_messages = list(np.cumsum(messages_sent))

    return FedGDHistory(
        train_loss=train_loss,
        val_loss=val_loss,
        weights=weights,
        messages_sent=messages_sent,
        cum_messages=cum_messages,
        test_loss=test_loss,
        alpha=config.alpha,
    )
