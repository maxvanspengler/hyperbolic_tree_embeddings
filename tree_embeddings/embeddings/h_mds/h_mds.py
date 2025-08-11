import os

import networkx as nx

import numpy as np
from numpy.linalg import norm

from scipy.sparse.linalg import eigsh

import torch

from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import ManifoldTensor

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sn

from ..evaluation.evaluation import distortion, mean_average_precision
from ..evaluation.visualization import plot_embeddings


def h_mds(
    graph: nx.DiGraph,
    dataset: str,
    graph_name: str,
    embedding_dim: int,
    tau: float,
    root: int = 0,
) -> np.ndarray:
    # Compute metric and apply cosh
    cosh_metric = np.cosh(
        tau * nx.floyd_warshall_numpy(
            graph.to_undirected(), nodelist=list(range(graph.number_of_nodes()))
        )
    )

    # Get normalized eigenvectors belonging to embedding_dim largest eigenvalues of -cosh_metric
    eigenvalues, embeddings = eigsh(
        -cosh_metric, k=embedding_dim, which="LM", maxiter=10000, tol=1e-8
    )
    embeddings = embeddings[:, eigenvalues > 0]
    eigenvalues = eigenvalues[eigenvalues > 0]

    # Scale embeddings with eigenvalues
    embeddings = np.sqrt(eigenvalues) * embeddings

    # Project from hyperboloid to Poincaré ball
    embeddings = embeddings / (1 + np.sqrt(1 + np.square(norm(embeddings, axis=1, keepdims=True))))

    # Compute distortion
    ball = PoincareBall(Curvature(1.0, constraining_strategy=lambda x: x))
    embeddings = torch.tensor(embeddings)
    rel_dist, true_dist = distortion(embeddings, graph, ball, tau)
    rel_dist_mean = (rel_dist.sum() / (rel_dist.size(0) * (rel_dist.size(0) - 1))).item()
    rel_dist_max = rel_dist.max().item()
    map_val = mean_average_precision(
        embeddings=embeddings, graph=graph, ball=ball
    ).item()
    print(f"Mean relative distortion: {rel_dist_mean:.3f}")
    print(f"Maximum relative distortion: {rel_dist_max:.3f}")
    print(f"Worst-case distortion: {true_dist.max().item() / true_dist.clamp_min(1e-16).min().item():.3f}")
    print(f"Mean average precision: {map_val:.3f}")

    # Create output directory
    dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    fig_dir = os.path.join(
        dir,
        "results",
        "h_mds",
        dataset,
        graph_name,
    )
    os.makedirs(fig_dir, exist_ok=True)

    # Reorder nodes according to bfs and create heatmap of relative distortions
    node_order = [root] + [t for _, t in nx.bfs_edges(graph, root)]
    rel_dist_df = pd.DataFrame(
        rel_dist.detach(),
        node_order,
        node_order,
    )
    fig = plt.figure(figsize=(10, 8))
    ax = sn.heatmap(rel_dist_df)
    ax.collections[0].set_clim(0, 1)
    plt.savefig(
        os.path.join(
            fig_dir, f"relative_pairwise_distortions_tau_{tau}_dim_{embedding_dim}_nc_1.png"
        )
    )

    # If there are only 2 dimensions, visualize the embeddings    
    if embedding_dim == 2:
        plot_embeddings(
            hierarchy_embeddings=embeddings,
            hierarchy=graph,
            tau=tau,
            fig_dir=fig_dir,
        )

    return embeddings, rel_dist_mean, rel_dist_max
