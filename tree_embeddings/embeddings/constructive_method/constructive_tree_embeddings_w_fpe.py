from math import exp
import os
import time

import mpmath

import torch

from fpe_torch import FPETensor

import networkx as nx

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sn

from hypll.manifolds.poincare_ball import Curvature, PoincareBall

from ..evaluation.expanded_evaluation import distortion, mean_average_precision
from .isometries_expanded import (
    get_circle_inversion_mapping_x_to_origin,
    get_householder_reflection_mapping_x_to_y,
)
from .sphere_points import get_sphere_points
from ..evaluation.visualization import plot_embeddings


def embed_tree_w_fpe(
    hierarchy: nx.DiGraph,
    dataset: str,
    hierarchy_name: str,
    # graph: nx.Graph,
    embedding_dim: int,
    tau: float,
    nc: int,
    curvature: float = 1.0,
    root: int = 0,
    gen_type: str = "optim",
    dtype: torch.dtype = torch.float64,
) -> tuple[FPETensor, float, float]:
    # Grab number of non-leaf nodes, so we can properly give progress updates later
    num_non_leaf_nodes = len([n for n in hierarchy.nodes() if hierarchy.out_degree(n) != 0])

    # Define Poincare ball
    ball = PoincareBall(
        c=Curvature(value=curvature, constraining_strategy=lambda x: x)
    )

    mpmath.mp.dps = nc * 20
    tau = mpmath.mp.mpf(str(tau))
    rad = (mpmath.exp(tau) - 1) / (mpmath.exp(tau) + 1)
    rad = FPETensor.from_mpmath(mp_numbers=[rad], terms=nc, dtype=dtype)

    # Initialize embeddings
    hierarchy_embeddings = FPETensor.from_tensor(
        torch.zeros(hierarchy.number_of_nodes(), embedding_dim, nc, dtype=dtype)
    )

    # Place children of root on (n-1)-sphere with hyperbolic radius tau around the origin
    if nx.is_weighted(hierarchy):
        children_edges = list(hierarchy.out_edges(root, data=True))
        children = [e[1] for e in children_edges]
        scaled_weights = [tau * e[2]["weight"] for e in children_edges]

        rescaling = FPETensor.from_mpmath(
            mp_numbers=[(mpmath.exp(sw) - 1) / (mpmath.exp(sw) + 1) for sw in scaled_weights],
            terms=nc,
            dtype=dtype,
        )
        
        hierarchy_embeddings[children] = get_sphere_points(
            n=len(children),
            d=embedding_dim,
            gen_type=gen_type,
        ).to(dtype)
        hierarchy_embeddings[children] = (
            rescaling.unsqueeze(1) * hierarchy_embeddings[children]
            / hierarchy_embeddings[children].norm(dim=-1, keepdim=True)
        )
    else:
        children = list(hierarchy.successors(root))
        hierarchy_embeddings[children] = get_sphere_points(
            n=len(children),
            d=embedding_dim,
            gen_type=gen_type,
        ).to(dtype)
        hierarchy_embeddings[children] = (
            rad * hierarchy_embeddings[children]
            / hierarchy_embeddings[children].norm(dim=-1, keepdim=True)
        )

    # Setup generator over the levels of the hierarchy and loop
    count = 1
    start = time.time()
    levels = nx.bfs_layers(hierarchy, root)
    next(levels)
    for level in levels:
        for node in level:
            it_start = time.time()

            # Grab node embedding, parent embedding and children
            if nx.is_weighted(hierarchy):
                children_edges = list(hierarchy.out_edges(node, data=True))
                children = [e[1] for e in children_edges]
                scaled_weights = (
                    [tau * list(hierarchy.in_edges(node, data=True))[0][2]["weight"]]
                    + [tau * e[2]["weight"] for e in children_edges]
                )

                rescaling = FPETensor.from_mpmath(
                    mp_numbers=[(mpmath.exp(sw) - 1) / (mpmath.exp(sw) + 1) for sw in scaled_weights],
                    terms=nc,
                    dtype=dtype,
                )
            else:
                children = list(hierarchy.successors(node))
                rescaling = rad * torch.ones(len(children) + 1)

            if not children:
                continue

            # Grab node embedding and parent node embedding
            node_embedding = hierarchy_embeddings[node]
            parent = next(hierarchy.predecessors(node))
            parent_embedding = hierarchy_embeddings[parent]

            # Create circle inversion mapping node embedding to origin and reflect parent embedding
            circle_inversion = get_circle_inversion_mapping_x_to_origin(
                x=node_embedding, curv=ball.c()
            )
            reflected_parent_embedding = circle_inversion(x=parent_embedding)

            # Generate points on sphere and reflect so first point aligns with reflected parent
            sphere_points = FPETensor.from_tensor(
                torch.zeros((1 + len(children), embedding_dim, nc), dtype=dtype)
            )
            sphere_points[:] = get_sphere_points(
                n=1 + len(children), d=embedding_dim, gen_type=gen_type
            )
            sphere_points = (
                rescaling.unsqueeze(1) * sphere_points
                / sphere_points.norm(dim=-1, keepdim=True)
            )
            hh_reflection = get_householder_reflection_mapping_x_to_y(
                x=sphere_points[0],
                y=reflected_parent_embedding,
                equal_norm=True,
            )
            aligned_sphere_points = hh_reflection(x=sphere_points)

            # Reflect back and insert embedded children into embeddings tensor
            new_embeddings = circle_inversion(x=aligned_sphere_points)
            hierarchy_embeddings[children] = new_embeddings[1:]

            count += 1

            print(
                f"Progress: {count} / {num_non_leaf_nodes},    "
                f"current node: {node},    "
                f"elapsed time: {time.time() - start:.2f},    "
                f"iteration time: {time.time() - it_start:2f}."
            )

    tau = float(tau)

    # Compute relative distortions and print some results
    n = hierarchy.number_of_nodes()
    rel_dist, true_dist = distortion(
        embeddings=hierarchy_embeddings[:n], graph=hierarchy, ball=ball, tau=tau
    )
    rel_dist_mean = (
        rel_dist.sum() / (rel_dist.size(0) * (rel_dist.size(0) - 1))
    ).tensor[..., 0].item()
    rel_dist_max = rel_dist.tensor[..., 0].max().item()
    worst_case_dist = true_dist.tensor[..., 0].max().item() / true_dist.tensor[..., 0].min().item()
    map_val = mean_average_precision(
        embeddings=hierarchy_embeddings[:n], graph=hierarchy, ball=ball
    ).item()
    print(f"Tau: {tau:.2f}, embedding dimension: {embedding_dim}")
    print(f"Mean relative distortion: {rel_dist_mean:.3f}")
    print(f"Maximum relative distortion: {rel_dist_max:.3f}")
    print(f"Worst-case distortion: {worst_case_dist:.3f}")
    print(f"Mean average precision: {map_val:.3f}")

    # Set output directory
    dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    fig_dir = os.path.join(
        dir,
        "results",
        "constructive_method",
        dataset,
        hierarchy_name,
    )
    os.makedirs(fig_dir, exist_ok=True)

    # Reorder nodes through a bfs and create heatmap of result
    node_order = [0] + [t for _, t in nx.bfs_edges(hierarchy, 0)]
    node_order = [node for node in node_order if node < n]
    rel_dist_df = pd.DataFrame(
        rel_dist.tensor[..., 0].detach(),
        node_order,
        node_order,
    )
    fig = plt.figure(figsize=(10, 8))
    ax = sn.heatmap(rel_dist_df)
    ax.collections[0].set_clim(0, 1)
    plt.savefig(
        os.path.join(
            fig_dir, f"relative_pairwise_distortions_tau_{tau}_dim_{embedding_dim}_nc_{nc}.png"
        )
    )
    
    # If the embedding dim is 2, plot the embeddings
    if embedding_dim == 2:
        plot_embeddings(
            hierarchy_embeddings=hierarchy_embeddings.tensor[..., 0],
            hierarchy=hierarchy,
            tau=tau,
            fig_dir=fig_dir,
        )

    return hierarchy_embeddings, rel_dist_mean, rel_dist_max
