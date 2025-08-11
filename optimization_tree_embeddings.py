from itertools import product
import os
import random
import time

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sn

import networkx as nx

import torch
from torch.utils.data import DataLoader

from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.optim import RiemannianSGD

from tree_embeddings.trees.file_utils import load_hierarchy
from tree_embeddings.trees.dataset import TreeEmbeddingDataset
from tree_embeddings.embeddings.distortion import DistortionEmbedding
from tree_embeddings.embeddings.hyperbolic_entailment_cones import EntailmentConeEmbedding
from tree_embeddings.embeddings.poincare_embeddings import PoincareEmbedding
from tree_embeddings.embeddings.evaluation.evaluation import distortion, mean_average_precision
from tree_embeddings.embeddings.evaluation.visualization import plot_embeddings


torch.manual_seed(42)
random.seed(42)


def run_optimization_tree_embeddings(
    dataset_name: str,
    hierarchy_name: str,
    root: int = 0,
    optimization_method: str = "distortion",
    tau: float = 1.0,
    embedding_dim: int = 20,
    curvature: float = 1.0,
    dtype: torch.dtype = torch.float64,
    epochs: int = 200,
    lr: float = 1.0,
):
    # Load hierarchy and turn to directed if necessary
    hierarchy = load_hierarchy(dataset=dataset_name, hierarchy_name=hierarchy_name)
    if not nx.is_directed(hierarchy):
        # Store edge weights, turn into directed graph and reassign weights
        edge_data = {
            (source, target): data
            for source, target, data in nx.DiGraph(hierarchy).edges(data=True)
        }
        hierarchy = nx.bfs_tree(hierarchy, root)
        nx.set_edge_attributes(hierarchy, edge_data)
    hierarchy = nx.convert_node_labels_to_integers(hierarchy, ordering="sorted")

    # Wrap hierarchy into dataloader
    dataset = TreeEmbeddingDataset(
        hierarchy=hierarchy,
        num_negs=10,
        edge_sample_from="both",
        edge_sample_strat="uniform",
        dist_sample_strat="shortest_path",
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=3500,
        shuffle=True,
    )

    # Initialize embedding model
    ball = PoincareBall(c=Curvature(curvature, constraining_strategy=lambda x: x))
    if optimization_method == "hyperbolic_entailment_cones":
        model = EntailmentConeEmbedding(
            num_embeddings=hierarchy.number_of_nodes(),
            embedding_dim=embedding_dim,
            ball=ball,
        )
    if optimization_method == "poincare_embeddings":
        model = PoincareEmbedding(
            num_embeddings=hierarchy.number_of_nodes(),
            embedding_dim=embedding_dim,
            ball=ball,
        )
    if optimization_method == "distortion":
        model = DistortionEmbedding(
            num_embeddings=hierarchy.number_of_nodes(),
            embedding_dim=embedding_dim,
            ball=ball,
            tau=tau,
        )

    # Initialize optimizer
    optimizer = RiemannianSGD(
        params=model.parameters(),
        lr=lr,
        momentum=0,
        weight_decay=0,
    )

    # Train the model
    start = time.time()
    losses, _ = model.train(
        dataloader=dataloader,
        epochs=epochs,
        optimizer=optimizer,
        burn_in_epochs=20,
        burn_in_lr_mult=0.1 * lr,
        store_losses=True,
    )
    print(f"Elapsed training time: {time.time() - start:.3f} seconds")

    hierarchy_embeddings = model.weight.tensor.detach()

    # Compute the relative distortions and print some results
    n = hierarchy.number_of_nodes()
    rel_dist, true_dist = distortion(
        embeddings=hierarchy_embeddings[:n], graph=hierarchy, ball=ball, tau=tau
    )
    rel_dist_mean = (rel_dist.sum() / (rel_dist.size(0) * (rel_dist.size(0) - 1))).item()
    rel_dist_max = rel_dist.max().item()
    worst_case_dist = true_dist.max().item() / true_dist.clamp_min(1e-15).min().item()
    map_val = mean_average_precision(
        embeddings=hierarchy_embeddings[:n], graph=hierarchy, ball=ball
    ).item()
    print(f"Dataset: {dataset_name}, hierarchy: {hierarchy_name}, graph: {hierarchy_name}")
    print(f"Embedding type: {optimization_method}")
    print(f"Tau: {tau:.2f}, embedding dimension: {embedding_dim}")
    print(f"Mean relative distortion: {rel_dist_mean:.3f}")
    print(f"Maximum relative distortion: {rel_dist_max:.3f}")
    print(f"Worst-case distortion: {worst_case_dist:.3f}")
    print(f"Mean average precision: {map_val:.3f}")

    # Create output directory
    dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(
        dir,
        "results",
        optimization_method,
        dataset_name,
        hierarchy_name,
    )
    os.makedirs(fig_dir, exist_ok=True)

    # Reorder nodes according to bfs and create heatmap of relative distortions
    node_order = [root] + [t for _, t in nx.bfs_edges(hierarchy, root)]
    node_order = [node for node in node_order if node < n]
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
            fig_dir,
            f"relative_pairwise_distortions_tau_{tau}_dim_{embedding_dim}_nc_1.png",
        )
    )
    plt.close()

    # If there are only 2 dimensions, visualize the embeddings    
    if embedding_dim == 2:
        plot_embeddings(
            hierarchy_embeddings=hierarchy_embeddings,
            hierarchy=hierarchy,
            tau=tau,
            fig_dir=fig_dir,
        )

    torch.save(
        hierarchy_embeddings,
        os.path.join(fig_dir, "embeddings.pt"),
    )
