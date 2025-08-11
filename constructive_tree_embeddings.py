from itertools import product
import os

import networkx as nx

import pandas as pd

import torch

from tree_embeddings.embeddings.constructive_method import constructively_embed_tree
from tree_embeddings.trees.file_utils import load_hierarchy


def run_constructive_tree_embeddings(
    dataset: str,
    hierarchy_name: str,
    root: int = 0,
    gen_type: str = "optim",
    tau: float = 1.0,
    embedding_dim: int = 20,
    nc: int = 1,
    curvature: float = 1.0,
    dtype: torch.dtype = torch.float64,
):
    # Load hierarchy and turn to directed if necessary
    hierarchy = load_hierarchy(dataset=dataset, hierarchy_name=hierarchy_name)
    if not nx.is_directed(hierarchy):
        # Store edge weights, turn into directed graph and reassign weights
        edge_data = {
            (source, target): data
            for source, target, data in nx.DiGraph(hierarchy).edges(data=True)
        }
        hierarchy = nx.bfs_tree(hierarchy, root)
        nx.set_edge_attributes(hierarchy, edge_data)

    embeddings, rel_dist_mean, rel_dist_max = constructively_embed_tree(
        hierarchy=hierarchy,
        dataset=dataset,
        hierarchy_name=hierarchy_name,
        embedding_dim=embedding_dim,
        tau=tau,
        nc=nc,
        curvature=curvature,
        root=root,
        gen_type=gen_type,
        dtype=dtype,
    )

    res_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "constructive_method",
        dataset,
        hierarchy_name,
    )
    os.makedirs(res_dir, exist_ok=True)
    torch.save(
        embeddings,
        os.path.join(res_dir, "embeddings.pt"),
    )
