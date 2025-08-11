import os

import networkx as nx

import torch

from tree_embeddings.embeddings.h_mds import h_mds
from tree_embeddings.trees.file_utils import load_hierarchy


def run_h_mds_embeddings(
    dataset: str,
    hierarchy_name: str,
    root: int = 0,
    tau: float = 1.0,
    embedding_dim: int = 20,
):
    # Load hierarchy and turn to directed if necessary
    graph = load_hierarchy(dataset=dataset, hierarchy_name=hierarchy_name)
    if not graph.is_directed():
        graph = nx.bfs_tree(graph, root)
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")

    embeddings, _, _ = h_mds(
        graph=graph,
        dataset=dataset,
        graph_name=hierarchy_name,
        embedding_dim=20,
        tau=tau,
        root=root,
    )

    res_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "h_mds",
        dataset,
        hierarchy_name,
    )
    os.makedirs(res_dir, exist_ok=True)
    torch.save(
        embeddings,
        os.path.join(res_dir, "embeddings.pt"),
    )

