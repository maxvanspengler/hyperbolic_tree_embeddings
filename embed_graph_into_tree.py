import os

import networkx as nx

import numpy as np

from tree_embeddings.trees.graph_to_tree.convert_graph_to_tree import embed_graph_into_tree
from tree_embeddings.trees.graph_to_tree.contract_trivial_edges import contract_tree
from tree_embeddings.trees.file_utils import load_hierarchy, store_hierarchy


if __name__ == "__main__":
    # Load hierarchy and compute metric
    dataset = "grqc"
    graph_name = "grqc"
    method = "repo"
    graph = load_hierarchy(dataset=dataset, hierarchy_name=graph_name)
    metric = nx.floyd_warshall_numpy(graph, nodelist=list(range(graph.number_of_nodes())))
    n = len(graph.nodes())

    # Embed the graph into a tree starting from each node and track which root gave best results
    min_rel, min_root = float("inf"), -1
    for r in range(n):
        tree, rel = embed_graph_into_tree(
            dataset=dataset,
            graph_name=graph_name,
            graph=graph,
            metric=metric,
            root=r,
            method=method,
        )
        if rel < min_rel:
            min_rel, min_root = rel, r

    # Report best result and recompute the corresponding tree
    print(min_rel, min_root)
    tree, rel = embed_graph_into_tree(
        dataset=dataset,
        graph_name=graph_name,
        graph=graph,
        metric=metric,
        root=min_root,
        method=method,
    )

    # Contract the resulting tree (removes trivial edges)
    contracted_tree = contract_tree(tree=tree, n=n)
    contracted_tree = nx.convert_node_labels_to_integers(contracted_tree, ordering="sorted")

    # Compute performance metrics
    np.fill_diagonal(metric, 1)
    shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(contracted_tree))
    tree_metric = np.array([
        list(v2 for _, v2 in sorted(v.items())) for _, v in sorted(shortest_path_lengths.items())
    ])[:n, :n]
    rel_distortion = (tree_metric - metric) / metric
    true_distortion = (
        (tree_metric / metric - 1e10 * np.eye(n)).max()
        / (tree_metric / metric + 1e10 * np.eye(n)).min()
    )
    np.fill_diagonal(rel_distortion, 0)

    # Report metrics
    tree_metric = nx.floyd_warshall_numpy(graph)
    longest_path_length = tree_metric.max()
    print("Maximum absolute relative distortion:", np.abs(rel_distortion).max())
    print("Mean distortion:", rel_distortion.sum() / (n * (n - 1)))
    print("Worst case distortion:", true_distortion)
    print("Max path length of tree:", longest_path_length)

    # Store contracted tree and metrics
    dirname = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "tree_embeddings",
        "data",
        "hierarchies",
        dataset,
    )

    with open(os.path.join(dirname, f"{graph_name}_from_{min_root}.txt"), "w") as f:
        f.write(f"Maximum absolute relative distortion: {np.abs(rel_distortion).max()}\n")
        f.write(f"Mean distortion: {rel_distortion.sum() / (n * (n - 1))}\n")
        f.write(f"Worst case distortion: {true_distortion}\n")
        f.write(f"Max path length of tree: {longest_path_length}")

    store_hierarchy(
        hierarchy=contracted_tree,
        destination=os.path.join(
            dirname,
            f"{graph_name}_contracted_tree_from_{min_root}.json",
        ),
    )
