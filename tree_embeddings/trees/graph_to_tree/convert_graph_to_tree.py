import os

import networkx as nx

import numpy as np

from ..file_utils import store_hierarchy


def gromov_product(metric: np.ndarray, x: int, y: int, z: int):
    dxy = metric[x, y]
    dxz = metric[x, z]
    dyz = metric[y, z]
    return 1 / 2 * (dxz + dyz - dxy)


def embed_graph_into_tree(
    dataset: str,
    graph_name: str,
    graph: nx.Graph,
    metric: np.ndarray,
    root: int = 0,
    method: str = "repo",
) -> tuple[nx.Graph, float]:
    # Precompute gromov products
    gps = 1 / 2 * (metric[root][None, :] + metric[root][:, None] - metric)
    np.fill_diagonal(gps, 0)

    # Find order in which to remove nodes
    n = len(graph.nodes())
    still_in = np.ones(n)
    ind_p, ind_q = np.unravel_index(np.argsort(-gps, axis=None), gps.shape)

    # Remove nodes
    cur_pos = 0
    removed = 0
    qs, ps = np.zeros(n - 2), np.zeros(n - 2)
    while removed < n - 2:
        p, q = ind_p[cur_pos], ind_q[cur_pos]
        if p != q and still_in[p] and still_in[q]:
            if metric[p, root] > metric[q, root]:
                p, q = q, p
            qs[removed], ps[removed] = q, p
            removed += 1
            still_in[q] = 0
        cur_pos += 1

    # Create tree and add first nodes
    tree = nx.Graph()
    still_in[root] = 0
    last = int(np.where(still_in)[0][0])
    tree.add_edge(root, last, weight=metric[root, last])

    # Add the rest of the nodes, including the Steiner nodes
    new_node = n
    removed -= 1
    while removed != -1:
        # Method corresponding to the actual paper
        if method == "paper":
            p, q = int(ps[removed]), int(qs[removed])
            tree.add_edge(new_node, p, weight=float(gromov_product(metric, q, root, p)))
            tree.add_edge(new_node, q, weight=float(gromov_product(metric, p, root, q)))
            removed -= 1
            new_node += 1

        # Method corresponding to the one found in the repository
        elif method == "repo":
            p, q = int(ps[removed]), int(qs[removed])
            qrp = float(gromov_product(metric, q, root, p))
            prq = float(gromov_product(metric, p, root, q))
            for node in tree.neighbors(p):
                new_weight = max(0, tree[p][node]["weight"] - qrp)
                tree.add_edge(new_node, node, weight=new_weight)
            tree.remove_node(p)
            tree.add_edge(new_node, p, weight=qrp)
            tree.add_edge(new_node, q, weight=prq)
            removed -= 1
            new_node += 1

    # Compute performance metrics
    metricp1 = metric + np.eye(n)
    # np.fill_diagonal(metric, 1)
    shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(tree))
    tree_metric = np.array([
        list(v2 for k2, v2 in sorted(v.items())) for k, v in sorted(shortest_path_lengths.items())
    ])[:n, :n]
    rel_distortion = (tree_metric - metric) / (metricp1)
    true_distortion = (
        (tree_metric / metricp1 - 1e10 * np.eye(n)).max()
        / (tree_metric / metricp1 + 1e10 * np.eye(n)).min()
    )
    np.fill_diagonal(rel_distortion, 0)

    # Report numbers
    print("Starting from root:", root)
    print("Maximum absolute relative distortion:", np.abs(rel_distortion).max())
    print("Mean distortion:", rel_distortion.sum() / (n * (n - 1)))
    print("Worst case distortion:", true_distortion)

    # Store the output tree
    json_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
        "data",
        "hierarchies",
        dataset,
    )
    json_file = os.path.join(json_dir, f"{graph_name}_tree.json")
    store_hierarchy(tree, json_file)

    return tree, (rel_distortion.sum() / (n * (n - 1)))
