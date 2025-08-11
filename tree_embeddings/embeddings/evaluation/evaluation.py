import os

import torch

import networkx as nx

from hypll.manifolds.poincare_ball import PoincareBall


def dist(x: torch.Tensor, y: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    mx2 = (1 - x.square().sum(dim=-1))
    my2 = (1 - y.square().sum(dim=-1))
    xmy2 = (x - y).square().sum(dim=-1)
    return (1 + 2 * xmy2 / (mx2 * my2)).acosh()


def distortion(
    embeddings: torch.Tensor,
    graph: nx.DiGraph | nx.Graph,
    # graph_name: str,
    ball: PoincareBall,
    tau: float,
) -> torch.Tensor:
    # Compute pairwise distances of embeddings
    embedding_dists = dist(embeddings, embeddings[:, None, :], ball.c())

    # Set some stuff up for computing target distances
    undirected_graph = graph.to_undirected()
    number_of_nodes = embeddings.size(0)

    # Compute the target distances as lengths of shortest paths
    target_dists = torch.empty([number_of_nodes, number_of_nodes])
    for dist_tuple in nx.shortest_path_length(undirected_graph, weight="weight"):
        distances_sorted_by_node_id = [d for n, d in sorted(dist_tuple[1].items())]
        target_dists[dist_tuple[0], :] = torch.tensor(distances_sorted_by_node_id)

    # Scale by tau
    target_dists = tau * target_dists

    # Compute and return the relative distortion
    rel_distortion = (embedding_dists - target_dists).abs() / target_dists
    rel_distortion.fill_diagonal_(0.0)
    true_dist = embedding_dists / target_dists
    true_dist.fill_diagonal_(1.0)

    return rel_distortion, true_dist


def mean_average_precision(
    embeddings: torch.Tensor,
    graph: nx.DiGraph | nx.Graph,
    ball: PoincareBall,
) -> torch.Tensor:
    n = len(graph.nodes())

    if graph.is_directed():
        graph = graph.to_undirected()

    # Compute pairwise distances of embeddings
    embedding_dists = dist(embeddings, embeddings[:, None, :], ball.c())
    embedding_dists.fill_diagonal_(float("inf"))
    embedding_dists_neighbours = embedding_dists.clone()

    # Grab indices of neighbourhood nodes for each row
    non_neighbourhood_index = torch.ones_like(embedding_dists, dtype=bool)
    for node in graph:
        neighbourhood = list(graph.neighbors(node))
        non_neighbourhood_index[node, neighbourhood] = False

    # Set all non-neighbouring nodes distances to inf in the cloned distance tensor
    embedding_dists_neighbours[non_neighbourhood_index] = float("inf")

    # Argsort the rows for both distance tensors
    argsorted_all = embedding_dists.argsort(stable=True)
    argsorted_neighbourhood = embedding_dists_neighbours.argsort(stable=True)

    # Rank each column node according to its proximity to the row node
    rank_all = torch.empty_like(embedding_dists)
    rank_neighbourhood = rank_all.clone()
    src_vals = torch.arange(1, n + 1, dtype=rank_all.dtype).expand(n, n)
    rank_all.scatter_(dim=1, index=argsorted_all, src=src_vals)
    rank_neighbourhood.scatter_(dim=1, index=argsorted_neighbourhood, src=src_vals)

    # Check fraction of rank within neighbourhood to rank overall
    prec = rank_neighbourhood / rank_all

    # Scale the values by the degree of the row node so we can compute the mean
    weighted = prec / (~non_neighbourhood_index).sum(dim=1, keepdim=True)

    # Sum and divide by the number of nodes to obtain the mean average precision
    return weighted[~non_neighbourhood_index].sum() / n
