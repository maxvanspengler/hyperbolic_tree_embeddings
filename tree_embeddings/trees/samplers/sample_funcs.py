import random

import networkx as nx

import torch


def edge_sample_uniform(
    negative_nodes: list[int],
    sample_size: int,
    **kwargs
) -> list[int]:
    sample = random.sample(negative_nodes, sample_size)
    return sample


def edge_sample_prioritize_siblings(
    negative_nodes: list[int],
    sample_size: int,
    node: int,
    hierarchy: nx.DiGraph,
) -> list[int]:
    # Only force sibling nodes when the node is a leaf node
    if hierarchy.out_degree(node) > 0:
        return edge_sample_uniform(
            negative_nodes=negative_nodes, sample_size=sample_size
        )
    else:
        parent = next(hierarchy.predecessors(node))
        siblings = list(s for s in hierarchy.successors(parent) if s != node)

        max_sibling_sample_size = sample_size // 2

        if len(siblings) < max_sibling_sample_size:
            sample = siblings + edge_sample_uniform(
                negative_nodes=negative_nodes,
                sample_size=sample_size - len(siblings)
            )
        else:
            sample = random.sample(siblings, max_sibling_sample_size)
            sample += random.sample(negative_nodes, sample_size - max_sibling_sample_size)
        return sample


def edge_corrupt_source_sampler(
    rel: tuple[int, int], hierarchy: nx.DiGraph, num_negs: int, sample_strat: callable
) -> tuple[torch.Tensor, torch.Tensor]:
    negative_source_nodes = list(
        hierarchy.nodes() - nx.ancestors(hierarchy, rel[1]) - {rel[1]}
    )
    if len(negative_source_nodes) < num_negs:
        negative_source_nodes = list(
            hierarchy.nodes() - hierarchy.predecessors(rel[1]) - {rel[1]}
        )
    negative_source_nodes = sample_strat(
        negative_nodes=negative_source_nodes,
        sample_size=num_negs,
        node=rel[1],
        hierarchy=hierarchy,
    )
    inputs = torch.tensor([rel] + [[neg, rel[1]] for neg in negative_source_nodes])

    edge_label_targets = torch.cat(
        tensors=[torch.ones(1).bool(), torch.zeros(num_negs).bool()],
        dim=0,
    )

    return inputs, edge_label_targets


def edge_corrupt_target_sampler(
    rel: tuple[int, int], hierarchy: nx.DiGraph, num_negs: int, sample_strat: callable
) -> tuple[torch.Tensor, torch.Tensor]:
    negative_target_nodes = list(
        hierarchy.nodes() - nx.descendants(hierarchy, rel[0]) - {rel[0]}
    )
    if len(negative_target_nodes) < num_negs:
        negative_target_nodes = list(
            hierarchy.nodes() - hierarchy.successors(rel[0]) - {rel[0]}
        )
    negative_target_nodes = sample_strat(
        negative_nodes=negative_target_nodes,
        sample_size=num_negs,
        node=rel[0],
        hierarchy=hierarchy,
    )
    inputs = torch.tensor([rel] + [[rel[0], neg] for neg in negative_target_nodes])

    edge_label_targets = torch.cat(
        tensors=[torch.ones(1).bool(), torch.zeros(num_negs).bool()],
        dim=0,
    )

    return inputs, edge_label_targets


def edge_corrupt_both_sampler(
    rel: tuple[int, int], hierarchy: nx.DiGraph, num_negs: int, sample_strat: callable
) -> tuple[torch.Tensor, torch.Tensor]:
    num_neg_source = num_negs // 2 + num_negs % 2
    num_neg_target = num_negs // 2

    negative_source_nodes = list(
        hierarchy.nodes() - nx.ancestors(hierarchy, rel[1]) - {rel[1]}
    )
    if len(negative_source_nodes) < num_neg_source:
        negative_source_nodes = list(
            hierarchy.nodes() - hierarchy.predecessors(rel[1]) - {rel[1]}
        )
    negative_source_nodes = sample_strat(
        negative_nodes=negative_source_nodes,
        sample_size=num_neg_source,
        node=rel[1],
        hierarchy=hierarchy,
    )

    inputs = torch.tensor(
        [rel] + [[neg, rel[1]] for neg in negative_source_nodes]
    )

    negative_target_nodes = list(
        hierarchy.nodes() - nx.descendants(hierarchy, rel[0]) - {rel[0]}
    )
    if len(negative_target_nodes) < num_neg_target:
        negative_target_nodes = list(
            hierarchy.nodes() - hierarchy.successors(rel[0]) - {rel[0]}
        )
    negative_target_nodes = sample_strat(
        negative_nodes=negative_target_nodes,
        sample_size=num_neg_target,
        node=rel[0],
        hierarchy=hierarchy,
    )

    inputs = torch.cat(
        tensors=(
            inputs,
            torch.tensor([[rel[0], neg] for neg in negative_target_nodes])
        ),
        dim=0,
    )

    edge_label_targets = torch.cat(
        tensors=[torch.ones(1).bool(), torch.zeros(num_negs).bool()],
        dim=0,
    )

    return inputs, edge_label_targets


def dist_sample_shortest_path(
    edges: torch.Tensor, dist_matrix: torch.Tensor
) -> torch.Tensor:
    dist_targets = dist_matrix[edges[:, 0], edges[:, 1]]
    return dist_targets
