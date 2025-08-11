from functools import partial
from typing import Literal, Optional

import networkx as nx

import torch

from .sample_funcs import (
    edge_corrupt_both_sampler,
    edge_corrupt_source_sampler,
    edge_corrupt_target_sampler,
    edge_sample_uniform,
    edge_sample_prioritize_siblings,
    dist_sample_shortest_path,
)


class EdgeSampler:
    edge_sample_from_dict = {
        "both": edge_corrupt_both_sampler,
        "source": edge_corrupt_source_sampler,
        "target": edge_corrupt_target_sampler,
    }

    edge_sample_strat_dict = {
        "uniform": edge_sample_uniform,
        "siblings": edge_sample_prioritize_siblings,
    }

    dist_sample_strat_dict = {
        "shortest_path": dist_sample_shortest_path,
    }

    def __init__(
        self,
        hierarchy: nx.DiGraph,
        num_negs: int,
        edge_sample_from: Literal["both", "source", "target"] = "both",
        edge_sample_strat: Literal["uniform", "siblings"] = "uniform",
        dist_sample_strat: Optional[Literal["shortest_path"]] = None,
    ) -> None:
        self.hierarchy = hierarchy
        self.num_negs = num_negs
        self.edge_sample_from = edge_sample_from
        self.edge_sample_strat = edge_sample_strat
        self.dist_sample_strat = dist_sample_strat

        if dist_sample_strat is not None:
            self.dist_sample_strat_fn = self.dist_sample_strat_dict[dist_sample_strat]
            self.undirected_hierarchy = self.hierarchy.to_undirected()
            n = self.undirected_hierarchy.number_of_nodes()

            self.dist_matrix = torch.empty([n, n])
            for dist_tuple in nx.shortest_path_length(self.undirected_hierarchy, weight="weight"):
                distances_sorted_by_node_id = [d for n, d in sorted(dist_tuple[1].items())]
                self.dist_matrix[dist_tuple[0], :] = torch.tensor(distances_sorted_by_node_id)

        self.edge_sample_fn = partial(
            self.edge_sample_from_dict[edge_sample_from],
            hierarchy=self.hierarchy,
            num_negs=self.num_negs,
            sample_strat=self.edge_sample_strat_dict[edge_sample_strat],
        )

    def sample(self, rel: tuple[int, int]) -> dict[str, torch.Tensor]:
        edges, edge_label_targets = self.edge_sample_fn(rel=rel)

        sample = sample = {
            "edges": edges,
            "edge_label_targets": edge_label_targets,
        }

        if self.dist_sample_strat is not None:
            sample["dist_targets"] = self.dist_sample_strat_fn(
                edges=sample["edges"],
                dist_matrix=self.dist_matrix
            )

        return sample
