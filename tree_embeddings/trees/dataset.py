from typing import Literal, Optional

import networkx as nx

import torch
from torch.utils.data import Dataset

from .samplers.edge_sampler import EdgeSampler


class TreeEmbeddingDataset(Dataset):
    def __init__(
        self,
        hierarchy: nx.DiGraph,
        num_negs: int = 10,
        edge_sample_from: Literal["both", "source", "target"] = "both",
        edge_sample_strat: Literal["uniform", "siblings"] = "uniform",
        dist_sample_strat: Optional[str] = None,
    ):
        super(TreeEmbeddingDataset, self).__init__()
        self.hierarchy = hierarchy
        self.num_negs = num_negs
        self.edge_sample_from = edge_sample_from
        self.edge_sample_strat = edge_sample_strat
        self.dist_sample_strat = dist_sample_strat

        self.sampler = EdgeSampler(
            hierarchy=self.hierarchy,
            num_negs=self.num_negs,
            edge_sample_from=edge_sample_from,
            edge_sample_strat=edge_sample_strat,
            dist_sample_strat=dist_sample_strat
        )

        self.edges_list = list(hierarchy.edges())

    def __len__(self) -> int:
        return len(self.edges_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rel = self.edges_list[idx]
        sample = self.sampler.sample(rel=rel)
        return sample
