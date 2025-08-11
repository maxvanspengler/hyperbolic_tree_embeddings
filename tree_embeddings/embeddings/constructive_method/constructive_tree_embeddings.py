import networkx as nx

import torch

from fpe_torch import FPETensor

from .constructive_tree_embeddings_w_floats import embed_tree
from .constructive_tree_embeddings_w_fpe import embed_tree_w_fpe


def constructively_embed_tree(
    hierarchy: nx.DiGraph,
    dataset: str,
    hierarchy_name: str,
    embedding_dim: int,
    tau: float,
    nc: int,
    curvature: float = 1.0,
    root: int = 0,
    gen_type: str = "optim",
    dtype: torch.dtype = torch.float64,
) -> tuple[FPETensor, float, float]:
    """Constructively embed a tree. Uses normal floats if nc = 1 and floating point
    expansions if nc > 1.
    """
    if nc == 1:
        embeddings, rel_dist_mean, rel_dist_max = embed_tree(
            hierarchy=hierarchy,
            dataset=dataset,
            hierarchy_name=hierarchy_name,
            embedding_dim=embedding_dim,
            tau=tau,
            curvature=curvature,
            root=root,
            gen_type=gen_type,
            dtype=dtype,
        )
    else:
        embeddings, rel_dist_mean, rel_dist_max = embed_tree_w_fpe(
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

    return embeddings, rel_dist_mean, rel_dist_max
