import torch
from torch.nn.functional import relu


def xi(parent: torch.Tensor, child: torch.Tensor, dim: int = -1) -> torch.Tensor:
    parent_norm = parent.norm(dim=dim)
    parent_norm_sq = parent_norm.square()
    child_norm = child.norm(dim=dim)
    child_norm_sq = child_norm.square()

    parent_dot_child = torch.einsum("bij,bij->bi", parent, child)

    numerator = (
        parent_dot_child * (1 + parent_norm_sq)
        - parent_norm_sq * (1 + child_norm_sq)
    )
    denominator = (
        parent_norm * (parent - child).norm(dim=dim)
        * (1 + parent_norm_sq * child_norm_sq - 2 * parent_dot_child).sqrt()
    )

    return (numerator / denominator.clamp_min(1e-15)).clamp(min=-1 + 1e-5, max=1 - 1e-5).arccos()


def psi(x: torch.Tensor, K: float = 0.1, dim: int = -1) -> torch.Tensor:
    x_norm = x.norm(dim=dim)
    arcsin_arg = K * (1 - x_norm.square()) / x_norm.clamp_min(1e-15)
    return arcsin_arg.clamp(min=-1 + 1e-5, max=1 - 1e-5).arcsin()


def energy(
    parent_embeddings: torch.Tensor, child_embeddings, K: float = 0.1
) -> torch.Tensor:
    xi_angles = xi(parent=parent_embeddings, child=child_embeddings, dim=-1)
    psi_parent = psi(x=parent_embeddings, K=K, dim=-1)
    return relu(xi_angles - psi_parent)
