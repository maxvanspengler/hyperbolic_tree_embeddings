from functools import partial
from typing import Callable

import torch


def circle_inversion(x: torch.Tensor, c: torch.Tensor, r: float) -> torch.Tensor:
    diff = x - c
    frac = r.square() / diff.square().sum(dim=-1, keepdim=True) * diff
    return frac + c


def get_circle_inversion_mapping_x_to_origin(
    x: torch.Tensor, curv: torch.Tensor = torch.tensor(1.0)
) -> Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]:
    x_sq = x.square().sum()
    c = 1 / (curv * x_sq) * x
    r = (1 / (curv.square() * x_sq) - 1 / curv).sqrt()
    return partial(circle_inversion, c=c, r=r)


def householder_reflection(
    x: torch.Tensor, normal: torch.Tensor, normalized: bool = False
) -> torch.Tensor:
    if normal.dim() > 1:
        raise ValueError(
            f"Expecting a single normal vector, but received a tensor of size {normal.size()}"
        )
    if not normalized:
        normal = normal / normal.norm()
    return x - 2 * (x * normal).sum(dim=-1)[:, None] * normal[None, :]


def get_householder_reflection_mapping_x_to_y(
    x: torch.Tensor, y: torch.Tensor, equal_norm: bool = False
) -> Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor]:
    if x.dim() > 1 or y.dim() > 1:
        raise ValueError(
            f"Expecting both x and y to be vectors, "
            f"but got tensors of sizes {x.size()} and {y.size()}"
        )
    if not equal_norm:
        if x.norm() != y.norm():
            raise ValueError(
                f"Householder reflection cannot take x to y if their norms aren't equal"
            )
        
    if torch.allclose(x.tensor[:, 0], y.tensor[:, 0], atol=1e-3):
        return lambda x: x
    
    normal = (y - x)
    unit_normal = normal / normal.square().sum().sqrt()
    return partial(householder_reflection, normal=unit_normal, normalized=True)
