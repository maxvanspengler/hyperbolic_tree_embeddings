import torch
from torch.nn.functional import relu


def hyperbolic_entailment_cone_loss(
    energies: torch.Tensor, targets: torch.Tensor, margin: float = 0.01
) -> torch.Tensor:
    losses = torch.where(
        condition=targets, input=energies, other=relu(margin - energies)
    ).sum(dim=-1)
    return losses.mean()
