import torch


def poincare_embeddings_loss(
    dists: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    logits = dists.neg().exp()
    numerator = torch.where(condition=targets, input=logits, other=0).sum(dim=-1)
    denominator = logits.sum(dim=-1)
    loss = (numerator / denominator).log().mean().neg()
    return loss
