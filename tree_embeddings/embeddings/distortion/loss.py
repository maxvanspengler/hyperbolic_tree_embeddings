import torch

from hypll.manifolds.poincare_ball import PoincareBall


def distortion_loss(
    embeddings: torch.Tensor, dist_targets: torch.Tensor, ball: PoincareBall
) -> torch.Tensor:
    embedding_dists = ball.dist(x=embeddings[:, :, 0, :], y=embeddings[:, :, 1, :])
    losses = (embedding_dists - dist_targets).abs() / dist_targets
    return losses.mean()
