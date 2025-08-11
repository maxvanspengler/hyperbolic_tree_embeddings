import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from hypll.manifolds.poincare_ball import PoincareBall
from hypll.tensors import ManifoldParameter

from .loss import distortion_loss
from ..base import BaseEmbedding
from ..poincare_embeddings.embedding import PoincareEmbedding
from ..embedding_utils import clone_one_group_optimizer


class DistortionEmbedding(BaseEmbedding):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, ball: PoincareBall, tau: float = 1.0
    ) -> None:
        super(DistortionEmbedding, self).__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            ball=ball,
        )

        self.tau = tau

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        embeddings = super(DistortionEmbedding, self).forward(edges)
        return embeddings
    
    def score(self, edges: torch.Tensor, alpha: float = 1) -> torch.Tensor:
        """
        Score function used for predicting directed edges during evaluation.
        Trivial for entailment cones, but not for Poincare embeddings. Note that **_kwargs
        catches unused keywords such as the alpha from Poincare embeddings.
        """
        embeddings = super(DistortionEmbedding, self).forward(edges)
        embedding_norms = embeddings.norm(dim=-1)
        edge_distances = self.ball.dist(x=embeddings[:, :, 0, :], y=embeddings[:, :, 1, :])
        return - (
            1 + alpha * (embedding_norms[:, :, 0] - embedding_norms[:, :, 1])
        ) * edge_distances
    
    def train(
        self,
        dataloader: DataLoader,
        epochs: int,
        optimizer: Optimizer,
        scheduler: LRScheduler = None,
        pretrain_epochs: int = 100,
        pretrain_lr: float = 5.0,
        burn_in_epochs: int = 10,
        burn_in_lr_mult: float = 0.1,
        store_losses: bool = False,
        store_intermediate_weights: bool = False,
        **kwargs
    ) -> None:
        # Initialize a Poincare embeddings model for pretraining
        poincare_embeddings = PoincareEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            ball=self.ball,
        )

        # Copy the optimizer, but change the parameters to the Poincare embeddings model weights
        pretraining_optimizer = clone_one_group_optimizer(
            optimizer=optimizer,
            new_params=poincare_embeddings.parameters(),
            lr=pretrain_lr,
            momentum=0,
            weight_decay=0,
        )

        # Perform pretraining
        losses, weights = poincare_embeddings.train(
            dataloader=dataloader,
            epochs=pretrain_epochs,
            optimizer=pretraining_optimizer,
            scheduler=None,
            burn_in_epochs=burn_in_epochs,
            burn_in_lr_mult=burn_in_lr_mult,
            store_losses=store_losses,
            store_intermediate_weights=store_intermediate_weights,
            **kwargs
        )

        # Copy pretrained embeddings, rescale and clip these and reset optimizer param group
        with torch.no_grad():
            self.weight.copy_(poincare_embeddings.weight)
            self.weight.tensor.mul_(0.8)
            self._clip_embeddings()
            optimizer = clone_one_group_optimizer(
                optimizer=optimizer,
                new_params=self.parameters(),
            )

        for epoch in range(epochs):
            for idx, batch in enumerate(dataloader):
                edges = batch["edges"]
                dist_targets = self.tau * batch["dist_targets"]

                optimizer.zero_grad()

                embeddings = self(edges=edges)

                loss = distortion_loss(
                    embeddings=embeddings,
                    dist_targets=dist_targets,
                    ball=self.ball
                )

                loss.backward()
                optimizer.step()

                if not (epoch + 1) % 100:
                    print(f"Epoch {epoch + 1}, batch {idx + 1}/{len(dataloader)}:  {loss}")
                    if store_intermediate_weights:
                        weights.append(self.weight.clone().detach())

                if store_losses:
                    losses.append(loss.item())

            if store_intermediate_weights:
                weights.append(self.weight.clone().detach())

            if scheduler is not None:
                scheduler.step(epoch=epoch + 1)
        
        return (
            losses if store_losses else None,
            weights if store_intermediate_weights else None,
        )

    def _clip_embeddings(self, epsilon: float = 1e-5) -> None:
        norm = self.weight.tensor.norm(dim=-1, keepdim=True).clamp_min(epsilon)
        max_norm = 1 - epsilon
        cond = norm > max_norm
        projected = self.weight.tensor / norm * max_norm
        new_weight = torch.where(cond, projected, self.weight.tensor)
        self.weight = ManifoldParameter(
            data=new_weight, manifold=self.ball, man_dim=-1
        )
