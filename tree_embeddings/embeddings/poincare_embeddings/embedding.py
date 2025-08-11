import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from hypll.manifolds.poincare_ball import PoincareBall

from .loss import poincare_embeddings_loss
from ..base import BaseEmbedding


class PoincareEmbedding(BaseEmbedding):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, ball: PoincareBall
    ) -> None:
        super(PoincareEmbedding, self).__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            ball=ball,
        )

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        """
        Input:
            - edges: torch.Tensor of shape [batch_size, sample_size, 2],
              where sample_size is generally (1 + #negatives). 
        """
        embeddings = super(PoincareEmbedding, self).forward(edges)
        edge_distances = self.ball.dist(x=embeddings[:, :, 0, :], y=embeddings[:, :, 1, :])
        return edge_distances
    
    def score(self, edges: torch.Tensor, alpha: float = 1) -> torch.Tensor:
        embeddings = super(PoincareEmbedding, self).forward(edges)
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
        burn_in_epochs: int = 10,
        burn_in_lr_mult: float = 0.1,
        store_losses: bool = False,
        store_intermediate_weights: bool = False,
        **kwargs
    ) -> None | list:
        # Store initial learning rate
        lr = optimizer.param_groups[0]["lr"]

        if store_losses:
            losses = []
        if store_intermediate_weights:
            weights = [self.weight.clone().detach()]

        for epoch in range(epochs):
            # Scale learning rate during burn-in
            if epoch < burn_in_epochs:
                optimizer.param_groups[0]["lr"] = lr * burn_in_lr_mult
            else:
                optimizer.param_groups[0]["lr"] = lr

            for idx, batch in enumerate(dataloader):
                edges = batch["edges"].to(self.weight.device)
                edge_label_targets = batch["edge_label_targets"].to(self.weight.device)

                optimizer.zero_grad()

                dists = self(edges=edges)
                loss = poincare_embeddings_loss(dists=dists, targets=edge_label_targets)
                loss.backward()
                optimizer.step()

                if not (epoch + 1) % 100:
                    print(f"Epoch {epoch + 1}, batch {idx + 1}/{len(dataloader)}:  {loss}")
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
