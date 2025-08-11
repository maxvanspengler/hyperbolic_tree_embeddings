import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from hypll.manifolds.poincare_ball import PoincareBall
from hypll.tensors import ManifoldParameter

from .loss import hyperbolic_entailment_cone_loss
from .math import energy
from ..base import BaseEmbedding
from ..poincare_embeddings.embedding import PoincareEmbedding
from ..embedding_utils import clone_one_group_optimizer


class EntailmentConeEmbedding(BaseEmbedding):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, ball: PoincareBall, K: float = 0.1
    ) -> None:
        super(EntailmentConeEmbedding, self).__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            ball=ball,
        )
        self.K = K
        self.inner_radius = 2 * K / (1 + (1 + 4 * K ** 2) ** 0.5)

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        embeddings = super(EntailmentConeEmbedding, self).forward(edges)
        energies = energy(
            parent_embeddings=embeddings[:, :, 0, :].tensor,
            child_embeddings=embeddings[:, :, 1, :].tensor,
            K=self.K,
        )
        return energies
    
    def score(self, edges: torch.Tensor, **_kwargs) -> torch.Tensor:
        """
        Score function used for predicting directed edges during evaluation.
        Trivial for entailment cones, but not for Poincare embeddings.
        """
        return -self(edges)
    
    def train(
        self,
        dataloader: DataLoader,
        epochs: int,
        optimizer: Optimizer,
        scheduler: LRScheduler = None,
        margin: float = 0.01,
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
        ).to(self.weight.device)

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
                edges = batch["edges"].to(self.weight.device)
                edge_label_targets = batch["edge_label_targets"].to(self.weight.device)

                optimizer.zero_grad()

                energies = self(edges=edges)
                loss = hyperbolic_entailment_cone_loss(
                    energies=energies, targets=edge_label_targets, margin=margin
                )
                loss.backward()
                optimizer.step()

                # Hacky way to deal with fixing root
                with torch.no_grad():
                    self.weight.tensor[0] = torch.zeros(self.weight.size(1))

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
        min_norm = self.inner_radius + epsilon
        norm = self.weight.tensor.norm(dim=-1, keepdim=True).clamp_min(epsilon)
        cond = norm < min_norm
        projected = self.weight.tensor / norm * min_norm
        new_weight = torch.where(cond, projected, self.weight.tensor)
        max_norm = 1 - epsilon
        cond = norm > max_norm
        projected = new_weight / norm * max_norm
        new_weight = torch.where(cond, projected, self.weight.tensor)
        self.weight = ManifoldParameter(
            data=new_weight, manifold=self.ball, man_dim=-1
        )
