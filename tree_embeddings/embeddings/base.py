import torch
import torch.nn as nn

from hypll.manifolds.poincare_ball import PoincareBall
from hypll.tensors import ManifoldParameter, ManifoldTensor


class BaseEmbedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, ball: PoincareBall
    ) -> None:
        super(BaseEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.ball = ball
        
        self.weight = ManifoldParameter(
            data=torch.empty((num_embeddings, embedding_dim)), manifold=ball, man_dim=-1
        )

        self.reset_embeddings()

    def reset_embeddings(self) -> None:
        nn.init.uniform_(
            tensor=self.weight.tensor,
            a=-0.001,
            b=0.001,
        )

    def forward(self, labels: torch.Tensor) -> ManifoldTensor:
        return self.weight[labels]
    
    def score(self, edges: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
