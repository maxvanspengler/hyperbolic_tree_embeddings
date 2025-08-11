from typing import Optional

import torch

from .fpe_tensor import FPETensor


def zeros(
    *size, terms: int, dtype: Optional[torch.dtype] = None, device: Optional[torch.device]
) -> FPETensor:
    size += [terms]
    tensor = torch.zeros(size, dtype=dtype, device=device)
    return FPETensor.from_tensor(tensor)


def ones(
    *size, terms: int, dtype: Optional[torch.dtype] = None, device: Optional[torch.device]
) -> FPETensor:
    tensor = torch.cat(
        tensors=(
            torch.ones(*size, 1, dtype=dtype, device=device),
            torch.zeros(*size, terms - 1, dtype=dtype, device=device),
        ),
        dim=-1,
    )

    return FPETensor.from_tensor(tensor)
