from functools import wraps
from typing import Any

import torch

from ..types import floats
    

def _check_inputs(func):
    """
    Decorator for checking tensor inputs to our functions.
    """
    @wraps(func)
    def checked_func(*args: Any, **kwargs: Any):
        # Grab tensors from args and kwargs
        tensors = tuple(t for t in args + tuple(kwargs.values()) if isinstance(t, torch.Tensor))

        # Check for validity of dtypes
        if any(t.dtype not in floats for t in tensors):
            raise ValueError(
                f"Expected input tensors to have dtypes in {floats}, "
                f"but received {[t.dtype for t in tensors]}"
            )
        
        # Check for compatibility of dtypes
        if any(t.dtype != tensors[0].dtype for t in tensors[1:]):
            raise ValueError(
                f"Inputs must have identical dtypes, but received {[t.dtype for t in tensors]}"
            )
        
        # Check if everything is on the same device
        if any(t.device != tensors[0].device for t in tensors[1:]):
            raise ValueError(
                f"Inputs must be on the same device, but received {[t.device for t in tensors]}"
            )
        
        return func(*args, **kwargs)
    
    return checked_func


@_check_inputs
def two_sum(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = a + b
    t = s - b
    amt = a - t
    smt = s - t
    bmsmt = b - smt
    e = amt + bmsmt
    return s, e


@_check_inputs
def fast_two_sum(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = a + b
    z = s - a
    e = b - z
    return s, e


@_check_inputs
def two_mult_fma(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p = x * y
    e = torch.addcmul(-p, x, y)
    return p, e


@_check_inputs
def vec_sum(x: torch.Tensor) -> torch.Tensor:
    s = x[..., -1]
    out = torch.empty_like(x)
    for i in range(1, x.size(-1)):
        s, e = two_sum(x[..., -i - 1], s)
        out[..., -i] = e
    out[..., 0] = s
    return out


@_check_inputs
def vec_sum_err_branch(x: torch.Tensor, m: int) -> list[torch.Tensor]:
    # Grab some sizes
    batch_shapes = x.size()[:-1]
    n = x.size(-1)

    # Initialize some stuff that we need
    j = torch.zeros(batch_shapes, dtype=torch.int64)
    eps = x[..., 0]
    r = torch.zeros(*batch_shapes, m, dtype=x.dtype)

    for i in range(n - 1):
        # Mask where we already set our last output terms
        mask = j < m - 1

        # Compute two_sum and scatter the new values to the output r according to j
        r_out, eps = two_sum(eps, x[..., i + 1])
        r[mask] = torch.scatter(
            input=r[mask],
            dim=-1,
            index=j[mask].unsqueeze(-1),
            src=r_out[mask].unsqueeze(-1),
        )

        # Increment j when the remainder is nonzero
        j = torch.where(eps != 0, j + 1, j)

        # Set eps to prev r_out if eps is 0 itself
        eps = torch.where(eps == 0, r_out, eps)

    # Fill in final values
    mask = torch.logical_and(eps != 0, j < m)
    r[mask] = torch.scatter(
        input=r[mask],
        dim=-1,
        index=j[mask].unsqueeze(-1),
        src=eps[mask].unsqueeze(-1),
    )

    return r


if __name__ == "__main__":
    from mpmath import mp
    mp.dps = 100
    mp.prec = 400
    a = torch.randn(2, 2, dtype=torch.float64)
    b = torch.randn(2, 2, dtype=torch.float64)
    # b = - a
    c = torch.randn(2, 2, dtype=torch.float64)
    # c = torch.zeros(2, 2, dtype=torch.float64)
    print(a)
    print(b)
    print(c)
    print("two_sum:", two_sum(a, b))
    print("fast_two_sum:", fast_two_sum(a, b))
    print("two_mult_fma:", two_mult_fma(a, b))
    print("vec_sum:", vec_sum(torch.stack([a, b, c], dim=2)))
    print("vec_sum_err_branch:", vec_sum_err_branch(a, b, c, m=3))
    print("true sum:", sum(mp.mpf(f"{float(n):.100f}") for n in [a[0, 0], b[0, 0], c[0, 0]]))
    print("vec_sum_err_branch:", sum(mp.mpf(f"{float(n):.100f}") for n in vec_sum_err_branch(a, b, c, m=3)[0, 0, :]))

