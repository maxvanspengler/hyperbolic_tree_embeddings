from functools import wraps
from typing import Any

import torch

from .basic_flops import (
    fast_two_sum,
    two_mult_fma,
    vec_sum,
    vec_sum_err_branch,
)


CONSTANTS = {
    torch.float16: {
        "BIN_SIZE": 7,
        "PREC": 11,
    },
    torch.float32: {
        "BIN_SIZE": 18,
        "PREC": 24,
    },
    torch.float64: {
        "BIN_SIZE": 45,
        "PREC": 53,
    }
}


def _check_inputs(func):
    """
    Decorator for checking tensor inputs to our functions.
    """
    @wraps(func)
    def checked_func(*args: Any, **kwargs: Any):
        # Grab tensors from args and kwargs
        tensors = tuple(t for t in args + tuple(kwargs.values()) if isinstance(t, torch.Tensor))

        # Check for compatibility of dtypes
        if any(t.dtype != tensors[0].dtype for t in tensors[1:]):
            raise ValueError(
                f"Inputs must have identical dtypes, but received {[t.dtype for t in tensors]}"
            )
        
        # Check if everything is on the same device
        if any(t.device != tensors[0].device for t in tensors[1:]):
            raise ValueError(
                f"Inputs must be on the same device, but received {[t.device for t in args]}"
            )
        
        return func(*args, **kwargs)
    
    return checked_func


@_check_inputs
def fpe_merge(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    new_tensor = torch.cat([x, y], dim=-1)
    index = torch.argsort(new_tensor.abs(), dim=-1, descending=True)
    new_tensor = torch.gather(new_tensor, dim=-1, index=index)
    return new_tensor


@_check_inputs
def fpe_renorm(x: torch.Tensor, m: int) -> torch.Tensor:
    e = vec_sum(x)
    r = vec_sum_err_branch(e, m)
    return r


@_check_inputs
def fpe_renorm_arbit(x: torch.Tensor, m: int) -> torch.Tensor:
    e = torch.zeros_like(x)
    e[..., 0] = x[..., 0]
    for i in range(1, x.size(-1)):
        e[..., :i + 1] = vec_sum(
            torch.cat((e[..., :i], x[..., i].unsqueeze(-1)), dim=-1)
        )
    return vec_sum_err_branch(e, m)


@_check_inputs
def fpe_add(a: torch.Tensor, b: torch.Tensor, r: int) -> torch.Tensor:
    f = fpe_merge(a, b)
    s = fpe_renorm(f, r)
    return s


def fpe_mult(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    The inputs need to be normalized!
    """
    k = a.size(-1)
    assert k == b.size(-1), "Need a and b to have an equal number of terms"
    broadcasted_shape = torch.broadcast_shapes(a.size()[:-1], b.size()[:-1])

    r = torch.zeros(*broadcasted_shape, k + 1, dtype=a.dtype)
    e = torch.zeros(*broadcasted_shape, k ** 2 + 2 * k + 1, dtype=a.dtype)
    r[..., 0], e[..., 0] = two_mult_fma(a[..., 0], b[..., 0])

    for n in range(1, k):
        p = torch.zeros(*broadcasted_shape, n + 1, dtype=a.dtype)
        e_hat = torch.zeros(*broadcasted_shape, n + 1, dtype=a.dtype)
        for i in range(n + 1):
            p[..., i], e_hat[..., i] = two_mult_fma(a[..., i], b[..., n - i])
        vec_sum_out = vec_sum(torch.cat((p[..., :n + 1], e[..., :n ** 2]), dim=-1))
        r[..., n] = vec_sum_out[..., 0]
        e[..., :n ** 2 + 2 * n + 1] = torch.cat((vec_sum_out[..., 1:], e_hat), dim=-1)

    for i in range(1, k):
        r[..., k] = r[..., k] + a[..., i] * b[..., k - i]

    for i in range(k ** 2):
        r[..., k] = r[..., k] + e[..., i]

    r[..., :k] = fpe_renorm(r, k)
    return r[..., :k]


def accumulate(
    p: torch.Tensor,
    e: torch.Tensor,
    bins: torch.Tensor, 
    sh: torch.Tensor,
    l: torch.Tensor,
) -> torch.Tensor:
    prec = CONSTANTS[p.dtype]["PREC"]
    bin_size = CONSTANTS[p.dtype]["BIN_SIZE"]
    c = prec - bin_size - 1
    # mask1 = l < bin_size - 2 * c - 1
    # mask2 = torch.logical_and(~mask1, l < bin_size - c)
    # mask3 = l >= bin_size - c

    case_masks = [l < bin_size - 2 * c - 1]
    case_masks.append(torch.logical_and(~case_masks[0], l < bin_size - c))
    case_masks.append(l >= bin_size - c)
    sh_masks = [sh < bins.size(-1) - i for i in range(4)]

    mask_fn = lambda case, sh_n: torch.logical_and(case_masks[case], sh_masks[sh_n])


    # if l < bin_size - 2 * c - 1:

    mask = mask_fn(0, 0)
    new_bin_vals, p[mask] = fast_two_sum(
        torch.gather(bins[mask], dim=-1, index=sh[mask].unsqueeze(-1)).squeeze(),
        p[mask],
    )
    bins[mask] = torch.scatter(
        input=bins[mask], dim=-1, index=sh[mask].unsqueeze(-1), src=new_bin_vals.unsqueeze(-1)
    )

    mask = mask_fn(0, 1)
    bins[mask] = torch.scatter_add(
        input=bins[mask], dim=-1, index=(sh[mask] + 1).unsqueeze(-1), src=p[mask].unsqueeze(-1)
    )
    new_bin_vals, e[mask] = fast_two_sum(
        torch.gather(bins[mask], dim=-1, index=(sh[mask] + 1).unsqueeze(-1)).squeeze(),
        e[mask],
    )
    bins[mask] = torch.scatter(
        input=bins[mask], dim=-1, index=(sh[mask] + 1).unsqueeze(-1), src=new_bin_vals.unsqueeze(-1)
    )

    mask = mask_fn(0, 2)
    bins[mask] = torch.scatter_add(
        input=bins[mask], dim=-1, index=(sh[mask] + 2).unsqueeze(-1), src=e[mask].unsqueeze(-1)
    )


    # elif l < bin_size - c:
    mask = mask_fn(1, 0)
    new_bin_vals, p[mask] = fast_two_sum(
        torch.gather(bins[mask], dim=-1, index=sh[mask].unsqueeze(-1)).squeeze(),
        p[mask],
    )
    bins[mask] = torch.scatter(
        input=bins[mask], dim=-1, index=sh[mask].unsqueeze(-1), src=new_bin_vals.unsqueeze(-1)
    )

    mask = mask_fn(1, 1)
    bins[mask] = torch.scatter_add(
        input=bins[mask], dim=-1, index=(sh[mask] + 1).unsqueeze(-1), src=p[mask].unsqueeze(-1)
    )
    new_bin_vals, e[mask] = fast_two_sum(
        torch.gather(bins[mask], dim=-1, index=(sh[mask] + 1).unsqueeze(-1)).squeeze(),
        e[mask],
    )
    bins[mask] = torch.scatter(
        input=bins[mask], dim=-1, index=(sh[mask] + 1).unsqueeze(-1), src=new_bin_vals.unsqueeze(-1)
    )

    mask = mask_fn(1, 2)
    new_bin_vals, e[mask] = fast_two_sum(
        torch.gather(bins[mask], dim=-1, index=(sh[mask] + 2).unsqueeze(-1)).squeeze(),
        e[mask],
    )
    bins[mask] = torch.scatter(
        input=bins[mask], dim=-1, index=(sh[mask] + 2).unsqueeze(-1), src=new_bin_vals.unsqueeze(-1)
    )

    mask = mask_fn(1, 3)
    bins[mask] = torch.scatter_add(
        input=bins[mask], dim=-1, index=(sh[mask] + 3).unsqueeze(-1), src=e[mask].unsqueeze(-1)
    )


    # else:
    mask = mask_fn(2, 0)
    new_bin_vals, p[mask] = fast_two_sum(
        torch.gather(bins[mask], dim=-1, index=sh[mask].unsqueeze(-1)).squeeze(),
        p[mask],
    )
    bins[mask] = torch.scatter(
        input=bins[mask], dim=-1, index=sh[mask].unsqueeze(-1), src=new_bin_vals.unsqueeze(-1)
    )

    mask = mask_fn(2, 1)
    new_bin_vals, p[mask] = fast_two_sum(
        torch.gather(bins[mask], dim=-1, index=(sh[mask] + 1).unsqueeze(-1)).squeeze(),
        p[mask],
    )
    bins[mask] = torch.scatter(
        input=bins[mask], dim=-1, index=(sh[mask] + 1).unsqueeze(-1), src=new_bin_vals.unsqueeze(-1)
    )

    mask = mask_fn(2, 2)
    bins[mask] = torch.scatter_add(
        input=bins[mask], dim=-1, index=(sh[mask] + 2).unsqueeze(-1), src=p[mask].unsqueeze(-1)
    )
    new_bin_vals, e[mask] = fast_two_sum(
        torch.gather(bins[mask], dim=-1, index=(sh[mask] + 2).unsqueeze(-1)).squeeze(),
        e[mask],
    )
    bins[mask] = torch.scatter(
        input=bins[mask], dim=-1, index=(sh[mask] + 2).unsqueeze(-1), src=new_bin_vals.unsqueeze(-1)
    )

    mask = mask_fn(2, 3)
    bins[mask] = torch.scatter_add(
        input=bins[mask], dim=-1, index=(sh[mask] + 3).unsqueeze(-1), src=e[mask].unsqueeze(-1)
    )

    return bins


def fpe_mult_acc(a: torch.Tensor, b: torch.Tensor, r: int) -> torch.Tensor:
    # Compute some shapes and grab some constants
    broadcasted_shape = torch.broadcast_shapes(a.size()[:-1], b.size()[:-1])
    bin_size = CONSTANTS[a.dtype]["BIN_SIZE"]
    prec = CONSTANTS[a.dtype]["PREC"]
    bin_count = r * prec // bin_size + 2
    bins = torch.zeros(*broadcasted_shape, bin_count, dtype=a.dtype)

    # Compute some relevant masks, indicating the locations of zeros in a and b
    a_mask = a != 0
    b_mask = b != 0
    mask0 = torch.logical_and(a_mask[..., 0], b_mask[..., 0])

    # Compute the sums of the binary exponents of the first terms of a and b
    e0 = (
        a[..., 0].abs().log2().floor()
        + b[..., 0].abs().log2().floor()
    )
    
    # Fill the bins with some initial values based on e0
    for i in range(bin_count):
        bins[..., i][mask0] = 1.5 * 2 ** (e0[mask0] - (i + 1) * bin_size + prec - 1)

    # Expand a and b for some non-broadcastable stuff in the loop below
    a = a.broadcast_to(*broadcasted_shape, a.size(-1))
    b = b.broadcast_to(*broadcasted_shape, b.size(-1))

    for i in range(min(a.size(-1), r + 1)):
        for j in range(min(b.size(-1), r - i)):
            # Grab the zero-mask for ai and bj
            mask = torch.logical_and(a_mask[..., i], b_mask[..., j])

            # Compute the product of ai and bj
            p, e = two_mult_fma(a[..., i][mask], b[..., j][mask])

            # Compute which bins we have to add to
            l = (
                e0[mask]
                - a[..., i][mask].abs().log2().floor()
                - b[..., j][mask].abs().log2().floor()
            ).to(torch.int64)
            sh = l // bin_size
            l = l - sh * bin_size

            # Add stuff to the bins
            bins[mask] = accumulate(p, e, bins[mask], sh, l)

        if j < b.size(-1) - 1:
            p = a[..., i][mask] * b[..., j][mask]
            l = (
                e0[mask]
                - a[..., i][mask].abs().log2().floor()
                - b[..., j][mask].abs().log2().floor()
            ).to(torch.int64)
            sh = l // bin_size
            l = l - sh * bin_size
            bins[mask] = accumulate(p, torch.zeros_like(p), bins[mask], sh, l)

    # Remove the initial values from the bins again
    for i in range(int(r * prec / bin_size) + 2):
        bins[..., i][mask0] = (
            bins[..., i][mask0]
            - 1.5 * 2 ** (e0[mask0] - (i + 1) * bin_size + prec - 1)
        )

    # Renormalize and return
    return fpe_renorm(bins[..., :bin_count], r)


def fpe_reciprocal(x: torch.Tensor, q: int) -> torch.Tensor:
    r = torch.zeros(*x.size()[:-1], 2 ** q, dtype=x.dtype)
    r[..., 0] = 1 / x[..., 0]
    for i in range(q):
        v = fpe_mult_acc(r[..., :2 ** i], x[..., :2 ** (i + 1)], 2 ** (i + 1))
        w = fpe_renorm(
            torch.cat((-v, 2 * torch.ones(*v.size()[:-1], 1)), dim=-1),
            2 ** (i + 1),
        )
        r = fpe_mult_acc(r[..., :2 ** i], w[..., :2 ** (i + 1)], 2 ** (i + 1))
    
    return r


def fpe_reciprocal_sqrt(x: torch.Tensor, q: int) -> torch.Tensor:
    r = torch.zeros(*x.size()[:-1], 2 ** q, dtype=x.dtype)
    r[..., 0] = 1 / x[..., 0].sqrt()
    for i in range(q):
        v = fpe_mult_acc(r[..., :2 ** i], x[..., :2 ** (i + 1)], 2 ** (i + 1))
        w = fpe_mult_acc(r[..., :2 ** i], v[..., :2 ** (i + 1)], 2 ** (i + 1))
        y = fpe_renorm(
            torch.cat((-w, 3 * torch.ones(*w.size()[:-1], 1)), dim=-1),
            2 ** (i + 1),
        )
        z = fpe_mult_acc(r[..., :2 ** i], y[..., :2 ** (i + 1)], 2 ** (i + 1))
        r[..., :2 ** (i + 1)] = 0.5 * z[..., :2 ** (i + 1)]
        
    return r


def fpe_sqrt(x: torch.Tensor, q: int) -> torch.Tensor:
    r = torch.zeros(*x.size()[:-1], 2 ** q, dtype=x.dtype)
    mask = x[..., 0] != 0
    f = fpe_reciprocal_sqrt(x[mask], q)
    r[mask] = fpe_mult_acc(x[mask], f, 2 ** q)
    return r
