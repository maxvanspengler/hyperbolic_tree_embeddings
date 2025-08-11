from __future__ import annotations

import math
from typing import Any, Optional

import mpmath.ctx_mp_python
import torch

import mpmath
from mpmath import mp

from ..math import basic_fpeops as ops


class FPETensor:
    def __init__(
        self,
        data: Any,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False,
        renorm_arbit: bool = False,
    ) -> None:
        if isinstance(data, torch.Tensor):
            self.tensor = data
        else:
            self.tensor = torch.tensor(
                data, dtype=dtype, device=device, requires_grad=requires_grad
            )

        if renorm_arbit:
            self.tensor = ops.fpe_renorm_arbit(self.tensor, m=self.terms)
        else:
            self.tensor = ops.fpe_renorm(self.tensor, m=self.terms)

    @property
    def terms(self) -> int:
        return self.tensor.size(-1)
    
    @property
    def shape(self) -> torch.Size:
        return self.tensor.size()[:-1]
    
    @property
    def dtype(self)-> torch.dtype:
        return self.tensor.dtype
    
    @property
    def device(self) -> torch.device:
        return self.tensor.device
    
    @property
    def requires_grad(self) -> bool:
        return self.tensor.requires_grad
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, renorm_arbit: bool = True) -> FPETensor:
        return cls(data=tensor.data, renorm_arbit=renorm_arbit)
    
    @classmethod
    def from_mpmath(
        cls,
        mp_numbers: list[mpmath.ctx_mp_python.mpf],
        terms: int,
        renorm_arbit: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> FPETensor:
        mp.dps = 20 * terms

        out = torch.empty(len(mp_numbers), terms, dtype=dtype)     
        for idx, mp_number in enumerate(mp_numbers):
            for t in range(terms):
                out[idx, t] = torch.tensor(float(mp_number), dtype=dtype)
                mp_number = mp_number - mp.mpf(str(float(out[idx, t])))

        return cls.from_tensor(tensor=out, renorm_arbit=renorm_arbit)
    
    def __repr__(self) -> str:
        str_repr = repr(self.tensor).replace("tensor", "FPETensor")
        return str_repr.replace(")", f", terms={self.terms})")
    
    def __getitem__(self, args: Any) -> FPETensor:
        if isinstance(args, torch.Tensor) and args.dim() > self.dim():
            raise ValueError(
                f"Attempting to __getitem__ from FPETensor with {self.dim()} dimensions with a "
                f"tensor with {args.dim()} dimensions."
            )

        if isinstance(args, tuple):
            has_ellipsis = Ellipsis in args
            none_count = args.count(None)
            slicing_dim_count = len(args) - has_ellipsis - none_count

            if slicing_dim_count > self.dim():
                raise ValueError(
                    f"Trying to slice into at least {slicing_dim_count} dimensions, but this "
                    f"FPETensor only has {self.dim()} dimensions."
                )
            
            if has_ellipsis:
                args = args + (slice(None, None, None),)

        new_tensor = self.tensor.__getitem__(args)
        return FPETensor.from_tensor(new_tensor)
    
    def __setitem__(self, indices: Any, data: Any):
        if isinstance(indices, torch.Tensor) and indices.dim() > self.dim():
            raise ValueError(
                f"Attempting to __getitem__ from FPETensor with {self.dim()} dimensions with a "
                f"tensor with {indices.dim()} dimensions."
            )
        
        if isinstance(indices, tuple):
            has_ellipsis = Ellipsis in indices
            slicing_dim_count = len(indices) - has_ellipsis

            if slicing_dim_count > self.dim():
                raise ValueError(
                    f"Trying to slice into {slicing_dim_count}, but this FPETensor only has "
                    f"{self.dim()} dimensions."
                )

        if isinstance(data, FPETensor):
            self.tensor.__setitem__(indices, data.tensor)
        elif isinstance(data, (int, float, bool, torch.Tensor)):
            self.tensor[..., 0].__setitem__(indices, data)
            self.tensor[..., 1:].__setitem__(indices, 0)
        else:
            raise ValueError(
                f"Trying to set with {type(data)}, but that's not defined."
            )
        
    def __neg__(self) -> FPETensor:
        new_tensor = -self.tensor
        return FPETensor.from_tensor(new_tensor)
    
    def __add__(self, other: Any) -> FPETensor:
        if isinstance(other, FPETensor):
            broadcasted_shape = torch.broadcast_shapes(self.size(), other.size())
            new_tensor = ops.fpe_add(
                a=self.tensor.broadcast_to(*broadcasted_shape, self.terms),
                b=other.tensor.broadcast_to(*broadcasted_shape, other.terms),
                r=max(self.terms, other.terms),
            )

        elif isinstance(other, torch.Tensor):
            broadcasted_shape = torch.broadcast_shapes(self.size(), other.size())
            new_tensor = torch.cat(
                tensors=(
                    self.tensor.broadcast_to(*broadcasted_shape, self.terms),
                    other.broadcast_to(broadcasted_shape).unsqueeze(-1),
                ),
                dim=-1,
            )
            new_tensor = ops.fpe_renorm(new_tensor, m=self.terms)

        elif isinstance(other, (int, float)):
            new_tensor = torch.cat(
                tensors=(self.tensor, other * torch.ones(*self.shape, 1)),
                dim=-1,
            )
            new_tensor = ops.fpe_renorm(new_tensor, m=self.terms)

        else:
            return NotImplemented

        return FPETensor.from_tensor(new_tensor)
    
    def __radd__(self, other: Any) -> FPETensor:
        return self.__add__(other)
    
    def __sub__(self, other: Any) -> FPETensor:
        return self.__add__(-other)
    
    def __rsub__(self, other: Any) -> FPETensor:
        return (-self).__add__(other)
    
    def __mul__(self, other: Any) -> FPETensor:
        if isinstance(other, FPETensor):
            new_tensor = ops.fpe_mult_acc(
                self.tensor, other.tensor, r=max(self.terms, other.terms)
            )
        elif isinstance(other, torch.Tensor):
            new_tensor = ops.fpe_renorm(self.tensor * other.unsqueeze(-1), m=self.terms)
        elif isinstance(other, (int, float)):
            new_tensor = ops.fpe_renorm(self.tensor * other, m=self.terms)
        else:
            return NotImplemented
        
        return FPETensor.from_tensor(new_tensor)
    
    def __rmul__(self, other: Any) -> FPETensor:
        return self.__mul__(other)
    
    def __truediv__(self, other: Any) -> FPETensor:
        if isinstance(other, FPETensor):
            q = math.floor(math.log2(other.terms))
            new_tensor = ops.fpe_reciprocal(other.tensor, q=q)
            new_tensor = ops.fpe_mult_acc(
                self.tensor, new_tensor, r=max(self.terms, new_tensor.size(-1))
            )
            return FPETensor.from_tensor(new_tensor)
        elif isinstance(other, (int, float)):
            return 1 / other * self
        else:
            return NotImplemented
    
    def __rtruediv__(self, other: Any) -> FPETensor:
        if isinstance(other, (int, float)):
            q = math.floor(math.log2(self.terms))
            new_tensor = ops.fpe_reciprocal(self.tensor, q=q)
            return other * FPETensor.from_tensor(new_tensor)
        else:
            return NotImplemented
        
    def __eq__(self, other: Any) -> torch.Tensor:
        try:
            other = self._convert_other(other)
        except:
            return NotImplemented

        return (self.tensor == other).all(dim=-1)

    def __neq__(self, other: Any) -> torch.Tensor:
        return ~(self == other)
    
    def __lt__(self, other: Any) -> torch.Tensor:
        other = self._convert_other(other)
        if other == NotImplemented:
            return NotImplemented
        
        neq = self.tensor != other
        first_neq = neq.int().argmax(dim=-1)
        lt = self.tensor < other
        first_lt = lt.int().argmax(dim=-1)
        return torch.logical_and(
            torch.logical_and(neq.any(dim=-1), lt.any(dim=-1)),
            first_neq == first_lt,
        )

    def __le__(self, other: Any) -> torch.Tensor:
        return torch.logical_or(self == other, self < other)

    def __gt__(self, other: Any) -> torch.Tensor:
        other = self._convert_other(other)
        if other == NotImplemented:
            return NotImplemented
        
        neq = self.tensor != other
        first_neq = neq.int().argmax(dim=-1)
        first_gt = (self.tensor > other).int().argmax(dim=-1)
        return torch.logical_and(neq.any(dim=-1), first_neq == first_gt)

    def __ge__(self, other: Any) -> torch.Tensor:
        return torch.logical_or(self == other, self > other)

    def dim(self):
        return self.tensor.dim() - 1
    
    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> FPETensor:
        if end_dim < 0:
            end_dim = self.dim() + end_dim
        new_tensor = self.tensor.flatten(start_dim, end_dim)
        return FPETensor.from_tensor(new_tensor)
    
    def squeeze(self, dim: Optional[int | tuple[int]] = None) -> FPETensor:
        # If dim isn't given, exit early
        if dim is None:
            new_tensor = self.tensor.squeeze()
            return FPETensor.from_tensor(new_tensor)

        # Turn int arguments to iterable to remove some boilerplate
        if isinstance(dim, int):
            dim = (dim,)
        
        # Check if the arguments are in range
        if hasattr(dim, "__iter__") and any((d < -self.dim() or d > self.dim() - 1) for d in dim):
            raise IndexError(
                f"Dimension out of range (expected to be in range {-self.dim(), self.dim() - 1}) "
                f"but got {dim})"
            )
        
        # Convert negative arguments to positive arguments
        dim = tuple(self.dim() + d if d < 0 else d for d in dim)
        
        # Squeeze and return
        new_tensor = self.tensor.squeeze(dim)
        return FPETensor.from_tensor(new_tensor)
    
    def unsqueeze(self, dim: int) -> FPETensor:
        # Check if the arguments are in range
        if dim < -self.dim() - 1 or dim > self.dim():
            raise IndexError(
                f"Dimension out of range (expected to be in range {-self.dim() - 1, self.dim()}) "
                f"but got {dim})"
            )
        
        # Convert negative arguments to positive arguments
        if dim < 0:
            dim = self.dim() + dim
        
        # Unsqueeze and return
        new_tensor = self.tensor.unsqueeze(dim)
        return FPETensor.from_tensor(new_tensor)
    
    def select(self, dim: int, index: int) -> FPETensor:
        new_tensor = self.tensor.select(dim=dim, index=index)
        return FPETensor.from_tensor(new_tensor)
    
    def index_select(self, dim: int, index: torch.IntTensor | torch.LongTensor) -> FPETensor:
        new_tensor = self.tensor.index_select(dim=dim, index=index)
        return FPETensor.from_tensor(new_tensor)
    
    def index_fill(
        self, dim: int, index: torch.IntTensor | torch.LongTensor, value: FPETensor | torch.Tensor
    ) -> FPETensor:
        if isinstance(value, FPETensor):
            value = value.tensor
        new_tensor = self.tensor.index_fill(dim=dim, index=index, value=value)
        return FPETensor.from_tensor(new_tensor)
    
    def index_put(
        self,
        dim: int,
        index: torch.IntTensor | torch.LongTensor,
        value: FPETensor | torch.Tensor,
        accumulate: bool = False,
    ) -> FPETensor:
        if accumulate:
            value = self.index_select(dim=dim, index=index) + value
        return self.index_fill(dim=dim, index=index, value=value)
    
    def chunk(self, chunks: int, dim: int = 0) -> tuple[FPETensor]:
        if dim < 0:
            dim = self.dim() + dim
        chunk_tuple = self.tensor.chunk(chunks=chunks, dim=dim)
        return tuple(FPETensor.from_tensor(c) for c in chunk_tuple)

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.shape
        elif dim >= self.dim():
            raise ValueError(
                f"Cannot access dimension {dim} of FPETensor with {self.dim()} dimensions."
            )
        elif dim < 0:
            dim = self.dim() + dim

        return self.tensor.size(dim)
        
    def abs(self) -> FPETensor:
        sign = self.tensor[..., 0].sign()
        return self * sign

    def square(self) -> FPETensor:
        return self * self

    def sqrt(self) -> FPETensor:
        new_tensor = ops.fpe_sqrt(self.tensor, int(math.log2(self.terms)))
        return FPETensor.from_tensor(new_tensor)
    
    def sum(self, dim: Optional[int | tuple] = None, keepdim: bool = False) -> FPETensor:
        if isinstance(dim, (list, tuple)):
            # This case is kind of annoying as we have to flatten the specified dimensions to do it
            # efficiently (I think). Can also sum across the specified dims sequentially, but that
            # leads to complexity O(\sum_{i=1}^{len(dim)} log_2(self.size(dim[i]))) instead of
            # O(log_2(\sum_{i=1}^{len(dim)} self.size(dim[i]))), which is worse by Jensen's ineq.
            # So, we have to shift the dims around to get the specified dims to be sequential and
            # then flatten these together, which is annoying.
            raise NotImplementedError("TODO: Still have to implement this.")
        
        # Flatten the tensor to make things a bit easier
        if dim is None:
            fpe_tensor = self.flatten()
            chunk_dim = 0
        elif isinstance(dim, int):
            fpe_tensor = self
            chunk_dim = dim if dim >= 0 else self.dim() + dim

        # Deal with odd number of terms by taking the first term and adding to the sum of the rest
        if fpe_tensor.size(chunk_dim) % 2:
            residual = fpe_tensor.select(dim=chunk_dim, index=0)
            fpe_tensor = fpe_tensor.index_select(
                dim=chunk_dim,
                index=torch.arange(1, fpe_tensor.size(chunk_dim)),
            )
            fpe_tensor = fpe_tensor.sum(dim=chunk_dim) + residual

            if keepdim:
                fpe_tensor = fpe_tensor.unsqueeze(chunk_dim)

        # Deal with even number of terms by adding first half to second half and recursing
        else:
            chunks = fpe_tensor.chunk(chunks=2, dim=chunk_dim)
            fpe_tensor = chunks[0] + chunks[1]

            # Check if we are done. If we are, correct for keepdim; if not, recurse.
            if fpe_tensor.size(chunk_dim) == 1:
                if not keepdim:
                    fpe_tensor = fpe_tensor.squeeze(chunk_dim)
            else:
                return fpe_tensor.sum(dim=chunk_dim, keepdim=keepdim)

        return fpe_tensor
    
    def mean(self, dim: Optional[int | tuple[int]] = None, keepdim: bool = False) -> FPETensor:
        if isinstance(dim, int):
            return self.sum(dim, keepdim) / self.size(dim)
        else:
            return self.sum(dim, keepdim) / math.prod(d for d in self.size())
    
    def norm(
        self, p: int = 2, dim: Optional[int | tuple[int]] = None, keepdim: bool = False
    ) -> FPETensor:
        if p != 2:
            # This case required pow to be properly implemented (for most p)
            raise NotImplementedError("Not implemented")
        else:
            return self.pow(p).sum(dim=dim, keepdim=keepdim).sqrt()
    
    def pow(self, exponent: int) -> FPETensor:
        if not isinstance(exponent, int):
            raise NotImplementedError(
                "Still need to implement float and tensor exponents "
                "(can probably base off of usual math.h power algorithms)"
            )
        
        if exponent < 0:
            return 1 / self.pow(-exponent)
        elif exponent == 0:
            return 1
        elif (exponent % 2) == 0:
            half_exp = self.pow(exponent // 2)
            return half_exp * half_exp
        else:
            return self * self.pow(exponent - 1)

    def clamp_min(self, min_val: float | torch.Tensor | FPETensor) -> FPETensor:
        to_clamp = self < min_val
        min_val = self._convert_other(other=min_val)
        new_tensor = torch.where(to_clamp[..., None], min_val, self.tensor)
        return FPETensor.from_tensor(new_tensor)
    
    def clamp_max(self, max_val: float | torch.Tensor | FPETensor) -> FPETensor:
        raise NotImplementedError("TODO")
    
    def clamp(
        self,
        min: Optional[float | torch.Tensor | FPETensor] = None,
        max: Optional[float | torch.Tensor | FPETensor] = None,
    ) -> FPETensor:
        raise NotImplementedError("TODO")
    
    def log(self) -> FPETensor:
        new_tensor = torch.zeros_like(self.tensor)
        new_tensor[..., 0] = self.tensor[..., 0].log()
        return FPETensor.from_tensor(new_tensor)

    def atanh(self) -> FPETensor:
        # Compute sign and absolute value
        s = torch.sign(self.tensor[..., 0])
        xa = self * s

        # Initialize output tensor
        out = torch.empty_like(self.tensor)

        # Deal with inputs outside of the domain or on the boundary of the domain
        out = torch.where((xa > 1.0).unsqueeze(-1), float("nan"), out)
        out = torch.where((xa == 1.0).unsqueeze(-1), s.unsqueeze(-1) * float("inf"), out)

        # Deal with inputs inside of the domain (approximately t1: |x| < 0.5, t2: |x| >= 0.5)
        t1 = 0.5 * (1 + 2 * xa + 2 * xa * xa / (1.0 - xa)).log()
        t2 = 0.5 * (1 + 2 * xa / (1.0 - xa)).log()
        out = torch.where(
            condition=(xa.tensor[..., 0] < 0.5).unsqueeze(-1),
            input=t1.tensor,
            other=out,
        )
        out = torch.where(
            condition=torch.logical_and(0.5 <= xa, xa < 1.0).unsqueeze(-1),
            input=t2.tensor,
            other=out,
        )

        return FPETensor.from_tensor(out)
    
    def acosh(self) -> FPETensor:
        new_tensor = torch.zeros_like(self.tensor)
        new_tensor[..., 0] = self.tensor[..., 0].acosh()
        return FPETensor.from_tensor(new_tensor)
    
    def fill_diagonal_(self, fill_value: float) -> FPETensor:
        self.tensor[..., 0].fill_diagonal_(fill_value)
        for t in range(1, self.terms):
            self.tensor[..., t].fill_diagonal_(0.0)

        return self
    
    def _convert_other(self, other: Any) -> torch.Tensor:
        """Used for converting certain inputs to a unified tensor shape.
        """
        if isinstance(other, (int, float)):
            return torch.cat(
                tensors=(
                    other * torch.ones(*self.size(), 1),
                    torch.zeros(*self.size(), self.terms - 1),
                ),
                dim=-1,
            )
        elif isinstance(other, torch.Tensor):
            broadcasted_shape = torch.broadcast_shapes(self.size(), other.size())
            return torch.cat(
                tensors=(
                    other.broadcast_to(broadcasted_shape).unsqueeze(-1),
                    torch.zeros(*broadcasted_shape, self.terms - 1),
                ),
                dim=-1,
            )
        elif isinstance(other, FPETensor):
            return other.tensor
        else:
            return NotImplemented
