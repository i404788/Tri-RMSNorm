from __future__ import annotations

from .version import VERSION, VERSION_SHORT

from tri_rmsnorm.kernel.rms_normalization_kernel import (
    _rms_norm_fwd_fused,
    _rms_norm_bwd_dx_fused,
    _rms_norm_bwd_dwdb,
)

import torch
import numbers

class RMSNormFunctionKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        M, N = x.shape
        y = torch.empty_like(x)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        _rms_norm_fwd_fused[(M,)](x, y, weight, bias, rstd, x.stride(0), N, eps, BLOCK_SIZE=1024)
        ctx.save_for_backward(x, weight, bias, rstd)
        ctx.eps = eps
        ctx.N = N
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias, rstd = ctx.saved_tensors
        eps = ctx.eps
        N = ctx.N
        M = x.shape[0]
        dx = torch.empty_like(x)
        _dw = torch.empty_like(weight)
        _db = torch.empty_like(bias)
        locks = torch.zeros(2 * 32, dtype=torch.int32, device=x.device)
        _rms_norm_bwd_dx_fused[(M,)](dx, dy, _dw, _db, x, weight, bias, rstd, locks, x.stride(0), N, eps, GROUP_SIZE_M=32, BLOCK_SIZE_N=1024)
        return dx, _dw, _db, None


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, bias = True, eps=1e-5, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))

            self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.init.ones_(self.weight)
        if self.bias is not None:
            torch.init.zeros_(self.bias)

    def forward(self, input):
        return RMSNormFunctionCustomKernel.apply(input, self.weight, self.bias, self.eps)



__all__ = [
    "VERSION",
    "VERSION_SHORT",
    "_rms_norm_fwd_fused",
    "_rms_norm_bwd_dx_fused",
    "_rms_norm_bwd_dwdb",
    "RMSNormFunctionKernel",
    "RMSNorm"
]
