"""
fast bruteforce gaussian blue noise, based on a flash attention hack. Available only:
-on GPU (maybe not even all gpu)
-if dimension D >= 4, 

see run_bruteforce otherwise
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import time

from ..progress import _LevelCtx


def _make_hyperparams(N: int, D: int, device: torch.device):
    DX     = 1.0 / N ** (1.0 / D)
    S      = 1.0
    sigma2 = S * 2.0 * DX ** 2

    if D <= 2:   lr = 0.4
    elif D == 3: lr = 0.1
    elif D == 4:   lr = 0.05
    elif D == 5: lr = 0.02
    else:        lr = 0.01

    a_ = torch.tensor(2.0 * math.pi, device=device)
    b_ = torch.tensor(2.0 / (sigma2 * (2.0 * math.pi) ** 2), device=device)
    c_ = torch.tensor(1.0 / (2.0 * S * math.pi), device=device)

    return sigma2, lr, a_, b_, c_


def _unnorm_attn(
    Q: torch.Tensor,   # (N, dk)
    K: torch.Tensor,   # (N, dk)
    V: torch.Tensor,   # (N, dv)
) -> torch.Tensor:     # (N, dv)
    """
    sum_j V_j * exp(Q_i · K_j) = exp(lse) * softmax_out
    """
    N, dk = Q.shape
    dv = V.shape[1]

    # Pad Q, K et V de la même façon (head_dim multiple de 8)
    # → comportement identique + pas de crash CUDA sur dk=4/6
    pad_d = (8 - dk % 8) % 8
    if pad_d > 0:
        Q = F.pad(Q, (0, pad_d))
        K = F.pad(K, (0, pad_d))
        V = F.pad(V, (0, pad_d))   # ← important

    Q4 = Q.unsqueeze(0).unsqueeze(0).contiguous()  # (1,1,N,dk_pad)
    K4 = K.unsqueeze(0).unsqueeze(0).contiguous()
    V4 = V.unsqueeze(0).unsqueeze(0).contiguous()

    def _finalize(out, lse):
        # out / lse peuvent être paddés en séquence par le kernel
        out = out.squeeze(0).squeeze(0)[:N, :dv]   # (N, dv)  ← on coupe aussi les features
        lse = lse.squeeze(0).squeeze(0)[:N]         # (N,)
        return torch.exp(lse).unsqueeze(-1) * out

    # 1) Flash (Ampere+)
    try:
        out, lse, *rest = torch.ops.aten._scaled_dot_product_flash_attention(
            Q4, K4, V4,
            dropout_p=0.0,
            is_causal=False,
            return_debug_mask=False,
            scale=1.0,
        )
        return _finalize(out, lse)
    except RuntimeError:
        pass

    # 2) Efficient Attention (Turing / Volta / …)
    out, lse, *rest = torch.ops.aten._scaled_dot_product_efficient_attention(
        Q4, K4, V4,
        attn_bias=None,
        compute_log_sumexp=True,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0,
    )
    return _finalize(out, lse)


def grad_flash(
    x:  torch.Tensor,
    a_: torch.Tensor,
    b_: torch.Tensor,
    c_: torch.Tensor,
) -> torch.Tensor:
    N, D = x.shape

    ax     = a_ * x
    cos_ax = torch.cos(ax)
    sin_ax = torch.sin(ax)

    phi = torch.cat([cos_ax, sin_ax], dim=-1) * b_.sqrt()   # (N, 2D)

    # Un seul appel : V = [sin | cos]
    V = torch.cat([sin_ax, cos_ax], dim=-1)                 # (N, 2D)
    attn = _unnorm_attn(phi, phi, V)                        # (N, 2D)

    attn_sin = attn[:, :D]
    attn_cos = attn[:, D:]

    grad = cos_ax * attn_sin - sin_ax * attn_cos
    prefactor = c_ * torch.exp(-b_ * D)
    return prefactor * grad


def _flash_pipeline(N: int, D: int, n_iter: int, lr: float = 1.0, ctx: _LevelCtx | None = None):
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    assert D >= 4, (
        "gaussian flash unavailable in dimension D < 4, because it would lead to exponential overflows (unstable exp)"
    )

    assert torch._preload_cuda_deps
    _, _lr, a_, b_, c_ = _make_hyperparams(N, D, device)
    lr = lr * _lr


    def sample(init: np.ndarray | None = None) -> np.ndarray:
        if init is None:
            init = np.random.rand(N, D).astype(np.float32)

        x = torch.tensor(init, dtype=torch.float16, device=device)

        if ctx is not None: #just time profiling (optional)
            start = time.time
            for _ in range(3):
                g = grad_flash(x, a_, b_, c_)
            ctx.on_bruteforce_start(eta_seconds = (time.time - start)/3)

        for _ in range(n_iter):
            g = grad_flash(x, a_, b_, c_)
            x = (x - lr * g) % 1.0

        if ctx is not None: 
            ctx.on_bruteforce_done()

        return x.cpu().numpy().astype(np.float32)

    return sample