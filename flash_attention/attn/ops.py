"""Registry of attention implementations.

Every implementation shares the same signature:

    fn(q, k, v, *, causal: bool = False) -> out

where q is shaped (batch, heads, q_seq_len, head_dim), k and v are shaped
(batch, heads, kv_seq_len, head_dim), and out has the same shape as q.
Softmax scaling is the standard 1/sqrt(head_dim). Causal masking follows
torch's is_causal convention: the triangle is aligned to the top-left corner.

The benchmark harness and the correctness tests iterate over this registry,
so future from-scratch implementations (Triton, CUDA) only need to be
registered here to get benchmarked and tested.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

AttentionFn = Callable[..., torch.Tensor]

IMPLEMENTATIONS: dict[str, AttentionFn] = {}


def register(name: str) -> Callable[[AttentionFn], AttentionFn]:
    def decorator(fn: AttentionFn) -> AttentionFn:
        assert name not in IMPLEMENTATIONS, f"duplicate implementation: {name}"
        IMPLEMENTATIONS[name] = fn
        return fn

    return decorator


@register("torch-sdpa")
def torch_sdpa(q, k, v, *, causal: bool = False) -> torch.Tensor:
    """torch SDPA with automatic backend selection."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def _sdpa_with_backend(backend: SDPBackend) -> AttentionFn:
    """torch SDPA pinned to a specific backend (raises if unsupported)."""

    def fn(q, k, v, *, causal: bool = False) -> torch.Tensor:
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    return fn


IMPLEMENTATIONS["torch-flash"] = _sdpa_with_backend(SDPBackend.FLASH_ATTENTION)
IMPLEMENTATIONS["torch-cudnn"] = _sdpa_with_backend(SDPBackend.CUDNN_ATTENTION)
IMPLEMENTATIONS["torch-mem-eff"] = _sdpa_with_backend(SDPBackend.EFFICIENT_ATTENTION)
IMPLEMENTATIONS["torch-math"] = _sdpa_with_backend(SDPBackend.MATH)


def reference_attention(q, k, v, *, causal: bool = False) -> torch.Tensor:
    """Naive O(seq_len^2)-memory attention in fp32. Ground truth for tests.

    Deliberately not registered: it OOMs at benchmark-scale sequence lengths.
    """
    out_dtype = q.dtype
    q, k, v = (t.float() for t in (q, k, v))
    scores = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    if causal:
        q_len, kv_len = q.shape[-2], k.shape[-2]
        # Rectangular tril == torch's top-left-aligned is_causal mask.
        mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    return (scores.softmax(dim=-1) @ v).to(out_dtype)
