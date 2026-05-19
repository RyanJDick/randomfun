"""Correctness tests for sdpa_triton vs torch.nn.functional.scaled_dot_product_attention."""

import pytest
import torch

from sdpa_triton import triton_sdpa


@pytest.fixture(autouse=True)
def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _reference_sdpa(q, k, v):
    # Math backend gives a deterministic high-precision reference.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "B,H,N,D",
    [
        (1, 1, 128, 64),
        (2, 4, 256, 64),
        (1, 2, 512, 128),
        (4, 8, 1024, 64),
        (1, 1, 256, 256),
    ],
)
def test_parity_self_attn(B, H, N, D, dtype):
    torch.manual_seed(0)
    q = torch.randn((B, H, N, D), device="cuda", dtype=dtype)
    k = torch.randn((B, H, N, D), device="cuda", dtype=dtype)
    v = torch.randn((B, H, N, D), device="cuda", dtype=dtype)

    out = triton_sdpa(q, k, v)
    ref = _reference_sdpa(q, k, v)

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "N_Q,N_K",
    [
        (128, 256),
        (256, 128),
        (300, 200),  # non-multiples of BLOCK_{M,N}
        (200, 300),
        (1, 512),  # single-query (decode-shaped)
    ],
)
def test_parity_cross_attn_shapes(N_Q, N_K):
    B, H, D = 2, 4, 64
    dtype = torch.bfloat16
    torch.manual_seed(0)
    q = torch.randn((B, H, N_Q, D), device="cuda", dtype=dtype)
    k = torch.randn((B, H, N_K, D), device="cuda", dtype=dtype)
    v = torch.randn((B, H, N_K, D), device="cuda", dtype=dtype)

    out = triton_sdpa(q, k, v)
    ref = _reference_sdpa(q, k, v)

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_parity_matches_benchmark_shape():
    # Mirrors the shape used in profile_sdpa.py so the benchmark is exercising
    # a code path we've actually validated.
    B, H, N, D = 4, 1, 1024, 256
    dtype = torch.bfloat16
    torch.manual_seed(0)
    q = torch.randn((B, H, N, D), device="cuda", dtype=dtype)
    k = torch.randn((B, H, N, D), device="cuda", dtype=dtype)
    v = torch.randn((B, H, N, D), device="cuda", dtype=dtype)

    out = triton_sdpa(q, k, v)
    ref = _reference_sdpa(q, k, v)

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
