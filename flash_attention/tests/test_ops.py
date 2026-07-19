"""Correctness tests: every registered implementation vs. the naive fp32 reference.

New Triton/CUDA implementations registered in attn.ops are picked up
automatically. Run on GPU via ../modal_run.py::tests.
"""

import pytest
import torch

from attn.ops import IMPLEMENTATIONS, reference_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# (batch, heads, q_seq_len, kv_seq_len, head_dim)
SHAPES = [
    (2, 4, 128, 128, 64),
    (1, 8, 1024, 1024, 64),
    (2, 4, 500, 500, 128),  # seq_len not a multiple of typical tile sizes
    (4, 1, 32, 1024, 96),  # cross-attention: short q, long k/v
]


def _make_qkv(shape, *, requires_grad=False):
    batch, heads, q_seq_len, kv_seq_len, head_dim = shape
    torch.manual_seed(0)

    def rand(seq_len):
        return torch.randn(
            (batch, heads, seq_len, head_dim),
            device="cuda",
            dtype=torch.float16,
            requires_grad=requires_grad,
        )

    return rand(q_seq_len), rand(kv_seq_len), rand(kv_seq_len)


def _run_or_skip(impl, q, k, v, *, causal):
    try:
        return impl(q, k, v, causal=causal)
    except RuntimeError as e:
        pytest.skip(f"backend unsupported here: {e}")


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_forward_matches_reference(name, causal, shape):
    q, k, v = _make_qkv(shape)
    out = _run_or_skip(IMPLEMENTATIONS[name], q, k, v, causal=causal)
    expected = reference_attention(q, k, v, causal=causal)
    torch.testing.assert_close(out, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize(
    "shape", [(2, 4, 256, 256, 64), (4, 1, 32, 512, 96)], ids=["self", "cross"]
)
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_backward_matches_reference(name, causal, shape):
    q, k, v = _make_qkv(shape, requires_grad=True)
    out = _run_or_skip(IMPLEMENTATIONS[name], q, k, v, causal=causal)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    q_ref, k_ref, v_ref = (t.detach().clone().requires_grad_() for t in (q, k, v))
    reference_attention(q_ref, k_ref, v_ref, causal=causal).backward(grad_out)

    for got, want, label in [
        (q.grad, q_ref.grad, "dq"),
        (k.grad, k_ref.grad, "dk"),
        (v.grad, v_ref.grad, "dv"),
    ]:
        torch.testing.assert_close(got, want, atol=5e-3, rtol=5e-3, msg=lambda m: f"{label}: {m}")
