"""Profile torch.nn.functional.scaled_dot_product_attention on Modal."""

import modal

GPU = "H100"

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.5.1", "numpy"
)

app = modal.App("profile-sdpa", image=image)


# (batch, num_heads, seq_len, head_dim)
SHAPES = [
    (2048, 1, 200, 64),
    (2048, 1, 200, 256),
    (2048, 1, 3000, 256),
    (2048, 8, 3000, 32),
]

WARMUP_ITERS = 5
TIMING_ITERS = 20


with image.imports():
    import numpy as np
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _sdpa_fwd_kernel(
        Q, K, V, Out, sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        N_CTX_Q, N_CTX_K, H,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        off_bh = tl.program_id(1)
        off_b = off_bh // H
        off_h = off_bh % H

        Q_ptr = Q + off_b * stride_qb + off_h * stride_qh
        K_ptr = K + off_b * stride_kb + off_h * stride_kh
        V_ptr = V + off_b * stride_vb + off_h * stride_vh
        O_ptr = Out + off_b * stride_ob + off_h * stride_oh

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, BLOCK_N)

        q = tl.load(
            Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=offs_m[:, None] < N_CTX_Q,
            other=0.0,
        )

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # Scale by 1/ln(2) so we can use exp2 instead of exp.
        qk_scale = sm_scale * 1.44269504

        for start_n in range(0, N_CTX_K, BLOCK_N):
            n_offs = start_n + offs_n
            kv_mask = n_offs[:, None] < N_CTX_K
            k = tl.load(
                K_ptr + n_offs[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=kv_mask,
                other=0.0,
            )
            v = tl.load(
                V_ptr + n_offs[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=kv_mask,
                other=0.0,
            )
            qk = tl.dot(q, tl.trans(k)) * qk_scale
            qk = tl.where(n_offs[None, :] < N_CTX_K, qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + l_ij
            m_i = m_ij

        acc = acc / l_i[:, None]
        tl.store(
            O_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
            acc.to(Out.type.element_ty),
            mask=offs_m[:, None] < N_CTX_Q,
        )

    def triton_sdpa(q, k, v):
        # q, k, v: (B, H, N, D), bf16/fp16. D must be a power of 2.
        B, H, N_Q, D = q.shape
        N_K = k.shape[2]
        sm_scale = 1.0 / (D ** 0.5)

        out = torch.empty_like(q)

        BLOCK_M = 64 if D <= 128 else 32
        BLOCK_N = 64 if D <= 128 else 32

        grid = (triton.cdiv(N_Q, BLOCK_M), B * H)

        _sdpa_fwd_kernel[grid](
            q, k, v, out, sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            N_Q, N_K, H,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D,
        )
        return out


def _bench(fn, warmup_iters: int, timing_iters: int) -> dict:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(timing_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(timing_iters)]
    for i in range(timing_iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times_ms = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return {
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std()),
        "min_ms": float(times_ms.min()),
        "max_ms": float(times_ms.max()),
    }


@app.function(gpu=GPU)
def profile_shape(
    batch: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    warmup_iters: int = WARMUP_ITERS,
    timing_iters: int = TIMING_ITERS,
) -> dict:
    device = torch.device("cuda")
    dtype = torch.bfloat16  # flash-attn requires fp16/bf16

    shape = (batch, num_heads, seq_len, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)

    def torch_fn():
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    def triton_fn():
        return triton_sdpa(q, k, v)

    o_torch = torch_fn()
    o_triton = triton_fn()
    max_abs_err = (o_torch.float() - o_triton.float()).abs().max().item()

    return {
        "shape": shape,
        "torch": _bench(torch_fn, warmup_iters, timing_iters),
        "triton": _bench(triton_fn, warmup_iters, timing_iters),
        "max_abs_err": max_abs_err,
    }


@app.local_entrypoint()
def main():
    header = (
        f"{'shape':<28} {'torch (ms)':>18} {'triton (ms)':>18} "
        f"{'speedup':>9} {'max err':>10}"
    )
    print(header)
    print("-" * len(header))
    for shape in SHAPES:
        r = profile_shape.remote(*shape)
        t_mean, t_std = r["torch"]["mean_ms"], r["torch"]["std_ms"]
        tr_mean, tr_std = r["triton"]["mean_ms"], r["triton"]["std_ms"]
        print(
            f"{str(r['shape']):<28} "
            f"{t_mean:>10.4f} ± {t_std:>5.2f}  "
            f"{tr_mean:>10.4f} ± {tr_std:>5.2f}  "
            f"{t_mean / tr_mean:>7.2f}x  "
            f"{r['max_abs_err']:>10.4f}"
        )
