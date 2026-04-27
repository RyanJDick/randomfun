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


@app.function(gpu=GPU)
def profile_shape(
    batch: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    warmup_iters: int = WARMUP_ITERS,
    timing_iters: int = TIMING_ITERS,
) -> dict:
    import torch

    device = torch.device("cuda")
    dtype = torch.bfloat16  # flash-attn requires fp16/bf16

    shape = (batch, num_heads, seq_len, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        for _ in range(warmup_iters):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(timing_iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(timing_iters)]

        for i in range(timing_iters):
            starts[i].record()
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            ends[i].record()
        torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]

    import numpy as np

    arr = np.array(times_ms)
    return {
        "shape": shape,
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "iters": timing_iters,
    }


@app.local_entrypoint()
def main():
    print(f"{'shape':<32} {'mean (ms)':>12} {'std (ms)':>12} {'min (ms)':>12}")
    print("-" * 72)
    for shape in SHAPES:
        result = profile_shape.remote(*shape)
        s = str(result["shape"])
        print(
            f"{s:<32} {result['mean_ms']:>12.4f} {result['std_ms']:>12.4f} "
            f"{result['min_ms']:>12.4f}"
        )
