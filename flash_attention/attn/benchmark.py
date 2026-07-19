"""Benchmark harness for attention implementations.

Times the forward and backward passes of every requested implementation with
triton.testing.do_bench and renders TFLOP/s-vs-seq_len plots (PNG + CSV +
results.html) via triton.testing.perf_report.

CUDA-only, but has no idea Modal exists — run it directly on any GPU box:

    python -m attn.benchmark --modes fwd bwd --causal both

or through Modal via ../modal_run.py.
"""

from __future__ import annotations

import argparse
import itertools
import os

import torch
import triton.testing

from .ops import IMPLEMENTATIONS

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

DEFAULT_PROVIDERS = ["torch-flash", "torch-cudnn", "torch-mem-eff"]
DEFAULT_KV_SEQ_LENS = [2**i for i in range(9, 15)]  # 512 .. 16384


def attention_flops(
    *,
    batch: int,
    heads: int,
    q_seq_len: int,
    kv_seq_len: int,
    head_dim: int,
    causal: bool,
    mode: str,
) -> float:
    """Matmul FLOPs for one attention call.

    Forward: 2 matmuls (QK^T and PV), 2 FLOPs per MAC each, over the attended
    (query, key) pairs. torch's is_causal aligns the triangle to the top-left
    corner, which is what the causal pair count assumes.
    Backward: 5 matmuls, i.e. 2.5x the forward (standard flash-attention
    accounting; recomputation inside the kernel is not counted).
    """
    if causal:
        pairs = sum(min(i + 1, kv_seq_len) for i in range(q_seq_len))
    else:
        pairs = q_seq_len * kv_seq_len
    flops = 4.0 * batch * heads * pairs * head_dim
    if mode == "bwd":
        flops *= 2.5
    return flops


def _make_configs(args: argparse.Namespace) -> list[triton.testing.Benchmark]:
    causal_vals = {"true": [True], "false": [False], "both": [False, True]}[args.causal]
    ylabels = {"tflops": "TFLOP/s", "mem": "peak memory (GB)"}
    return [
        triton.testing.Benchmark(
            x_names=["kv_seq_len"],
            x_vals=args.kv_seq_lens,
            line_arg="provider",
            line_vals=args.providers,
            line_names=args.providers,
            x_log=True,
            ylabel=ylabels[metric],
            plot_name=(
                f"attention-{mode}-{metric}-causal_{causal}-b{args.batch}-h{args.heads}"
                f"-q{args.q_seq_len}-d{args.head_dim}-{args.dtype}"
            ),
            args={"mode": mode, "causal": causal, "metric": metric},
        )
        for mode, causal, metric in itertools.product(args.modes, causal_vals, args.metrics)
    ]


def _make_bench_fn(args: argparse.Namespace):
    def bench(kv_seq_len: int, provider: str, mode: str, causal: bool, metric: str) -> float:
        dtype = DTYPES[args.dtype]

        def rand(seq_len: int) -> torch.Tensor:
            shape = (args.batch, args.heads, seq_len, args.head_dim)
            return torch.randn(
                shape, device="cuda", dtype=dtype, requires_grad=(mode == "bwd")
            )

        q, k, v = rand(args.q_seq_len), rand(kv_seq_len), rand(kv_seq_len)
        impl = IMPLEMENTATIONS[provider]
        try:
            # Reset before the forward pass so that for bwd mode the peak spans
            # the whole fwd+bwd (saved activations included), not just the
            # backward call.
            if metric == "mem":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            if mode == "fwd":
                fn = lambda: impl(q, k, v, causal=causal)
            else:
                out = impl(q, k, v, causal=causal)
                grad = torch.randn_like(out)
                fn = lambda: out.backward(grad, retain_graph=True)
            if metric == "mem":
                fn()
                torch.cuda.synchronize()
                return torch.cuda.max_memory_allocated() / 2**30
            ms = triton.testing.do_bench(fn)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            # Unsupported backend / OOM at this size: leave a gap in the plot.
            print(f"[skip] {provider} {mode} {metric} kv_seq_len={kv_seq_len}: {e}")
            return float("nan")
        flops = attention_flops(
            batch=args.batch,
            heads=args.heads,
            q_seq_len=args.q_seq_len,
            kv_seq_len=kv_seq_len,
            head_dim=args.head_dim,
            causal=causal,
            mode=mode,
        )
        return flops * 1e-12 / (ms * 1e-3)

    return bench


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=sorted(IMPLEMENTATIONS),
        default=DEFAULT_PROVIDERS,
        help="Attention implementations to benchmark.",
    )
    parser.add_argument("--modes", nargs="+", choices=["fwd", "bwd"], default=["fwd", "bwd"])
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["tflops", "mem"],
        default=["tflops", "mem"],
        help="What to plot: throughput and/or peak GPU memory.",
    )
    parser.add_argument("--causal", choices=["true", "false", "both"], default="false")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=96)
    parser.add_argument("--q-seq-len", type=int, default=32)
    parser.add_argument(
        "--kv-seq-lens",
        nargs="+",
        type=int,
        default=DEFAULT_KV_SEQ_LENS,
        help="k/v sequence lengths to sweep (the plot's x-axis).",
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--out-dir", default="bench_out", help="Where to save plots and CSVs.")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    # triton's perf_report assumes save_path already exists.
    os.makedirs(args.out_dir, exist_ok=True)
    assert torch.cuda.is_available(), "benchmark requires a CUDA device"
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"torch:  {torch.__version__}")
    report = triton.testing.perf_report(_make_configs(args))(_make_bench_fn(args))
    report.run(print_data=True, save_path=args.out_dir)
    print(f"saved plots and CSVs to {args.out_dir}/")


if __name__ == "__main__":
    run()
