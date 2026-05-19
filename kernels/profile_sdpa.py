"""Profile torch.nn.functional.scaled_dot_product_attention on Modal."""

import modal

GPU = "H100"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy", "matplotlib", "pandas")
    .add_local_python_source("sdpa_triton")
)

app = modal.App("profile-sdpa", image=image)


# Fixed dims for the sweep; only sequence length varies.
# TODO(ryand): Figure out why things fail with larger batch sizes.
BATCH = 1024
NUM_HEADS = 1
HEAD_DIM = 256
SEQ_LENS = [256 * i for i in range(1, 17)]  # 256 .. 4096


with image.imports():
    import torch
    import triton

    from sdpa_triton import triton_sdpa


@app.function(gpu=GPU)
def run_benchmark(
    batch: int = BATCH,
    num_heads: int = NUM_HEADS,
    head_dim: int = HEAD_DIM,
    seq_lens: list[int] = SEQ_LENS,
) -> dict[str, bytes]:
    import os

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=list(seq_lens),
            line_arg="provider",
            line_vals=["torch", "triton"],
            line_names=["Torch (FLASH_ATTENTION)", "Triton"],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="ms",
            plot_name="sdpa-vs-seqlen",
            args={"B": batch, "H": num_heads, "D": head_dim},
        )
    )
    def benchmark(B, H, N, D, provider):
        q = torch.randn((B, H, N, D), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((B, H, N, D), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((B, H, N, D), device="cuda", dtype=torch.bfloat16)
        quantiles = [0.5, 0.2, 0.8]
        if provider == "torch":
            with torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.FLASH_ATTENTION
            ):
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v),
                    quantiles=quantiles,
                )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_sdpa(q, k, v), quantiles=quantiles
            )
        return ms, max_ms, min_ms

    save_path = "/tmp/sdpa_bench"
    os.makedirs(save_path, exist_ok=True)
    benchmark.run(show_plots=False, print_data=True, save_path=save_path)

    out: dict[str, bytes] = {}
    for name in os.listdir(save_path):
        with open(os.path.join(save_path, name), "rb") as f:
            out[name] = f.read()
    return out


@app.local_entrypoint()
def main():
    import pathlib

    out_dir = pathlib.Path("bench_out")
    out_dir.mkdir(exist_ok=True)
    files = run_benchmark.remote()
    for name, content in files.items():
        path = out_dir / name
        path.write_bytes(content)
        print(f"wrote {path}")
