"""Modal launch scripts for the benchmark harness and GPU tests.

This is the only file that imports Modal. Everything under attn/ and tests/
is plain PyTorch/Triton code that runs on any CUDA box; this file just ships
it to a Modal GPU, runs the same entry points, and copies results back.

Usage (from flash_attention/):

    modal run modal_run.py::bench
    modal run modal_run.py::bench --args "--modes fwd --causal true"
    modal run modal_run.py::tests
    modal run modal_run.py::tests --args "-k causal -x"

Pick a GPU with the MODAL_GPU env var (default H100):

    MODAL_GPU=A10G modal run modal_run.py::bench
"""

import io
import os
import shlex
import tarfile
from pathlib import Path

import modal

GPU = os.environ.get("MODAL_GPU", "H100")
REMOTE_OUT_DIR = "/root/bench_out"

app = modal.App("flash-attention-lab")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.7", "matplotlib>=3.9", "pandas>=2.2", "pytest>=8.0")
    # Mounted at runtime (not baked into the image), so code edits don't
    # trigger an image rebuild.
    .add_local_python_source("attn")
    .add_local_dir("tests", remote_path="/root/tests")
)


@app.function(image=image, gpu=GPU, timeout=3600)
def _bench_remote(argv: list[str]) -> bytes:
    from attn import benchmark

    benchmark.run(argv + ["--out-dir", REMOTE_OUT_DIR])

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(REMOTE_OUT_DIR, arcname=".")
    return buf.getvalue()


@app.function(image=image, gpu=GPU, timeout=3600)
def _tests_remote(argv: list[str]) -> int:
    import pytest

    return pytest.main(["/root/tests", *argv])


@app.local_entrypoint()
def bench(args: str = "", out_dir: str = "bench_out"):
    """Run the benchmark on a Modal GPU and download plots/CSVs to out_dir."""
    results = _bench_remote.remote(shlex.split(args))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(results), mode="r:gz") as tar:
        tar.extractall(out, filter="data")
    print(f"results written to {out}/")


@app.local_entrypoint()
def tests(args: str = ""):
    """Run the pytest suite on a Modal GPU."""
    exit_code = _tests_remote.remote(shlex.split(args))
    if exit_code != 0:
        raise SystemExit(exit_code)
