# flash_attention

Re-implementing flash-attention from scratch (Triton, then CUDA) for learning
purposes, benchmarked against `torch.nn.functional.scaled_dot_product_attention`.

## Layout

```
attn/
  ops.py        # registry of attention implementations (torch SDPA backends today;
                # from-scratch Triton/CUDA kernels get registered here later)
  benchmark.py  # CLI benchmark harness built on triton.testing (fwd + bwd, TFLOP/s plots)
tests/
  test_ops.py   # every registered impl vs. a naive fp32 reference (fwd + bwd grads)
modal_run.py    # the ONLY file that knows about Modal; launches the above on a GPU
```

The boundary rule: `attn/` and `tests/` are plain PyTorch/Triton and run
directly on any CUDA machine. `modal_run.py` is a thin launcher that ships
them to a Modal GPU and copies results back.

## Setup

```sh
uv sync   # installs torch/matplotlib/pandas + modal/pytest (dev group)
modal setup   # once, to authenticate
```

## Running benchmarks on Modal

```sh
uv run modal run modal_run.py::bench
uv run modal run modal_run.py::bench --args "--modes fwd --causal true --dtype fp16"
MODAL_GPU=A10G uv run modal run modal_run.py::bench   # default GPU is H100
```

Plots (PNG), CSVs, and `results.html` land in `bench_out/` locally.
`--args` is passed straight through to the benchmark CLI; see all knobs with:

```sh
uv run python -m attn.benchmark --help
```

## Running tests on Modal

```sh
uv run modal run modal_run.py::tests
uv run modal run modal_run.py::tests --args "-k causal -x"
```

## Running directly on a CUDA box (no Modal)

```sh
python -m attn.benchmark
pytest tests/
```

## Adding a new implementation

Register it in `attn/ops.py`:

```python
@register("triton-v1")
def triton_attention(q, k, v, *, causal=False): ...
```

It then shows up in `--providers` for the benchmark and is automatically
covered by the forward/backward correctness tests.
