"""Run the sdpa_triton pytest suite on a Modal GPU container.

The tests themselves (tests/test_sdpa_triton.py) have no Modal awareness --
this file just ships the source + tests into a GPU container and invokes
pytest.
"""

import modal

GPU = "H100"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "pytest")
    .add_local_python_source("sdpa_triton")
    .add_local_dir("tests", "/root/tests")
    .add_local_file("pytest.ini", "/root/pytest.ini")
)

app = modal.App("test-sdpa-triton", image=image)


@app.function(gpu=GPU)
def run_pytest() -> int:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-v"],
        cwd="/root",
    )
    return result.returncode


@app.local_entrypoint()
def main():
    rc = run_pytest.remote()
    if rc != 0:
        raise SystemExit(rc)
