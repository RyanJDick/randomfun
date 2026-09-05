#!/usr/bin/env bash
# Format all files with Ruff, matching the editor's format-on-save behavior
# (ruff format + organize imports). Pass --check to verify without writing,
# e.g. in CI:  ./format.sh --check
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--check" ]]; then
    uv run ruff format --check .
    uv run ruff check --select I .
else
    uv run ruff format .
    uv run ruff check --select I --fix .
fi
