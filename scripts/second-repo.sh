#!/bin/sh
# A second repository's view of loomcore: build the wheel, install only it into a
# throwaway environment, and run evals/second_repo.py there with `python -I`, so
# nothing from this checkout can be imported by accident.
set -eu
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
uv build --wheel --out-dir "$dir" --quiet
uv run --isolated --no-project --quiet --with "$dir"/loomcore-*.whl python -I evals/second_repo.py
