#!/bin/sh
# Provision a session: sync the environment, run the gates, show the next unfinished feature.
set -eu
uv sync --locked
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy src tests evals bench
uv run --no-sync pytest
uv run --no-sync pytest evals
uv run --no-sync pytest bench
./scripts/second-repo.sh
gitleaks git --redact --no-banner
python3 - <<'PY'
import json
todo = [f for f in json.load(open("feature_list.json")) if not f["passes"]]
print("next:", f"{todo[0]['id']} (card {todo[0]['card']}): {todo[0]['title']}" if todo else "none, all features pass")
PY
