#!/usr/bin/env bash
# Use the existing environment to generate the DNA and PhyloLM table.
set -euo pipefail

relationship_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
relationship_repo_root="$(cd -- "$relationship_script_dir/.." && pwd)"

if [[ -n "${REPRO_PYTHON:-}" ]]; then
    relationship_python="$REPRO_PYTHON"
elif [[ -x "$relationship_repo_root/.venv/bin/python" ]]; then
    relationship_python="$relationship_repo_root/.venv/bin/python"
else
    relationship_python=python3
fi

export PYTHONDONTWRITEBYTECODE=1
exec "$relationship_python" "$relationship_script_dir/reproduce_relationship_table.py" "$@"
