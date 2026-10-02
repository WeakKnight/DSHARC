#!/usr/bin/env bash
set -euo pipefail

sample_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
slangpy_dir="${SLANGPY_LOCAL_DIR:-$(dirname "$(dirname "$sample_dir")")/slangpy}"
export PYTHONPATH="$slangpy_dir${PYTHONPATH:+:$PYTHONPATH}"
exec "$sample_dir/.venv/bin/python" "$sample_dir/entry_point.py" --renderer dsharc "$@"
