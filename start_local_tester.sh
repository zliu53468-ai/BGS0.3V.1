#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
printf 'Open http://127.0.0.1:8765 in your browser.\n'
python -m uvicorn local_road_test_app:app --host 127.0.0.1 --port 8765
