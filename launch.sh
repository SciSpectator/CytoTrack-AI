#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ -d "cell_track_venv" ]; then
  source cell_track_venv/bin/activate
fi
exec python3 main.py "$@"
