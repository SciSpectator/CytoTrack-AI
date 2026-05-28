#!/usr/bin/env bash
# Minimal source setup for CytoTrack AI.
# For a full system integration (icon in Applications menu, desktop shortcut)
# run ./install_linux.sh instead.
set -e
cd "$(dirname "$0")"

echo "CytoTrack AI v1.0 - Source Setup"
echo "-------------------------------"

python3 --version >/dev/null || { echo "Python 3 required"; exit 1; }

if command -v apt >/dev/null 2>&1; then
    echo "Installing system packages (sudo)..."
    sudo apt update
    sudo apt install -y python3-venv python3-pip libgl1-mesa-glx \
        libglib2.0-0 libxcb-xinerama0
fi

VENV=cell_track_venv
rm -rf "$VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"

pip install --upgrade pip wheel
pip install -r requirements.txt

# Optional NVIDIA LocateAnything-3B detection backend (default detector).
# NVIDIA non-commercial license; ~3B-param model, GPU recommended. The app
# falls back to the classical detector if this fails. Set
# CYTOTRACK_SKIP_LOCATE=1 to skip.
if [ -z "$CYTOTRACK_SKIP_LOCATE" ]; then
    echo "Installing optional LocateAnything-3B backend (NVIDIA non-commercial license)..."
    pip install -r requirements-locate.txt || \
        echo "LocateAnything deps not installed; the app will use the classical detector."
fi

chmod +x launch.sh
echo ""
echo "Done. Run ./launch.sh to start CytoTrack AI."
