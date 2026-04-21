#!/usr/bin/env bash
# Uninstall CytoTrack AI desktop integration (not the project files).
set -e
APP_ID="cytotrack-ai"
APPS_DIR="${HOME}/.local/share/applications"
ICON_DIR="${HOME}/.local/share/icons/hicolor"
DESKTOP_DIR="${HOME}/Desktop"

rm -f "${APPS_DIR}/${APP_ID}.desktop"
rm -f "${DESKTOP_DIR}/${APP_ID}.desktop"
for size in 16 24 32 48 64 128 256 512; do
    rm -f "${ICON_DIR}/${size}x${size}/apps/${APP_ID}.png"
done
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPS_DIR" 2>/dev/null || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -f -t "${ICON_DIR}" 2>/dev/null || true
fi
echo "CytoTrack AI desktop integration removed."
echo "To also remove the virtualenv: rm -rf cell_track_venv"
