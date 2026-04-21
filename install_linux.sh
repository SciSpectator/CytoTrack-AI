#!/usr/bin/env bash
# --------------------------------------------------------------
# CytoTrack AI - Ubuntu / Linux installer
# --------------------------------------------------------------
# Creates a virtualenv, installs dependencies, registers the app
# with the Applications menu, and places a launcher on the Desktop.
# --------------------------------------------------------------
set -e

APP_NAME="CytoTrack AI"
APP_ID="cytotrack-ai"
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${HERE}/cell_track_venv"
ICON_SRC="${HERE}/assets/icon.png"
APPS_DIR="${HOME}/.local/share/applications"
ICON_DIR="${HOME}/.local/share/icons/hicolor"
DESKTOP_DIR="${HOME}/Desktop"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${CYAN}==============================================${NC}"
echo -e "${CYAN}   ${APP_NAME} - Ubuntu/Linux installation   ${NC}"
echo -e "${CYAN}==============================================${NC}"

# 1) Check python
if ! command -v python3 >/dev/null; then
    echo "python3 not found - please install it first (sudo apt install python3 python3-venv)."
    exit 1
fi

# 2) Ensure system dependencies for Qt + OpenCV
need_pkgs=()
for pkg in python3-venv python3-pip libxcb-xinerama0; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
        need_pkgs+=("$pkg")
    fi
done
if [ ${#need_pkgs[@]} -gt 0 ]; then
    echo -e "${YELLOW}Installing system packages: ${need_pkgs[*]}${NC}"
    sudo apt-get update
    sudo apt-get install -y "${need_pkgs[@]}"
fi

# 3) Build virtualenv
if [ ! -d "$VENV" ]; then
    echo -e "${CYAN}Creating virtualenv at ${VENV}${NC}"
    python3 -m venv "$VENV"
fi

# shellcheck source=/dev/null
source "${VENV}/bin/activate"
pip install --upgrade pip
pip install -r "${HERE}/requirements.txt"

# 4) Generate the app icon if it's missing
if [ ! -f "$ICON_SRC" ]; then
    echo -e "${CYAN}Generating application icon...${NC}"
    python3 "${HERE}/packaging/generate_icon.py"
fi

# 5) Install icons into hicolor theme
for size in 16 24 32 48 64 128 256 512; do
    src="${HERE}/assets/icon_${size}.png"
    if [ -f "$src" ]; then
        dst="${ICON_DIR}/${size}x${size}/apps"
        mkdir -p "$dst"
        cp -f "$src" "${dst}/${APP_ID}.png"
    fi
done
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -f -t "${ICON_DIR}" 2>/dev/null || true
fi

# 6) Launcher script (bounces into the venv)
LAUNCHER="${HERE}/launch.sh"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
cd "${HERE}"
source "${VENV}/bin/activate"
exec python3 "${HERE}/main.py" "\$@"
EOF
chmod +x "$LAUNCHER"

# 7) Render the .desktop file
mkdir -p "$APPS_DIR"
DESKTOP_FILE="${APPS_DIR}/${APP_ID}.desktop"
sed -e "s|@EXEC@|${LAUNCHER}|g" \
    -e "s|@ICON@|${APP_ID}|g" \
    "${HERE}/packaging/cytotrack-ai.desktop.in" > "$DESKTOP_FILE"
chmod +x "$DESKTOP_FILE"
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPS_DIR" 2>/dev/null || true
fi

# 8) Desktop shortcut (if Desktop exists)
if [ -d "$DESKTOP_DIR" ]; then
    DESK_ENTRY="${DESKTOP_DIR}/${APP_ID}.desktop"
    cp -f "$DESKTOP_FILE" "$DESK_ENTRY"
    chmod +x "$DESK_ENTRY"
    # GNOME 42+ requires marking as trusted
    if command -v gio >/dev/null 2>&1; then
        gio set "$DESK_ENTRY" "metadata::trusted" true 2>/dev/null || true
    fi
fi

echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}  Installation complete.${NC}"
echo -e "  Launch from the Applications menu ('${APP_NAME}')"
echo -e "  or double-click the desktop icon."
echo -e "  Command-line: ${LAUNCHER}"
echo -e "${GREEN}==============================================${NC}"
