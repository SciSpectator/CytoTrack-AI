# CytoTrack AI — Installation

## Ubuntu / Linux

1. Open a terminal in the project folder.
2. Run:
   ```bash
   ./install_linux.sh
   ```
   This will:
   - install required system packages (`python3-venv`, `libxcb-xinerama0`),
   - create a virtualenv at `./cell_track_venv`,
   - install Python dependencies,
   - register **CytoTrack AI** in the Applications menu,
   - drop a launcher on your Desktop (trusted automatically on GNOME).

3. Launch from the Applications menu (search "CytoTrack AI") or
   double-click the desktop icon.

To remove the menu entry / desktop icon (but keep the project files):

```bash
./uninstall_linux.sh
```

---

## Windows

### A. Build the portable .exe

1. Install **Python 3.10+** from python.org (tick *Add to PATH*).
2. Double-click **`build_windows_exe.bat`** in the project folder.
3. The build creates `dist\CytoTrackAI\CytoTrackAI.exe` — a fully
   portable application. You can copy that folder anywhere and run
   `CytoTrackAI.exe` directly.

### B. Build a proper .exe installer (optional, recommended)

1. Install **Inno Setup 6** (free): <https://jrsoftware.org/isdl.php>.
2. After step A above, run:
   ```
   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss
   ```
3. The installer will appear at
   `installer_output\CytoTrackAI-1.0-setup.exe`.
   Run it on any Windows 10/11 machine — the installer creates Start
   Menu and desktop icons, registers an uninstaller, and installs into
   `Program Files`.

---

## macOS (ad-hoc)

CytoTrack AI runs on macOS via:

```bash
python3 -m venv cell_track_venv
source cell_track_venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

A signed `.app` / `.dmg` is not currently provided.
