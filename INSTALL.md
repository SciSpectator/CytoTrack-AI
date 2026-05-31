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
   - install Python dependencies from `requirements.txt`,
   - register **CytoTrack AI** in the Applications menu,
   - drop a launcher on your Desktop (trusted automatically on GNOME).

3. Launch from the Applications menu (search "CytoTrack AI") or
   double-click the desktop icon.

> **Optional backend (NVIDIA LocateAnything-3B).** The installers can also
> install `requirements-locate.txt`. This backend is not used by the default
> research-paper result path because it is under the **NVIDIA non-commercial
> license** (academic / non-profit research only). The default detector is
> the open classical multi-strategy detector. To install this optional backend,
> set `CYTOTRACK_INSTALL_LOCATE=1` before running the installer.

Example:

```bash
CYTOTRACK_INSTALL_LOCATE=1 ./install_linux.sh
```

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

For source setup, run `Setup_Windows.bat`, then `LaunchCellTracker.bat`.
The optional NVIDIA backend is opt-in:

```bat
set CYTOTRACK_INSTALL_LOCATE=1
Setup_Windows.bat
```

## Tracking Workflow

1. Open **Track Cells**.
2. Select a folder of microscopy frames.
3. Enter the cell line or comma-separated cell lines.
4. Choose pre-tracking morphology preparation:
   - **Website QAgents**: licence-checked public resources, cached outside
     `RESULT/`.
   - **User Data**: a local folder with one class folder per requested cell
     line.
   - **Existing Model**: a trained folder containing `class_map.json`.
   - **Single Line Label**: only for one declared cell line.
5. Confirm brightness/contrast/gamma/filter settings.
6. Run tracking.

All paper-facing outputs go under `RESULT/` with videos, plots, dashboards,
CSV metrics, manifests, and QC/provenance files. Keep
`RESULT/RESEARCH_USE_PROVENANCE.md`, `LICENSE`, `NOTICE`, and `CITATION.cff`
with any manuscript or supplementary result package.

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
