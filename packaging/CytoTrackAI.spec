# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for CytoTrack AI (Windows build).
#
# Produces a single-folder application at dist/CytoTrackAI/.
# Use the one-folder form because PyTorch + OpenCV + NumPy are large
# enough that onefile mode has long startup times and anti-virus
# false-positives on Windows.
#
# Build:  pyinstaller packaging\CytoTrackAI.spec --noconfirm --clean

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

PROJ = os.path.abspath(SPECPATH + os.sep + "..")

block_cipher = None

hidden = []
# Pull in runtime modules PyInstaller sometimes misses
for pkg in ("sklearn", "scipy.special", "scipy.optimize",
            "PyQt5.sip", "cv2", "plotly"):
    try:
        hidden += collect_submodules(pkg)
    except Exception:
        pass

datas = []
# Ship the assets and src tree as data (we import src dynamically)
datas += [(os.path.join(PROJ, "assets"), "assets")]
datas += [(os.path.join(PROJ, "src"), "src")]

a = Analysis(
    [os.path.join(PROJ, "main.py")],
    pathex=[PROJ, os.path.join(PROJ, "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pygame"],  # no longer used
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CytoTrackAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,        # windowed app (no console)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(PROJ, "assets", "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="CytoTrackAI",
)
