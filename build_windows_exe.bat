@echo off
REM ====================================================================
REM  CytoTrack AI - Windows build script
REM  Produces:  dist\CytoTrackAI\CytoTrackAI.exe  (portable)
REM  Optional:  signed installer via Inno Setup (installer.iss)
REM ====================================================================
setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0"

echo.
echo === CytoTrack AI - Windows build ===
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python not found in PATH. Install Python 3.10+ from python.org first.
    pause
    exit /b 1
)

REM 1) Fresh venv for the build
if not exist "build_venv" (
    echo [1/5] Creating build virtualenv...
    python -m venv build_venv
)
call build_venv\Scripts\activate.bat

REM 2) Install runtime + build deps
echo [2/5] Installing dependencies (this may take a few minutes)...
pip install --upgrade pip wheel
pip install -r requirements.txt
pip install pyinstaller>=6.0

REM 3) Make sure the app icon exists
if not exist "assets\icon.ico" (
    echo [3/5] Generating application icon...
    python packaging\generate_icon.py
)

REM 4) Clean previous build artefacts
echo [4/5] Cleaning previous build artefacts...
if exist "build" rd /s /q build
if exist "dist" rd /s /q dist

REM 5) Build
echo [5/5] Running PyInstaller...
pyinstaller packaging\CytoTrackAI.spec --noconfirm --clean
if errorlevel 1 (
    echo.
    echo Build failed. See errors above.
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo  Build complete.
echo  Portable app:  dist\CytoTrackAI\CytoTrackAI.exe
echo.
echo  To create a real .exe installer, install Inno Setup 6 and run:
echo     "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss
echo ====================================================================
pause
