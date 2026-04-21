@echo off
chcp 65001 >nul
title CytoTrack AI v1.0 - Setup

echo.
echo CytoTrack AI v1.0 - Source Setup
echo (If you want a one-click installer instead, run build_windows_exe.bat
echo  then compile packaging\installer.iss with Inno Setup 6.)
echo.

cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found! Install Python 3.10+ from python.org
    echo and make sure "Add to PATH" is ticked during install.
    pause
    exit /b 1
)

echo Creating virtual environment...
if exist "cell_track_venv" rmdir /s /q "cell_track_venv"
python -m venv cell_track_venv

call cell_track_venv\Scripts\activate.bat
python -m pip install --upgrade pip wheel
pip install -r requirements.txt

echo.
echo Done! Run LaunchCellTracker.bat to start CytoTrack AI.
pause
