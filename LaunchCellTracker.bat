@echo off
REM Launch CytoTrack AI v1.0
cd /d "%~dp0"
if not exist "cell_track_venv" (
    echo Run Setup_Windows.bat first!
    pause
    exit /b 1
)
call cell_track_venv\Scripts\activate.bat
python main.py
pause
