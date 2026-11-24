@echo off
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" play_neat.py %*
) else (
    python play_neat.py %*
)

if errorlevel 1 pause
