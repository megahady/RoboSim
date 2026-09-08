@echo off
rem robot_studio launcher using the HadyLab virtual environment.
cd /d "%~dp0"
".venv\Scripts\pythonw.exe" -m robot_studio %*