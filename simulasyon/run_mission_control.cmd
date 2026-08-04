@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_mission_control.ps1" %*
exit /b %errorlevel%
