@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
    python scripts\start.py %*
) else (
    py -3.11 scripts\start.py %*
)
exit /b %errorlevel%
