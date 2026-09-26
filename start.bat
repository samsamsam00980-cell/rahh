@echo off
REM Windows: double-click to start the Knowledge Galaxy (opens your browser).
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (py server.py) else (python server.py)
echo.
pause
