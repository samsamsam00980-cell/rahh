@echo off
REM Windows: double-click to start the Knowledge Galaxy (opens your browser).
cd /d "%~dp0"
echo Starting the Knowledge Galaxy. Its brain runs on this computer through Ollama (https://ollama.com/download).
where py >nul 2>nul
if %errorlevel%==0 (py server.py) else (python server.py)
echo.
pause
