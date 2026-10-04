@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "AQUAFLOW_PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%AQUAFLOW_PYTHON%" set "AQUAFLOW_PYTHON=python"
echo Starting Aqua Flow with: %AQUAFLOW_PYTHON%
echo Keep this window open. Dependency checks and model loading can take time.
echo Startup output will also be saved to presentation-startup.log.
echo.
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -Command "& $env:AQUAFLOW_PYTHON -u app.py 2>&1 | Tee-Object -FilePath 'presentation-startup.log'; exit $LASTEXITCODE"
set "AQUAFLOW_EXIT_CODE=%ERRORLEVEL%"
echo.
echo Server process ended with exit code %AQUAFLOW_EXIT_CODE%.
if exist "presentation-startup.log" echo Startup log: %CD%\presentation-startup.log
echo If startup failed, keep this window open and share the error above.
pause
exit /b %AQUAFLOW_EXIT_CODE%
