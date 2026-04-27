@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1

cd /d "%~dp0"

set "LOCAL_NODE_DIR=%~dp0.tools\node"
if exist "%LOCAL_NODE_DIR%\node.exe" set "PATH=%LOCAL_NODE_DIR%;%PATH%"

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found. Run install_all.bat first.
    pause
    exit /b 1
)

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo ERROR: Virtual environment Python not found. Run install_all.bat first.
    pause
    exit /b 1
)

where node >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo ERROR: Node.js not found. Run install_all.bat option 9 to install portable Node.js.
    pause
    exit /b 1
)

if not exist "node_modules" (
    echo ERROR: Frontend dependencies are missing.
    echo Run: install_all.bat --frontend
    echo You can still launch the previous UI with launch_legacy.bat
    pause
    exit /b 1
)

set "PYTHONPATH=%PYTHONPATH%;%~dp0SuperGluePretrainedNetwork"
call .venv\Scripts\activate.bat

"%PYTHON_EXE%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/internal/v1/health', timeout=1).read()" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo ReMap backend is already running on http://127.0.0.1:8765
    goto :backend_ready
)

start "ReMap Backend" cmd /k ""%PYTHON_EXE%" desktop_backend.py"

echo Waiting for ReMap backend on http://127.0.0.1:8765 ...
for /l %%i in (1,1,60) do (
    "%PYTHON_EXE%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/internal/v1/health', timeout=1).read()" >nul 2>&1
    if !ERRORLEVEL! equ 0 goto :backend_ready
    timeout /t 1 /nobreak >nul
)

echo.
echo ERROR: ReMap backend did not start. Check the "ReMap Backend" window above.
pause
exit /b 1

:backend_ready
cmd /c npm.cmd run desktop:dev
if !ERRORLEVEL! neq 0 (
    echo.
    echo ERROR: Tauri failed to start.
    pause
    exit /b 1
)
