@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1

:: Find Python
set PYTHON_CMD=
py -3 --version >nul 2>&1
if !ERRORLEVEL! equ 0 (
    set PYTHON_CMD=py -3
    goto :found
)
python --version 2>nul | findstr /i "Python 3" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    set PYTHON_CMD=python
    goto :found
)
echo ERROR: Python 3 not found. Run install_all.bat first.
pause
exit /b 1

:found
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found. Run install_all.bat first.
    pause
    exit /b 1
)

set "PYTHONPATH=%PYTHONPATH%;%~dp0SuperGluePretrainedNetwork"
call .venv\Scripts\activate.bat
python ReMap-GUI.py
