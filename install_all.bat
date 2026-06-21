@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1
cd /d "%~dp0"

:: Variables for state
set "HAS_PYTHON=0"
set "HAS_GIT=0"
set "HAS_FFMPEG=0"
set "HAS_COLMAP=0"
set "HAS_GLOMAP=0"
set "HAS_VENV=0"
set "HAS_PIP_REQ=0"
set "HAS_SUPERGLUE=0"
set "HAS_NODE=0"
set "HAS_NPM=0"
set "HAS_FRONTEND=0"
set "HAS_CARGO=0"
set "HAS_RUST_DEPS=0"

set "PYTHON_CMD="
set "NO_PAUSE=0"
set "LOCAL_NODE_DIR=%~dp0.tools\node"
if exist "%LOCAL_NODE_DIR%\node.exe" set "PATH=%LOCAL_NODE_DIR%;%PATH%"

if /i "%~1"=="--frontend" (
    set "NO_PAUSE=1"
    goto :install_frontend
)
if /i "%~1"=="--node" (
    set "NO_PAUSE=1"
    goto :install_node
)
if /i "%~1"=="--rust-deps" (
    set "NO_PAUSE=1"
    goto :install_rust_deps
)

:scan
cls
echo.
echo =======================================
echo     ReMap - Installer ^& Manager [Win]
echo =======================================
echo.
echo Scanning system dependencies...

:: 1. Python
set "PYTHON_CMD="
set "HAS_PYTHON=0"
set "_PYVER_TMP=%TEMP%\_remap_pyver.tmp"

py -3 --version >nul 2>&1
if !ERRORLEVEL! equ 0 (
    set "PYTHON_CMD=py -3"
    set "HAS_PYTHON=1"
)

if !HAS_PYTHON! equ 0 (
    where python3 >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        python3 --version > "!_PYVER_TMP!" 2>&1
        findstr /i "Python 3" "!_PYVER_TMP!" >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            set "PYTHON_CMD=python3"
            set "HAS_PYTHON=1"
        )
    )
)

if !HAS_PYTHON! equ 0 (
    where python >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        python --version > "!_PYVER_TMP!" 2>&1
        findstr /i "Python 3" "!_PYVER_TMP!" >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            set "PYTHON_CMD=python"
            set "HAS_PYTHON=1"
        )
    )
)

if exist "!_PYVER_TMP!" del "!_PYVER_TMP!" >nul 2>&1


:: 2. Git
where git >nul 2>&1
if !ERRORLEVEL! equ 0 (set "HAS_GIT=1") else (set "HAS_GIT=0")

:: 3. FFmpeg
where ffmpeg >nul 2>&1
if !ERRORLEVEL! equ 0 (set "HAS_FFMPEG=1") else (set "HAS_FFMPEG=0")

:: 4. COLMAP
where colmap >nul 2>&1
if !ERRORLEVEL! equ 0 (set "HAS_COLMAP=1") else (set "HAS_COLMAP=0")

:: 5. GLOMAP
where glomap >nul 2>&1
if !ERRORLEVEL! equ 0 (set "HAS_GLOMAP=1") else (set "HAS_GLOMAP=0")

:: 6. Virtual Environment
if exist ".venv\Scripts\activate.bat" (set "HAS_VENV=1") else (set "HAS_VENV=0")

:: 7. Python Requirements (Approximate check via hloc, LoMa and psutil)
if !HAS_VENV! equ 1 (
    .venv\Scripts\python.exe -c "import cv2, hloc, kornia, loma, numpy, psutil, pycolmap, torch, torchvision; import flask, matplotlib, PIL, requests, scipy, tqdm; import OpenImageIO" >nul 2>&1
    if !ERRORLEVEL! equ 0 (set "HAS_PIP_REQ=1") else (set "HAS_PIP_REQ=0")
) else (
    set "HAS_PIP_REQ=0"
)

:: 8. SuperGlue
if exist "SuperGluePretrainedNetwork" (set "HAS_SUPERGLUE=1") else (set "HAS_SUPERGLUE=0")

:: 9. Node.js 20+ and npm
set "HAS_NODE=0"
set "HAS_NPM=0"
set "NODE_MAJOR="
where node >nul 2>&1
if !ERRORLEVEL! equ 0 (
    for /f "tokens=1 delims=." %%v in ('node -v 2^>nul') do set "NODE_MAJOR=%%v"
    set "NODE_MAJOR=!NODE_MAJOR:v=!"
    if defined NODE_MAJOR (
        if !NODE_MAJOR! geq 20 set "HAS_NODE=1"
    )
)
where npm >nul 2>&1
if !ERRORLEVEL! equ 0 set "HAS_NPM=1"

:: 10. Frontend npm packages
if !HAS_NODE! equ 1 if !HAS_NPM! equ 1 (
    cmd /c npm.cmd ls --depth=0 >nul 2>&1
    if !ERRORLEVEL! equ 0 (set "HAS_FRONTEND=1") else (set "HAS_FRONTEND=0")
) else (
    set "HAS_FRONTEND=0"
)

:: 11. Rust/Cargo dependencies for Tauri
where cargo >nul 2>&1
if !ERRORLEVEL! equ 0 (set "HAS_CARGO=1") else (set "HAS_CARGO=0")
if !HAS_CARGO! equ 1 (
    cargo metadata --manifest-path src-tauri\Cargo.toml --locked --offline --format-version 1 >nul 2>&1
    if !ERRORLEVEL! equ 0 (set "HAS_RUST_DEPS=1") else (set "HAS_RUST_DEPS=0")
) else (
    set "HAS_RUST_DEPS=0"
)

:menu
cls
echo.
echo =======================================
echo     ReMap - Installer ^& Manager [Win]
echo =======================================
echo.
echo   Current Status:

if !HAS_PYTHON! equ 1 (echo   [1] Python      : [OK]) else (echo   [1] Python      : [MISSING])
if !HAS_GIT! equ 1 (echo   [2] Git         : [OK]) else (echo   [2] Git         : [MISSING])
if !HAS_FFMPEG! equ 1 (echo   [3] FFmpeg      : [OK]) else (echo   [3] FFmpeg      : [MISSING])
if !HAS_COLMAP! equ 1 (echo   [4] COLMAP      : [OK]) else (echo   [4] COLMAP      : [MISSING])
if !HAS_GLOMAP! equ 1 (echo   [5] GLOMAP      : [OK ^(Optional^)]) else (echo   [5] GLOMAP      : [MISSING ^(Optional^)])
if !HAS_VENV! equ 1 (echo   [6] Python venv : [OK]) else (echo   [6] Python venv : [MISSING])
if !HAS_PIP_REQ! equ 1 (echo   [7] PIP packages: [OK]) else (echo   [7] PIP packages: [MISSING])
if !HAS_SUPERGLUE! equ 1 (echo   [8] SuperGlue   : [OK]) else (echo   [8] SuperGlue   : [MISSING])
if !HAS_NODE! equ 1 (echo   [9] Node.js/npm : [OK]) else (echo   [9] Node.js/npm : [MISSING])
if !HAS_FRONTEND! equ 1 (echo   [10] Frontend   : [OK]) else (echo   [10] Frontend   : [MISSING])
if !HAS_CARGO! equ 1 (
    if !HAS_RUST_DEPS! equ 1 (echo   [11] Tauri/Rust : [OK]) else (echo   [11] Tauri/Rust : [MISSING])
) else (
    echo   [11] Tauri/Rust : [MISSING]
)

echo.
echo   Actions:
echo   [A] Auto-Install missing components
echo   [F] Full Reinstall ^(Force all^)
echo   [L] Launch ReMap GUI
echo   [Q] Quit
echo.
echo   Type 1-11 to reinstall/configure a specific component.
echo.

set /p CHOICE="> Select an option: "

if /i "!CHOICE!"=="Q" exit /b 0
if /i "!CHOICE!"=="L" goto :launch

if /i "!CHOICE!"=="A" goto :do_auto_install
if /i "!CHOICE!"=="F" goto :do_full_reinstall

if "!CHOICE!"=="1" call :install_python & goto :scan
if "!CHOICE!"=="2" call :install_git & goto :scan
if "!CHOICE!"=="3" call :install_ffmpeg & goto :scan
if "!CHOICE!"=="4" call :install_colmap & goto :scan
if "!CHOICE!"=="5" call :install_glomap & goto :scan
if "!CHOICE!"=="6" call :install_venv & goto :scan
if "!CHOICE!"=="7" call :install_pip & goto :scan
if "!CHOICE!"=="8" call :install_superglue & goto :scan
if "!CHOICE!"=="9" call :install_node & goto :scan
if "!CHOICE!"=="10" call :install_frontend & goto :scan
if "!CHOICE!"=="11" call :install_rust & call :install_rust_deps & goto :scan

goto :menu

:do_auto_install
echo.
echo Starting Auto-Install of missing components...
if !HAS_PYTHON! equ 0 call :install_python
if !HAS_GIT! equ 0 call :install_git
if !HAS_FFMPEG! equ 0 call :install_ffmpeg
if !HAS_COLMAP! equ 0 call :install_colmap
if !HAS_GLOMAP! equ 0 call :install_glomap
if !HAS_VENV! equ 0 call :install_venv
if !HAS_PIP_REQ! equ 0 call :install_pip
if !HAS_SUPERGLUE! equ 0 call :install_superglue
if !HAS_NODE! equ 0 call :install_node
if !HAS_FRONTEND! equ 0 call :install_frontend
if !HAS_CARGO! equ 0 call :install_rust
if !HAS_RUST_DEPS! equ 0 call :install_rust_deps
goto :scan

:do_full_reinstall
echo.
echo Starting Full Reinstall...
call :install_python
call :install_git
call :install_ffmpeg
call :install_colmap
call :install_glomap
call :install_venv
call :install_pip
call :install_superglue
call :install_node
call :install_frontend
call :install_rust
call :install_rust_deps
goto :scan


:: =======================================
:: LAUNCH
:: =======================================
:launch
cls
if !HAS_PYTHON! equ 0 (
    echo ERROR: Python is required to launch.
    pause
    goto :menu
)
if !HAS_VENV! equ 0 (
    echo ERROR: Virtual environment not setup. Run install first.
    pause
    goto :menu
)
echo Launching ReMap...
set "PYTHONPATH=%PYTHONPATH%;%~dp0SuperGluePretrainedNetwork"
call .venv\Scripts\activate.bat
python ReMap-GUI.py
exit /b 0


:: =======================================
:: INSTALL SUBROUTINES
:: =======================================

:install_python
echo.
echo --- [1/11] Python ---
if !HAS_PYTHON! equ 1 goto :python_already_ok
echo Python 3.10+ must be installed manually.
echo Download from: https://www.python.org/downloads/
echo IMPORTANT: Check "Add python.exe to PATH" during install.
echo Also disable Microsoft Store aliases in:
echo   Settings ^> Apps ^> Advanced app settings ^> App execution aliases
pause
goto :eof
:python_already_ok
echo Python already detected: !PYTHON_CMD!
pause
goto :eof

:install_git
echo.
echo --- [2/11] Git ---
where winget >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo Installing Git via winget...
    winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
    call :refresh_path
    pause
    goto :eof
)
echo winget not found. Downloading Git installer...
set "PS_SCRIPT=%TEMP%\remap_install_git.ps1"
> "!PS_SCRIPT!" echo $ErrorActionPreference = 'Stop'
>> "!PS_SCRIPT!" echo $ProgressPreference = 'SilentlyContinue'
>> "!PS_SCRIPT!" echo try {
>> "!PS_SCRIPT!" echo     $h = @{ 'User-Agent' = 'ReMap-Installer' }
>> "!PS_SCRIPT!" echo     $rel = Invoke-RestMethod -Uri 'https://api.github.com/repos/git-for-windows/git/releases/latest' -Headers $h
>> "!PS_SCRIPT!" echo     $asset = $rel.assets ^| Where-Object { $_.name -match '64-bit\.exe$' } ^| Select-Object -First 1
>> "!PS_SCRIPT!" echo     if (-not $asset) { throw 'No 64-bit Git installer found' }
>> "!PS_SCRIPT!" echo     $installer = Join-Path $env:TEMP 'git_installer.exe'
>> "!PS_SCRIPT!" echo     Write-Host "  Downloading $($asset.name)..."
>> "!PS_SCRIPT!" echo     Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $installer
>> "!PS_SCRIPT!" echo     Write-Host '  Running installer (silent)...'
>> "!PS_SCRIPT!" echo     $iArgs = @('/VERYSILENT','/NORESTART','/NOCANCEL','/SP-','/CLOSEAPPLICATIONS','/RESTARTAPPLICATIONS','/COMPONENTS=icons,ext\reg\shellhere,assoc,assoc_sh')
>> "!PS_SCRIPT!" echo     Start-Process -FilePath $installer -ArgumentList $iArgs -Wait
>> "!PS_SCRIPT!" echo     Remove-Item $installer -Force
>> "!PS_SCRIPT!" echo     Write-Host '  Git installed.'
>> "!PS_SCRIPT!" echo } catch {
>> "!PS_SCRIPT!" echo     Write-Host "  ERROR: $($_.Exception.Message)"
>> "!PS_SCRIPT!" echo     Write-Host '  Please install Git manually: https://git-scm.com/download/win'
>> "!PS_SCRIPT!" echo }
powershell -NoProfile -ExecutionPolicy Bypass -File "!PS_SCRIPT!"
del "!PS_SCRIPT!" 2>nul
call :refresh_path
pause
goto :eof

:install_ffmpeg
echo.
echo --- [3/11] FFmpeg ---
where winget >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo Installing FFmpeg via winget...
    winget install --id Gyan.FFmpeg -e --source winget --accept-package-agreements --accept-source-agreements
    call :refresh_path
    pause
    goto :eof
)
echo winget not found. Downloading FFmpeg...
set "PS_SCRIPT=%TEMP%\remap_install_ffmpeg.ps1"
> "!PS_SCRIPT!" echo $ErrorActionPreference = 'Stop'
>> "!PS_SCRIPT!" echo $ProgressPreference = 'SilentlyContinue'
>> "!PS_SCRIPT!" echo $installDir = Join-Path $env:LOCALAPPDATA 'FFmpeg'
>> "!PS_SCRIPT!" echo try {
>> "!PS_SCRIPT!" echo     $url = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
>> "!PS_SCRIPT!" echo     $zip = Join-Path $env:TEMP 'ffmpeg.zip'
>> "!PS_SCRIPT!" echo     Write-Host '  Downloading FFmpeg essentials build...'
>> "!PS_SCRIPT!" echo     Invoke-WebRequest -Uri $url -OutFile $zip
>> "!PS_SCRIPT!" echo     Write-Host '  Extracting...'
>> "!PS_SCRIPT!" echo     if (Test-Path $installDir) { Remove-Item $installDir -Recurse -Force }
>> "!PS_SCRIPT!" echo     Expand-Archive -Path $zip -DestinationPath $installDir -Force
>> "!PS_SCRIPT!" echo     Remove-Item $zip -Force
>> "!PS_SCRIPT!" echo     $binDir = Get-ChildItem -Path $installDir -Recurse -Directory -Filter 'bin' ^| Select-Object -First 1
>> "!PS_SCRIPT!" echo     $ffmpegBin = if ($binDir) { $binDir.FullName } else { $installDir }
>> "!PS_SCRIPT!" echo     $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
>> "!PS_SCRIPT!" echo     if (-not $userPath) { $userPath = '' }
>> "!PS_SCRIPT!" echo     if ($userPath -notlike "*$ffmpegBin*") {
>> "!PS_SCRIPT!" echo         [Environment]::SetEnvironmentVariable('PATH', "$userPath;$ffmpegBin", 'User')
>> "!PS_SCRIPT!" echo     }
>> "!PS_SCRIPT!" echo     Write-Host "  FFmpeg installed to: $ffmpegBin"
>> "!PS_SCRIPT!" echo } catch {
>> "!PS_SCRIPT!" echo     Write-Host "  ERROR: $($_.Exception.Message)"
>> "!PS_SCRIPT!" echo     Write-Host '  Please install FFmpeg manually: https://www.gyan.dev/ffmpeg/builds/'
>> "!PS_SCRIPT!" echo }
powershell -NoProfile -ExecutionPolicy Bypass -File "!PS_SCRIPT!"
del "!PS_SCRIPT!" 2>nul
call :refresh_path
pause
goto :eof

:install_colmap
echo.
echo --- [4/11] COLMAP ---
echo Downloading COLMAP from GitHub releases...
set "PS_SCRIPT=%TEMP%\remap_install_colmap.ps1"
> "!PS_SCRIPT!" echo $ErrorActionPreference = 'Stop'
>> "!PS_SCRIPT!" echo $ProgressPreference = 'SilentlyContinue'
>> "!PS_SCRIPT!" echo $installDir = Join-Path $env:LOCALAPPDATA 'COLMAP'
>> "!PS_SCRIPT!" echo try {
>> "!PS_SCRIPT!" echo     Write-Host '  Querying GitHub for latest COLMAP release...'
>> "!PS_SCRIPT!" echo     $headers = @{ 'User-Agent' = 'ReMap-Installer' }
>> "!PS_SCRIPT!" echo     $release = Invoke-RestMethod -Uri 'https://api.github.com/repos/colmap/colmap/releases/latest' -Headers $headers
>> "!PS_SCRIPT!" echo     $asset = $release.assets ^| Where-Object { $_.name -match 'windows' -and $_.name -match 'cuda' -and $_.name -match '\.zip$' } ^| Select-Object -First 1
>> "!PS_SCRIPT!" echo     if (-not $asset) { $asset = $release.assets ^| Where-Object { $_.name -match 'windows' -and $_.name -match '\.zip$' } ^| Select-Object -First 1 }
>> "!PS_SCRIPT!" echo     if (-not $asset) { throw 'No Windows COLMAP release found on GitHub' }
>> "!PS_SCRIPT!" echo     $zipPath = Join-Path $env:TEMP 'colmap_release.zip'
>> "!PS_SCRIPT!" echo     Write-Host "  Downloading $($asset.name)..."
>> "!PS_SCRIPT!" echo     Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zipPath
>> "!PS_SCRIPT!" echo     if (Test-Path $installDir) { Remove-Item $installDir -Recurse -Force }
>> "!PS_SCRIPT!" echo     Write-Host '  Extracting...'
>> "!PS_SCRIPT!" echo     Expand-Archive -Path $zipPath -DestinationPath $installDir -Force
>> "!PS_SCRIPT!" echo     Remove-Item $zipPath -Force
>> "!PS_SCRIPT!" echo     $binDir = Get-ChildItem -Path $installDir -Recurse -Directory -Filter 'bin' ^| Select-Object -First 1
>> "!PS_SCRIPT!" echo     $colmapBin = if ($binDir) { $binDir.FullName } else { $installDir }
>> "!PS_SCRIPT!" echo     $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
>> "!PS_SCRIPT!" echo     if (-not $userPath) { $userPath = '' }
>> "!PS_SCRIPT!" echo     if ($userPath -notlike "*$colmapBin*") {
>> "!PS_SCRIPT!" echo         [Environment]::SetEnvironmentVariable('PATH', "$userPath;$colmapBin", 'User')
>> "!PS_SCRIPT!" echo     }
>> "!PS_SCRIPT!" echo     Write-Host "  COLMAP installed."
>> "!PS_SCRIPT!" echo } catch {
>> "!PS_SCRIPT!" echo     Write-Host "  ERROR: $($_.Exception.Message)"
>> "!PS_SCRIPT!" echo     exit 1
>> "!PS_SCRIPT!" echo }
powershell -NoProfile -ExecutionPolicy Bypass -File "!PS_SCRIPT!"
del "!PS_SCRIPT!" 2>nul
call :refresh_path
pause
goto :eof

:install_glomap
echo.
echo --- [5/11] GLOMAP ---
echo Downloading GLOMAP from GitHub releases...
set "PS_SCRIPT=%TEMP%\remap_install_glomap.ps1"
> "!PS_SCRIPT!" echo $ErrorActionPreference = 'Stop'
>> "!PS_SCRIPT!" echo $ProgressPreference = 'SilentlyContinue'
>> "!PS_SCRIPT!" echo $installDir = Join-Path $env:LOCALAPPDATA 'GLOMAP'
>> "!PS_SCRIPT!" echo try {
>> "!PS_SCRIPT!" echo     Write-Host '  Querying GitHub for latest GLOMAP release...'
>> "!PS_SCRIPT!" echo     $headers = @{ 'User-Agent' = 'ReMap-Installer' }
>> "!PS_SCRIPT!" echo     $release = Invoke-RestMethod -Uri 'https://api.github.com/repos/colmap/glomap/releases/latest' -Headers $headers
>> "!PS_SCRIPT!" echo     $asset = $release.assets ^| Where-Object { $_.name -match 'windows' -and $_.name -match 'cuda' -and $_.name -match '\.zip$' } ^| Select-Object -First 1
>> "!PS_SCRIPT!" echo     if (-not $asset) { $asset = $release.assets ^| Where-Object { $_.name -match 'windows' -and $_.name -match '\.zip$' } ^| Select-Object -First 1 }
>> "!PS_SCRIPT!" echo     if (-not $asset) { throw 'No Windows GLOMAP release found on GitHub' }
>> "!PS_SCRIPT!" echo     $zipPath = Join-Path $env:TEMP 'glomap_release.zip'
>> "!PS_SCRIPT!" echo     Write-Host "  Downloading $($asset.name)..."
>> "!PS_SCRIPT!" echo     Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zipPath
>> "!PS_SCRIPT!" echo     if (Test-Path $installDir) { Remove-Item $installDir -Recurse -Force }
>> "!PS_SCRIPT!" echo     Write-Host '  Extracting...'
>> "!PS_SCRIPT!" echo     Expand-Archive -Path $zipPath -DestinationPath $installDir -Force
>> "!PS_SCRIPT!" echo     Remove-Item $zipPath -Force
>> "!PS_SCRIPT!" echo     $binDir = Get-ChildItem -Path $installDir -Recurse -Directory -Filter 'bin' ^| Select-Object -First 1
>> "!PS_SCRIPT!" echo     $glomapBin = if ($binDir) { $binDir.FullName } else { $installDir }
>> "!PS_SCRIPT!" echo     $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
>> "!PS_SCRIPT!" echo     if (-not $userPath) { $userPath = '' }
>> "!PS_SCRIPT!" echo     if ($userPath -notlike "*$glomapBin*") {
>> "!PS_SCRIPT!" echo         [Environment]::SetEnvironmentVariable('PATH', "$userPath;$glomapBin", 'User')
>> "!PS_SCRIPT!" echo     }
>> "!PS_SCRIPT!" echo     Write-Host "  GLOMAP installed."
>> "!PS_SCRIPT!" echo } catch {
>> "!PS_SCRIPT!" echo     Write-Host "  ERROR: $($_.Exception.Message)"
>> "!PS_SCRIPT!" echo     exit 1
>> "!PS_SCRIPT!" echo }
powershell -NoProfile -ExecutionPolicy Bypass -File "!PS_SCRIPT!"
del "!PS_SCRIPT!" 2>nul
call :refresh_path
pause
goto :eof

:install_venv
echo.
echo --- [6/11] Python Virtual Environment ---
if !HAS_PYTHON! equ 0 (
    echo Python must be installed first.
    pause
    goto :eof
)
if exist ".venv" (
    echo Removing old virtual environment...
    rmdir /s /q .venv
)
echo Creating virtual environment...
!PYTHON_CMD! -m venv .venv
if !ERRORLEVEL! equ 0 (
    echo Virtual environment created.
) else (
    echo ERROR: Failed to create virtual environment.
)
pause
goto :eof

:install_pip
echo.
echo --- [7/11] PIP Packages ---
if !HAS_VENV! equ 0 (
    echo Virtual environment must be created first.
    pause
    goto :eof
)
set "PIP_LOG_DIR=%~dp0backend_state\install_logs"
if not exist "!PIP_LOG_DIR!" mkdir "!PIP_LOG_DIR!" >nul 2>&1
set "PIP_LOG=!PIP_LOG_DIR!\pip_install.log"
echo Writing pip install log to: !PIP_LOG!
echo ==== ReMap pip install %DATE% %TIME% ==== > "!PIP_LOG!"
echo Installing/Upgrading pip requirements...
call .venv\Scripts\activate.bat
echo ^> python -m pip install --upgrade pip >> "!PIP_LOG!"
python -m pip install --upgrade pip >> "!PIP_LOG!" 2>&1
if !ERRORLEVEL! neq 0 goto :pip_failed

echo Removing any existing CPU-only PyTorch versions...
echo ^> python -m pip uninstall torch torchvision torchaudio -y >> "!PIP_LOG!"
python -m pip uninstall torch torchvision torchaudio -y >> "!PIP_LOG!" 2>&1
if !ERRORLEVEL! neq 0 goto :pip_failed

echo Installing PyTorch with CUDA 12.8 support...
echo ^> python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 >> "!PIP_LOG!"
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 >> "!PIP_LOG!" 2>&1
if !ERRORLEVEL! neq 0 goto :pip_failed

echo Installing other dependencies...
echo ^> python -m pip install -r requirements.txt >> "!PIP_LOG!"
python -m pip install -r requirements.txt >> "!PIP_LOG!" 2>&1
if !ERRORLEVEL! neq 0 goto :pip_failed
echo Requirements installed successfully.

echo Installing LoMa metadata compatibility shim...
echo ^> python -m pip install --ignore-requires-python dataclasses==0.8 >> "!PIP_LOG!"
python -m pip install --ignore-requires-python dataclasses==0.8 >> "!PIP_LOG!" 2>&1
if !ERRORLEVEL! neq 0 goto :pip_failed

echo Installing pinned LoMa from official repository ^(--no-deps; dependencies are pinned in requirements.txt^)...
echo ^> python -m pip install --no-deps --force-reinstall git+https://github.com/davnords/LoMa.git@9105854833f55d18194d0505d913f0a74b194ef0#egg=lomatch >> "!PIP_LOG!"
python -m pip install --no-deps --force-reinstall "git+https://github.com/davnords/LoMa.git@9105854833f55d18194d0505d913f0a74b194ef0#egg=lomatch" >> "!PIP_LOG!" 2>&1
if !ERRORLEVEL! neq 0 goto :pip_failed

echo Verifying Python dependency consistency...
echo ^> python -m pip check >> "!PIP_LOG!"
python -m pip check >> "!PIP_LOG!" 2>&1
if !ERRORLEVEL! neq 0 goto :pip_failed

echo Verifying required imports...
echo ^> python -c "import required modules" >> "!PIP_LOG!"
python -c "import cv2, hloc, kornia, loma, numpy, psutil, pycolmap, torch, torchvision; import flask, matplotlib, PIL, requests, scipy, tqdm; import OpenImageIO" >> "!PIP_LOG!" 2>&1
if !ERRORLEVEL! neq 0 goto :pip_failed

echo PIP packages installed and verified successfully.
pause
goto :eof

:pip_failed
echo.
echo ERROR: Python dependency installation failed.
echo See the full log here:
echo   !PIP_LOG!
echo.
echo Common causes on a fresh machine:
echo   - Python version not supported by pinned wheels.
echo   - CUDA/PyTorch cu128 wheels unavailable for the platform.
echo   - Git/network access blocked for HLoc, LightGlue, or LoMa.
call :wait
goto :eof

:install_superglue
echo.
echo --- [8/11] SuperGluePretrainedNetwork ---
if !HAS_GIT! equ 0 (
    echo Git must be installed first to clone the repository.
    pause
    goto :eof
)
if exist "SuperGluePretrainedNetwork" (
    echo Removing existing SuperGlue repository...
    rmdir /s /q SuperGluePretrainedNetwork
)
echo Cloning SuperGluePretrainedNetwork...
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
pause
goto :eof

:install_node
echo.
echo --- [9/11] Node.js / npm ---
echo Installing portable Node.js LTS without admin rights...
set "PS_SCRIPT=%TEMP%\remap_install_node.ps1"
> "!PS_SCRIPT!" echo $ErrorActionPreference = 'Stop'
>> "!PS_SCRIPT!" echo $ProgressPreference = 'SilentlyContinue'
>> "!PS_SCRIPT!" echo $repo = (Resolve-Path '.').Path
>> "!PS_SCRIPT!" echo $toolsDir = Join-Path $repo '.tools'
>> "!PS_SCRIPT!" echo $installDir = Join-Path $toolsDir 'node'
>> "!PS_SCRIPT!" echo $installFull = [System.IO.Path]::GetFullPath($installDir)
>> "!PS_SCRIPT!" echo $toolsFull = [System.IO.Path]::GetFullPath($toolsDir)
>> "!PS_SCRIPT!" echo if (-not $installFull.StartsWith($toolsFull, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Refusing to install Node outside .tools' }
>> "!PS_SCRIPT!" echo New-Item -ItemType Directory -Force -Path $toolsDir ^| Out-Null
>> "!PS_SCRIPT!" echo $arch = if ([Environment]::Is64BitOperatingSystem) { 'win-x64' } else { 'win-x86' }
>> "!PS_SCRIPT!" echo $fileToken = "$arch-zip"
>> "!PS_SCRIPT!" echo Write-Host '  Querying Node.js release index...'
>> "!PS_SCRIPT!" echo $index = Invoke-RestMethod -Uri 'https://nodejs.org/dist/index.json'
>> "!PS_SCRIPT!" echo $release = $index ^| Where-Object { $_.lts -and $_.files -contains $fileToken -and [int]($_.version.TrimStart('v').Split('.')[0]) -ge 20 } ^| Select-Object -First 1
>> "!PS_SCRIPT!" echo if (-not $release) { throw 'No compatible Node.js LTS release found' }
>> "!PS_SCRIPT!" echo $zipName = "node-$($release.version)-$arch.zip"
>> "!PS_SCRIPT!" echo $url = "https://nodejs.org/dist/$($release.version)/$zipName"
>> "!PS_SCRIPT!" echo $zipPath = Join-Path $env:TEMP $zipName
>> "!PS_SCRIPT!" echo $extractDir = Join-Path $env:TEMP "remap-node-$($release.version)-$arch"
>> "!PS_SCRIPT!" echo Write-Host "  Downloading $zipName..."
>> "!PS_SCRIPT!" echo Invoke-WebRequest -Uri $url -OutFile $zipPath
>> "!PS_SCRIPT!" echo if (Test-Path $extractDir) { Remove-Item -LiteralPath $extractDir -Recurse -Force }
>> "!PS_SCRIPT!" echo if (Test-Path $installDir) { Remove-Item -LiteralPath $installDir -Recurse -Force }
>> "!PS_SCRIPT!" echo Write-Host '  Extracting portable Node.js...'
>> "!PS_SCRIPT!" echo Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force
>> "!PS_SCRIPT!" echo $expanded = Get-ChildItem -Path $extractDir -Directory ^| Select-Object -First 1
>> "!PS_SCRIPT!" echo if (-not $expanded) { throw 'Node.js archive did not contain an install directory' }
>> "!PS_SCRIPT!" echo Move-Item -LiteralPath $expanded.FullName -Destination $installDir
>> "!PS_SCRIPT!" echo Remove-Item -LiteralPath $zipPath -Force
>> "!PS_SCRIPT!" echo Remove-Item -LiteralPath $extractDir -Recurse -Force
>> "!PS_SCRIPT!" echo Write-Host "  Node.js $($release.version) installed to $installDir"
powershell -NoProfile -ExecutionPolicy Bypass -File "!PS_SCRIPT!"
set "NODE_INSTALL_EXIT=!ERRORLEVEL!"
del "!PS_SCRIPT!" 2>nul
if !NODE_INSTALL_EXIT! neq 0 (
    echo ERROR: Portable Node.js installation failed.
) else (
    set "PATH=%LOCAL_NODE_DIR%;%PATH%"
    echo Portable Node.js is ready for ReMap.
)
call :wait
goto :eof

:install_frontend
echo.
echo --- [10/11] Frontend Packages ---
set "NODE_MAJOR="
where node >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo Node.js is missing. Installing Node.js first...
    call :install_node
)
where npm >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo ERROR: npm was not found. Install Node.js 20+ and rerun this option.
    call :wait
    goto :eof
)
for /f "tokens=1 delims=." %%v in ('node -v 2^>nul') do set "NODE_MAJOR=%%v"
set "NODE_MAJOR=!NODE_MAJOR:v=!"
if not defined NODE_MAJOR (
    echo ERROR: Could not detect the Node.js version.
    call :wait
    goto :eof
)
if !NODE_MAJOR! lss 20 (
    echo ERROR: Node.js 20+ is required. Current major version: !NODE_MAJOR!
    call :install_node
    call :wait
    goto :eof
)
if exist "package-lock.json" (
    echo Installing frontend dependencies from package-lock.json...
    cmd /c npm.cmd ci
) else (
    echo Installing frontend dependencies...
    cmd /c npm.cmd install
)
if !ERRORLEVEL! neq 0 (
    echo WARNING: Frontend dependency installation failed.
) else (
    echo Frontend dependencies installed successfully.
)
call :wait
goto :eof

:install_rust
echo.
echo --- [11/11] Rust Toolchain ---
where cargo >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo Rust/Cargo is already available.
    call :wait
    goto :eof
)
where winget >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo Installing Rustup via winget...
    winget install --id Rustlang.Rustup -e --source winget --accept-package-agreements --accept-source-agreements
    call :refresh_path
    if exist "%USERPROFILE%\.cargo\bin" set "PATH=%PATH%;%USERPROFILE%\.cargo\bin"
) else (
    echo winget not found. Install Rustup manually:
    echo https://rustup.rs/
)
where rustup >nul 2>&1
if !ERRORLEVEL! equ 0 rustup default stable
where cargo >nul 2>&1
if !ERRORLEVEL! neq 0 echo WARNING: Cargo is still not available in PATH. Open a new terminal or rerun the installer.
call :wait
goto :eof

:install_rust_deps
echo.
echo --- [11/11] Tauri / Cargo Dependencies ---
where cargo >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo Cargo is missing. Install Rust first.
    call :wait
    goto :eof
)
echo Fetching Rust dependencies for the Tauri shell...
cargo fetch --manifest-path src-tauri\Cargo.toml --locked
if !ERRORLEVEL! neq 0 (
    echo WARNING: Cargo dependency fetch failed.
) else (
    echo Tauri/Rust dependencies fetched successfully.
)
call :wait
goto :eof

:: ==========================================
::  Wait helper
:: ==========================================
:wait
if /i "%NO_PAUSE%"=="1" goto :eof
pause
goto :eof

:: ==========================================
::  Refresh PATH subroutine
:: ==========================================
:refresh_path
for /f "tokens=*" %%a in ('powershell -NoProfile -Command "[System.Environment]::GetEnvironmentVariable('PATH','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('PATH','User')"') do set "PATH=%%a"
goto :eof
