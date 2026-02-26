@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1

:: Variables for state
set "HAS_PYTHON=0"
set "HAS_GIT=0"
set "HAS_FFMPEG=0"
set "HAS_COLMAP=0"
set "HAS_GLOMAP=0"
set "HAS_VENV=0"
set "HAS_PIP_REQ=0"
set "HAS_SUPERGLUE=0"

set "PYTHON_CMD="

:scan
cls
echo.
echo =======================================
echo     ReMap - Installer ^& Manager [Win]
echo =======================================
echo.
echo Scanning system dependencies...

:: 1. Python
set PYTHON_CMD=
py -3 --version >nul 2>&1
if !ERRORLEVEL! equ 0 (
    set "PYTHON_CMD=py -3"
    set "HAS_PYTHON=1"
) else (
    python --version 2>nul | findstr /i "Python 3" >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "PYTHON_CMD=python"
        set "HAS_PYTHON=1"
    ) else (
        set "HAS_PYTHON=0"
    )
)

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

:: 7. Python Requirements (Approximate check via hloc)
if !HAS_VENV! equ 1 (
    .venv\Scripts\python.exe -c "import hloc" >nul 2>&1
    if !ERRORLEVEL! equ 0 (set "HAS_PIP_REQ=1") else (set "HAS_PIP_REQ=0")
) else (
    set "HAS_PIP_REQ=0"
)

:: 8. SuperGlue
if exist "SuperGluePretrainedNetwork" (set "HAS_SUPERGLUE=1") else (set "HAS_SUPERGLUE=0")

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

echo.
echo   Actions:
echo   [A] Auto-Install missing components
echo   [F] Full Reinstall ^(Force all^)
echo   [L] Launch ReMap GUI
echo   [Q] Quit
echo.
echo   Type 1-8 to reinstall/configure a specific component.
echo.

set /p CHOICE="> Select an option: "

if /i "!CHOICE!"=="Q" exit /b 0
if /i "!CHOICE!"=="L" goto :launch

if /i "!CHOICE!"=="A" (
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
    goto :scan
)

if /i "!CHOICE!"=="F" (
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
    goto :scan
)

if "!CHOICE!"=="1" call :install_python & goto :scan
if "!CHOICE!"=="2" call :install_git & goto :scan
if "!CHOICE!"=="3" call :install_ffmpeg & goto :scan
if "!CHOICE!"=="4" call :install_colmap & goto :scan
if "!CHOICE!"=="5" call :install_glomap & goto :scan
if "!CHOICE!"=="6" call :install_venv & goto :scan
if "!CHOICE!"=="7" call :install_pip & goto :scan
if "!CHOICE!"=="8" call :install_superglue & goto :scan

goto :menu


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
echo --- [1/8] Python ---
echo Python 3.10+ must be installed manually.
echo Download from: https://www.python.org/downloads/
echo IMPORTANT: Check "Add python.exe to PATH" during install.
echo Also disable Microsoft Store aliases in:
echo   Settings ^> Apps ^> Advanced app settings ^> App execution aliases
pause
goto :eof

:install_git
echo.
echo --- [2/8] Git ---
where winget >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo Installing Git via winget...
    winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
    call :refresh_path
) else (
    echo winget not found. Downloading Git installer...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$ProgressPreference = 'SilentlyContinue'; " ^
        "try { " ^
        "  $h = @{ 'User-Agent' = 'ReMap-Installer' }; " ^
        "  $rel = Invoke-RestMethod -Uri 'https://api.github.com/repos/git-for-windows/git/releases/latest' -Headers $h; " ^
        "  $asset = $rel.assets | Where-Object { $_.name -match '64-bit\.exe$' } | Select-Object -First 1; " ^
        "  Write-Host \"  Downloading $($asset.name)...\"; " ^
        "  Invoke-WebRequest -Uri $asset.browser_download_url -OutFile \"$env:TEMP\git_installer.exe\"; " ^
        "  Write-Host '  Running installer (silent)...'; " ^
        "  Start-Process -FilePath \"$env:TEMP\git_installer.exe\" -ArgumentList '/VERYSILENT','/NORESTART','/NOCANCEL','/SP-','/CLOSEAPPLICATIONS','/RESTARTAPPLICATIONS','/COMPONENTS=icons,ext\reg\shellhere,assoc,assoc_sh' -Wait; " ^
        "  Remove-Item \"$env:TEMP\git_installer.exe\" -Force; " ^
        "  Write-Host '  Git installed.'; " ^
        "} catch { " ^
        "  Write-Host \"  ERROR: $($_.Exception.Message)\"; " ^
        "  Write-Host '  Please install Git manually: https://git-scm.com/download/win'; " ^
        "}"
    call :refresh_path
)
pause
goto :eof

:install_ffmpeg
echo.
echo --- [3/8] FFmpeg ---
where winget >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo Installing FFmpeg via winget...
    winget install --id Gyan.FFmpeg -e --source winget --accept-package-agreements --accept-source-agreements
    call :refresh_path
) else (
    echo winget not found. Downloading FFmpeg...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$ProgressPreference = 'SilentlyContinue'; " ^
        "try { " ^
        "  $installDir = Join-Path $env:LOCALAPPDATA 'FFmpeg'; " ^
        "  $url = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'; " ^
        "  $zip = Join-Path $env:TEMP 'ffmpeg.zip'; " ^
        "  Write-Host '  Downloading FFmpeg essentials build...'; " ^
        "  Invoke-WebRequest -Uri $url -OutFile $zip; " ^
        "  Write-Host '  Extracting...'; " ^
        "  if (Test-Path $installDir) { Remove-Item $installDir -Recurse -Force }; " ^
        "  Expand-Archive -Path $zip -DestinationPath $installDir -Force; " ^
        "  Remove-Item $zip -Force; " ^
        "  $binDir = Get-ChildItem -Path $installDir -Recurse -Directory -Filter 'bin' | Select-Object -First 1; " ^
        "  $ffmpegBin = if ($binDir) { $binDir.FullName } else { $installDir }; " ^
        "  $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User'); " ^
        "  if (-not $userPath) { $userPath = '' }; " ^
        "  if ($userPath -notlike \"*$ffmpegBin*\") { " ^
        "    [Environment]::SetEnvironmentVariable('PATH', \"$userPath;$ffmpegBin\", 'User'); " ^
        "  }; " ^
        "  Write-Host \"  FFmpeg installed to: $ffmpegBin\"; " ^
        "} catch { " ^
        "  Write-Host \"  ERROR: $($_.Exception.Message)\"; " ^
        "  Write-Host '  Please install FFmpeg manually: https://www.gyan.dev/ffmpeg/builds/'; " ^
        "}"
    call :refresh_path
)
pause
goto :eof

:install_colmap
echo.
echo --- [4/8] COLMAP ---
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
echo --- [5/8] GLOMAP ---
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
echo --- [6/8] Python Virtual Environment ---
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
echo --- [7/8] PIP Packages ---
if !HAS_VENV! equ 0 (
    echo Virtual environment must be created first.
    pause
    goto :eof
)
echo Installing/Upgrading pip requirements...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo Removing any existing CPU-only PyTorch versions...
python -m pip uninstall torch torchvision torchaudio -y

echo Installing PyTorch with CUDA 12.8 support...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo Installing other dependencies...
pip install -r requirements.txt
if !ERRORLEVEL! neq 0 (
    echo WARNING: Some packages failed to install.
) else (
    echo Requirements installed successfully.
)
pause
goto :eof

:install_superglue
echo.
echo --- [8/8] SuperGluePretrainedNetwork ---
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

:: ==========================================
::  Refresh PATH subroutine
:: ==========================================
:refresh_path
for /f "tokens=*" %%a in ('powershell -NoProfile -Command "[System.Environment]::GetEnvironmentVariable('PATH','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('PATH','User')"') do set "PATH=%%a"
goto :eof
