#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
TARGET_USER="${SUDO_USER:-$USER}"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
if [ -n "$TARGET_HOME" ]; then
    export PATH="$TARGET_HOME/.cargo/bin:$PATH"
fi

# Check for root
if [ "$EUID" -ne 0 ]; then 
  echo -e "${RED}Please run as root or with sudo:${NC}"
  echo "sudo ./install_all.sh"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

run_as_target_user() {
    if [ -n "$SUDO_USER" ]; then
        sudo -u "$TARGET_USER" bash -lc "$1"
    else
        bash -lc "$1"
    fi
}

scan_system() {
    clear
    echo -e "${BLUE}=======================================${NC}"
    echo -e "${BLUE}    ReMap - Installer & Manager [Linux] ${NC}"
    echo -e "${BLUE}=======================================${NC}"
    echo ""
    echo "Scanning system dependencies..."
    
    HAS_PKGS=0
    HAS_GIT=0
    HAS_FFMPEG=0
    HAS_COLMAP=0
    HAS_GLOMAP=0
    HAS_VENV=0
    HAS_PIP_REQ=0
    HAS_SUPERGLUE=0
    HAS_NODE=0
    HAS_NPM=0
    HAS_FRONTEND=0
    HAS_CARGO=0
    HAS_RUST_DEPS=0

    # 1. System packages (basic dev tools)
    if dpkg -l | grep -q "build-essential" \
        && dpkg -l | grep -q "cmake" \
        && dpkg -l | grep -q "libwebkit2gtk-4.1-dev"; then
        HAS_PKGS=1
    fi

    # 2. Git
    if command -v git &> /dev/null; then HAS_GIT=1; fi

    # 3. FFmpeg
    if command -v ffmpeg &> /dev/null; then HAS_FFMPEG=1; fi

    # 4. COLMAP
    if command -v colmap &> /dev/null; then HAS_COLMAP=1; fi

    # 5. GLOMAP
    if command -v glomap &> /dev/null; then HAS_GLOMAP=1; fi

    # 6. Venv
    if [ -f ".venv/bin/activate" ]; then HAS_VENV=1; fi

    # 7. Pip requirements
    if [ $HAS_VENV -eq 1 ]; then
        if run_as_target_user "cd '$SCRIPT_DIR' && .venv/bin/python3 -c 'import cv2, hloc, kornia, loma, numpy, psutil, pycolmap, torch, torchvision; import flask, matplotlib, PIL, requests, scipy, tqdm; import OpenImageIO'" 2>/dev/null; then
            HAS_PIP_REQ=1
        fi
    fi

    # 8. SuperGlue
    if [ -d "SuperGluePretrainedNetwork" ]; then HAS_SUPERGLUE=1; fi

    # 9. Node.js 20+ and npm
    if command -v node &> /dev/null; then
        NODE_MAJOR="$(node -p "parseInt(process.versions.node.split('.')[0], 10)" 2>/dev/null || echo 0)"
        if [ "${NODE_MAJOR:-0}" -ge 20 ]; then HAS_NODE=1; fi
    fi
    if command -v npm &> /dev/null; then HAS_NPM=1; fi

    # 10. Frontend npm packages
    if [ $HAS_NODE -eq 1 ] && [ $HAS_NPM -eq 1 ]; then
        if run_as_target_user "cd '$SCRIPT_DIR' && npm ls --depth=0 >/dev/null 2>&1"; then
            HAS_FRONTEND=1
        fi
    fi

    # 11. Rust/Cargo dependencies for Tauri
    if run_as_target_user "{ source ~/.cargo/env >/dev/null 2>&1 || true; command -v cargo >/dev/null 2>&1; }"; then
        HAS_CARGO=1
    fi
    if [ $HAS_CARGO -eq 1 ]; then
        if run_as_target_user "cd '$SCRIPT_DIR' && { source ~/.cargo/env >/dev/null 2>&1 || true; cargo metadata --manifest-path src-tauri/Cargo.toml --locked --offline --format-version 1 >/dev/null 2>&1; }"; then
            HAS_RUST_DEPS=1
        fi
    fi
}

display_menu() {
    clear
    echo -e "${BLUE}=======================================${NC}"
    echo -e "${BLUE}    ReMap - Installer & Manager [Linux] ${NC}"
    echo -e "${BLUE}=======================================${NC}"
    echo ""
    echo "  Current Status:"
    echo ""

    # Status formatting helper
    fmt_item() {
        if [ $2 -eq 1 ]; then
            echo -e "  [$1] $3 \t: ${GREEN}[OK]${NC}"
        else
            echo -e "  [$1] $3 \t: ${RED}[MISSING]${NC}"
        fi
    }
    
    fmt_item "1" $HAS_PKGS "Build tools "
    fmt_item "2" $HAS_GIT "Git         "
    fmt_item "3" $HAS_FFMPEG "FFmpeg      "
    fmt_item "4" $HAS_COLMAP "COLMAP      "
    if [ $HAS_GLOMAP -eq 1 ]; then
        echo -e "  [5] GLOMAP      \t: ${GREEN}[OK (Optional)]${NC}"
    else
        echo -e "  [5] GLOMAP      \t: ${RED}[MISSING (Optional)]${NC}"
    fi
    fmt_item "6" $HAS_VENV "Python venv "
    fmt_item "7" $HAS_PIP_REQ "PIP packages"
    fmt_item "8" $HAS_SUPERGLUE "SuperGlue   "
    fmt_item "9" $HAS_NODE "Node.js/npm "
    fmt_item "10" $HAS_FRONTEND "Frontend    "
    if [ $HAS_CARGO -eq 1 ] && [ $HAS_RUST_DEPS -eq 1 ]; then
        echo -e "  [11] Tauri/Rust \t: ${GREEN}[OK]${NC}"
    else
        echo -e "  [11] Tauri/Rust \t: ${RED}[MISSING]${NC}"
    fi

    echo ""
    echo "  Actions:"
    echo "  [A] Auto-Install missing components"
    echo "  [F] Full Reinstall (Force all)"
    echo "  [L] Launch ReMap GUI"
    echo "  [Q] Quit"
    echo ""
    echo "  Type 1-11 to reinstall/configure a specific component."
    echo ""
}

# --- Installation blocks ---

install_pkgs() {
    echo -e "\n${GREEN}--- [1/11] Build Tools ---${NC}"
    apt-get update
    apt-get install -y build-essential cmake ninja-build clang \
        ca-certificates curl wget file gnupg \
        libboost-all-dev libgoogle-glog-dev libgflags-dev libceres-dev \
        libfreeimage-dev libglew-dev qtbase5-dev libqt5opengl5-dev \
        libflann-dev libopencv-dev libeigen3-dev libmetis-dev \
        libsqlite3-dev libcgal-dev python3-pip python3-venv \
        libwebkit2gtk-4.1-dev libxdo-dev libssl-dev \
        libayatana-appindicator3-dev librsvg2-dev
    read -p "Press Enter to continue..."
}

install_git() {
    echo -e "\n${GREEN}--- [2/11] Git ---${NC}"
    apt-get update
    apt-get install -y git
    read -p "Press Enter to continue..."
}

install_ffmpeg() {
    echo -e "\n${GREEN}--- [3/11] FFmpeg ---${NC}"
    apt-get update
    apt-get install -y ffmpeg
    read -p "Press Enter to continue..."
}

install_colmap() {
    echo -e "\n${GREEN}--- [4/11] COLMAP ---${NC}"
    apt-get update
    apt-get install -y colmap || echo -e "${YELLOW}COLMAP package not found in repos.${NC}"
    read -p "Press Enter to continue..."
}

install_glomap() {
    echo -e "\n${GREEN}--- [5/11] GLOMAP ---${NC}"
    BUILD_DIR=$(mktemp -d)
    if [ ! -d "$BUILD_DIR" ]; then
        echo -e "${RED}Failed to create temporary directory.${NC}"
        return 1
    fi
    cd "$BUILD_DIR" || { echo -e "${RED}Failed to enter temporary directory.${NC}"; return 1; }
    echo "Cloning GLOMAP..."
    git clone --recursive https://github.com/colmap/glomap.git
    cd glomap
    git checkout 99806d0869f802fad218516a2e027793e7ca687d
    git submodule update --init --recursive
    echo "Configuring GLOMAP..."
    cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
    echo "Building GLOMAP (this may take a while)..."
    ninja -C build
    echo "Installing GLOMAP..."
    ninja -C build install
    cd "$SCRIPT_DIR"
    rm -rf "$BUILD_DIR"
    echo "GLOMAP installed successfully!"
    read -p "Press Enter to continue..."
}

install_venv() {
    echo -e "\n${GREEN}--- [6/11] Python Virtual Environment ---${NC}"
    if [ -d ".venv" ]; then
        echo "Removing existing virtual environment..."
        rm -rf .venv
    fi
    echo "Creating virtual environment..."
    python3 -m venv .venv
    if [ ! -z "$SUDO_USER" ]; then
        chown -R $SUDO_USER:$SUDO_USER .venv
    fi
    echo "Virtual environment created."
    read -p "Press Enter to continue..."
}

install_pip() {
    echo -e "\n${GREEN}--- [7/11] PIP Packages ---${NC}"
    if [ ! -d ".venv" ]; then
        echo -e "${RED}ERROR: Virtual environment missing. Install it first.${NC}"
        read -p "Press Enter to continue..."
        return
    fi
    PIP_LOG_DIR="$SCRIPT_DIR/backend_state/install_logs"
    PIP_LOG="$PIP_LOG_DIR/pip_install.log"
    mkdir -p "$PIP_LOG_DIR"
    touch "$PIP_LOG"
    if [ -n "$SUDO_USER" ]; then
        chown -R "$TARGET_USER":"$TARGET_USER" "$PIP_LOG_DIR"
    fi
    echo "Writing pip install log to: $PIP_LOG"
    echo "==== ReMap pip install $(date) ====" > "$PIP_LOG"
    echo "Installing/Upgrading pip requirements..."
    run_as_target_user "cd '$SCRIPT_DIR' && set -o pipefail && .venv/bin/pip install --upgrade pip 2>&1 | tee -a '$PIP_LOG'"
    run_as_target_user "cd '$SCRIPT_DIR' && set -o pipefail && .venv/bin/pip install -r requirements.txt 2>&1 | tee -a '$PIP_LOG'"
    run_as_target_user "cd '$SCRIPT_DIR' && set -o pipefail && .venv/bin/pip install --ignore-requires-python dataclasses==0.8 2>&1 | tee -a '$PIP_LOG'"
    echo "Installing pinned LoMa from official repository (--no-deps; dependencies are pinned in requirements.txt)..."
    run_as_target_user "cd '$SCRIPT_DIR' && set -o pipefail && .venv/bin/pip install --no-deps --force-reinstall 'git+https://github.com/davnords/LoMa.git@9105854833f55d18194d0505d913f0a74b194ef0#egg=lomatch' 2>&1 | tee -a '$PIP_LOG'"
    echo "Verifying Python dependency consistency..."
    run_as_target_user "cd '$SCRIPT_DIR' && set -o pipefail && .venv/bin/python3 -m pip check 2>&1 | tee -a '$PIP_LOG'"
    echo "Verifying required imports..."
    run_as_target_user "cd '$SCRIPT_DIR' && set -o pipefail && .venv/bin/python3 -c 'import cv2, hloc, kornia, loma, numpy, psutil, pycolmap, torch, torchvision; import flask, matplotlib, PIL, requests, scipy, tqdm; import OpenImageIO' 2>&1 | tee -a '$PIP_LOG'"
    echo "PIP packages installed and verified successfully."
    read -p "Press Enter to continue..."
}

install_superglue() {
    echo -e "\n${GREEN}--- [8/11] SuperGluePretrainedNetwork ---${NC}"
    if command -v git &> /dev/null; then
        if [ -d "SuperGluePretrainedNetwork" ]; then
            echo "Removing existing SuperGlue repository..."
            rm -rf SuperGluePretrainedNetwork
        fi
        echo "Cloning SuperGlue repository..."
        sudo -u ${SUDO_USER:-$USER} bash -c "git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git && cd SuperGluePretrainedNetwork && git checkout ddcf11f42e7e0732a0c4607648f9448ea8d73590"
    else
        echo -e "${RED}ERROR: Git must be installed first.${NC}"
    fi
    read -p "Press Enter to continue..."
}

install_node() {
    echo -e "\n${GREEN}--- [9/11] Node.js / npm ---${NC}"
    if command -v node &> /dev/null; then
        NODE_MAJOR="$(node -p "parseInt(process.versions.node.split('.')[0], 10)" 2>/dev/null || echo 0)"
        if [ "${NODE_MAJOR:-0}" -ge 20 ] && command -v npm &> /dev/null; then
            echo "Node.js/npm is already available."
            read -p "Press Enter to continue..."
            return
        fi
    fi

    echo "Installing Node.js 24 LTS via NodeSource..."
    apt-get update
    apt-get install -y ca-certificates curl gnupg
    mkdir -p /etc/apt/keyrings
    rm -f /etc/apt/keyrings/nodesource.gpg
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_24.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list
    apt-get update
    apt-get install -y nodejs
    read -p "Press Enter to continue..."
}

install_frontend() {
    echo -e "\n${GREEN}--- [10/11] Frontend Packages ---${NC}"
    if ! command -v node &> /dev/null || ! command -v npm &> /dev/null; then
        echo "Node.js/npm is missing. Installing Node.js first..."
        install_node
    fi
    if ! command -v npm &> /dev/null; then
        echo -e "${RED}ERROR: npm is still missing. Install Node.js 20+ and rerun this option.${NC}"
        read -p "Press Enter to continue..."
        return
    fi
    NODE_MAJOR="$(node -p "parseInt(process.versions.node.split('.')[0], 10)" 2>/dev/null || echo 0)"
    if [ "${NODE_MAJOR:-0}" -lt 20 ]; then
        echo -e "${RED}ERROR: Node.js 20+ is required. Current major version: ${NODE_MAJOR:-unknown}.${NC}"
        read -p "Press Enter to continue..."
        return
    fi
    if [ -f "package-lock.json" ]; then
        run_as_target_user "cd '$SCRIPT_DIR' && npm ci"
    else
        run_as_target_user "cd '$SCRIPT_DIR' && npm install"
    fi
    read -p "Press Enter to continue..."
}

install_rust() {
    echo -e "\n${GREEN}--- [11/11] Rust Toolchain ---${NC}"
    if run_as_target_user "{ source ~/.cargo/env >/dev/null 2>&1 || true; command -v cargo >/dev/null 2>&1; }"; then
        echo "Rust/Cargo is already available."
        read -p "Press Enter to continue..."
        return
    fi
    apt-get update
    apt-get install -y curl build-essential
    run_as_target_user "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source ~/.cargo/env && rustup default stable"
    read -p "Press Enter to continue..."
}

install_rust_deps() {
    echo -e "\n${GREEN}--- [11/11] Tauri / Cargo Dependencies ---${NC}"
    if ! run_as_target_user "{ source ~/.cargo/env >/dev/null 2>&1 || true; command -v cargo >/dev/null 2>&1; }"; then
        echo "Cargo is missing. Installing Rust first..."
        install_rust
    fi
    run_as_target_user "cd '$SCRIPT_DIR' && { source ~/.cargo/env >/dev/null 2>&1 || true; cargo fetch --manifest-path src-tauri/Cargo.toml --locked; }"
    read -p "Press Enter to continue..."
}

launch_app() {
    clear
    if [ $HAS_VENV -eq 0 ]; then
        echo -e "${RED}ERROR: Virtual environment not setup. Run install first.${NC}"
        read -p "Press Enter to return to menu..."
        return
    fi
    echo -e "${GREEN}Launching ReMap...${NC}"
    
    # Fix ownership just in case before launching
    if [ ! -z "$SUDO_USER" ]; then
        chown -R $SUDO_USER:$SUDO_USER "$SCRIPT_DIR"
    fi

    if [ ! -z "$SUDO_USER" ]; then
        sudo -u $SUDO_USER bash -c "export PYTHONPATH=\"\${PYTHONPATH}:$(pwd)/SuperGluePretrainedNetwork\" && source .venv/bin/activate && python3 ReMap-GUI.py"
    else
        export PYTHONPATH="${PYTHONPATH}:$(pwd)/SuperGluePretrainedNetwork"
        source .venv/bin/activate
        python3 ReMap-GUI.py
    fi
    exit 0
}

# --- Main loop ---
while true; do
    scan_system
    display_menu
    read -p "> Select an option: " CHOICE

    case $CHOICE in
        [Qq]) exit 0 ;;
        [Ll]) launch_app ;;
        [Aa])
            if [ $HAS_PKGS -eq 0 ]; then install_pkgs; fi
            if [ $HAS_GIT -eq 0 ]; then install_git; fi
            if [ $HAS_FFMPEG -eq 0 ]; then install_ffmpeg; fi
            if [ $HAS_COLMAP -eq 0 ]; then install_colmap; fi
            if [ $HAS_GLOMAP -eq 0 ]; then
                clear
                read -p "  Install GLOMAP automatically? (Requires compiling from source) [Y/n]: " YN
                if [[ "${YN:-Y}" =~ ^[Yy]$ ]]; then install_glomap; fi
            fi
            if [ $HAS_VENV -eq 0 ]; then install_venv; fi
            if [ $HAS_PIP_REQ -eq 0 ]; then install_pip; fi
            if [ $HAS_SUPERGLUE -eq 0 ]; then install_superglue; fi
            if [ $HAS_NODE -eq 0 ]; then install_node; fi
            if [ $HAS_FRONTEND -eq 0 ]; then install_frontend; fi
            if [ $HAS_CARGO -eq 0 ]; then install_rust; fi
            if [ $HAS_RUST_DEPS -eq 0 ]; then install_rust_deps; fi
            ;;
        [Ff])
            install_pkgs
            install_git
            install_ffmpeg
            install_colmap
            clear
            read -p "  Install GLOMAP automatically? (Requires compiling from source) [Y/n]: " YN
            if [[ "${YN:-Y}" =~ ^[Yy]$ ]]; then install_glomap; fi
            install_venv
            install_pip
            install_superglue
            install_node
            install_frontend
            install_rust
            install_rust_deps
            ;;
        1) install_pkgs ;;
        2) install_git ;;
        3) install_ffmpeg ;;
        4) install_colmap ;;
        5) install_glomap ;;
        6) install_venv ;;
        7) install_pip ;;
        8) install_superglue ;;
        9) install_node ;;
        10) install_frontend ;;
        11) install_rust; install_rust_deps ;;
        *) ;;
    esac
done
