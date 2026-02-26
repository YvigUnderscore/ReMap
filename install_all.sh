#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for root
if [ "$EUID" -ne 0 ]; then 
  echo -e "${RED}Please run as root or with sudo:${NC}"
  echo "sudo ./install_all.sh"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

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

    # 1. System packages (basic dev tools)
    if dpkg -l | grep -q "build-essential" && dpkg -l | grep -q "cmake"; then HAS_PKGS=1; fi

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
        if sudo -u ${SUDO_USER:-$USER} bash -c ".venv/bin/python3 -c 'import hloc'" 2>/dev/null; then
            HAS_PIP_REQ=1
        fi
    fi

    # 8. SuperGlue
    if [ -d "SuperGluePretrainedNetwork" ]; then HAS_SUPERGLUE=1; fi
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

    echo ""
    echo "  Actions:"
    echo "  [A] Auto-Install missing components"
    echo "  [F] Full Reinstall (Force all)"
    echo "  [L] Launch ReMap GUI"
    echo "  [Q] Quit"
    echo ""
    echo "  Type 1-8 to reinstall/configure a specific component."
    echo ""
}

# --- Installation blocks ---

install_pkgs() {
    echo -e "\n${GREEN}--- [1/8] Build Tools ---${NC}"
    apt-get update
    apt-get install -y build-essential cmake ninja-build clang \
        libboost-all-dev libgoogle-glog-dev libgflags-dev libceres-dev \
        libfreeimage-dev libglew-dev qtbase5-dev libqt5opengl5-dev \
        libflann-dev libopencv-dev libeigen3-dev libmetis-dev \
        libsqlite3-dev libcgal-dev python3-pip python3-venv
    read -p "Press Enter to continue..."
}

install_git() {
    echo -e "\n${GREEN}--- [2/8] Git ---${NC}"
    apt-get update
    apt-get install -y git
    read -p "Press Enter to continue..."
}

install_ffmpeg() {
    echo -e "\n${GREEN}--- [3/8] FFmpeg ---${NC}"
    apt-get update
    apt-get install -y ffmpeg
    read -p "Press Enter to continue..."
}

install_colmap() {
    echo -e "\n${GREEN}--- [4/8] COLMAP ---${NC}"
    apt-get update
    apt-get install -y colmap || echo -e "${YELLOW}COLMAP package not found in repos.${NC}"
    read -p "Press Enter to continue..."
}

install_glomap() {
    echo -e "\n${GREEN}--- [5/8] GLOMAP ---${NC}"
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
    echo -e "\n${GREEN}--- [6/8] Python Virtual Environment ---${NC}"
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
    echo -e "\n${GREEN}--- [7/8] PIP Packages ---${NC}"
    if [ ! -d ".venv" ]; then
        echo -e "${RED}ERROR: Virtual environment missing. Install it first.${NC}"
        read -p "Press Enter to continue..."
        return
    fi
    echo "Installing/Upgrading pip requirements..."
    sudo -u ${SUDO_USER:-$USER} bash -c ".venv/bin/pip install --upgrade pip"
    sudo -u ${SUDO_USER:-$USER} bash -c ".venv/bin/pip install -r requirements.txt"
    read -p "Press Enter to continue..."
}

install_superglue() {
    echo -e "\n${GREEN}--- [8/8] SuperGluePretrainedNetwork ---${NC}"
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
            ;;
        1) install_pkgs ;;
        2) install_git ;;
        3) install_ffmpeg ;;
        4) install_colmap ;;
        5) install_glomap ;;
        6) install_venv ;;
        7) install_pip ;;
        8) install_superglue ;;
        *) ;;
    esac
done
