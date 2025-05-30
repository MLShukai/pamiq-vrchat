#!/bin/bash

# PAMIQ VRChat Sample Runner for Linux
# This script automatically sets up dependencies and runs the PAMIQ VRChat sample

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_error() {
    echo -e "${RED}❌ ERROR: $1${NC}" >&2
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  WARNING: $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if process is running
process_exists() {
    pgrep -f "$1" >/dev/null 2>&1
}

# =============================================================================
#                              DEPENDENCY RESOLUTION
# =============================================================================

print_info "Checking dependencies..."

# 1. Check and install uv
if ! command_exists uv; then
    print_warning "uv is not installed."
    read -p "Do you want to install uv? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        if [ $? -ne 0 ]; then
            print_error "Failed to install uv"
            exit 1
        fi
        print_success "uv installed successfully"
    else
        print_error "uv installation was declined. Cannot proceed."
        exit 1
    fi
fi

# 2. Update PATH for uv
export PATH="$HOME/.local/bin:$PATH"

# Verify uv is accessible
if ! command_exists uv; then
    print_error "uv is still not accessible after installation. Please check your PATH."
    exit 1
fi

print_success "uv is available"

# 3. Install dependencies
print_info "Installing Python dependencies..."
if ! uv sync --all-extras; then
    print_error "Failed to install dependencies with uv sync --all-extras"
    exit 1
fi
print_success "Dependencies installed"

# 4. Check CUDA availability
print_info "Checking CUDA availability..."
CUDA_CHECK=$(uv run python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$CUDA_CHECK" != "True" ]; then
    print_error "CUDA is not available. torch.cuda.is_available() returned: $CUDA_CHECK"
    print_error "Please ensure you have a CUDA-compatible GPU and PyTorch with CUDA support."
    exit 1
fi
print_success "CUDA is available"

# =============================================================================
#                          SOFTWARE STARTUP VERIFICATION
# =============================================================================

print_info "Checking required software..."

# 1. Check VRChat process
if ! process_exists "VRChat.exe"; then
    print_warning "VRChat.exe process can not found!"
    print_warning "Please start VRChat from Steam before running this script."
    print_warning "Launch Steam → Library → VRChat → Play"
else
    print_success "VRChat is running"
fi

# 2. Check OBS
if ! process_exists "obs"; then
    print_warning "OBS is not running!"
    print_warning "Please:"
    print_warning "1. Install OBS Studio if not installed"
    print_warning "2. Start OBS"
    print_warning "3. Set up VRChat window capture"
    print_warning "4. Enable Virtual Camera (Start Virtual Camera button)"
else
    print_success "OBS is running"
fi

# =============================================================================
#                                    LAUNCH
# =============================================================================

# 1. Start pamiq-kbctl if desktop environment is available
if [ -n "$DISPLAY" ] || [ -n "$WAYLAND_DISPLAY" ]; then
    print_info "Starting pamiq-kbctl in background..."

    # Check if already running
    if process_exists "pamiq-kbctl"; then
        print_warning "pamiq-kbctl is already running"
    else
        # Try to start in a new terminal if available
        if command_exists gnome-terminal; then
            gnome-terminal -- bash -c "uv run pamiq-kbctl; read -p 'Press Enter to close...'"
        elif command_exists konsole; then
            konsole -e bash -c "uv run pamiq-kbctl; read -p 'Press Enter to close...'"
        elif command_exists xterm; then
            xterm -e bash -c "uv run pamiq-kbctl; read -p 'Press Enter to close...'" &
        else
            # Fallback to background process
            print_warning "No suitable terminal found, starting pamiq-kbctl in background"
            nohup uv run pamiq-kbctl > /dev/null 2>&1 &
        fi
        print_success "pamiq-kbctl started"
    fi
else
    print_warning "No desktop environment detected, skipping pamiq-kbctl"
fi

# 2. Model size selection
echo
print_info "Model size selection:"
echo "Available sizes:"
echo "  1) tiny    - ~2.5GiB VRAM (recommended for most systems)"
echo "  2) small   - ~4GiB VRAM"
echo "  3) medium  - ~6.5GiB VRAM"
echo "  4) large   - ~12GiB VRAM"
echo "  5) huge    - ~23GiB VRAM"
echo "  6) [Enter] - Use default (tiny)"
echo

read -p "Select model size (1-6 or Enter for default): " -n 1 -r
echo

MODEL_SIZE=""
case $REPLY in
    1) MODEL_SIZE="tiny" ;;
    2) MODEL_SIZE="small" ;;
    3) MODEL_SIZE="medium" ;;
    4) MODEL_SIZE="large" ;;
    5) MODEL_SIZE="huge" ;;
    6|"") MODEL_SIZE="" ;;
    *)
        print_warning "Invalid selection, using default (tiny)"
        MODEL_SIZE=""
        ;;
esac

# 3. Run the sample
print_info "Starting PAMIQ VRChat sample..."

if [ -n "$MODEL_SIZE" ]; then
    print_info "Running with model size: $MODEL_SIZE"
    uv run python src/run_sample.py --model-size "$MODEL_SIZE"
else
    print_info "Running with default model size"
    uv run python src/run_sample.py
fi

print_success "PAMIQ VRChat sample completed"
