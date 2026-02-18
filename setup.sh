#!/bin/bash

# ============================================================
# JETSON NANO YOLOV5 + POSTURE DETECTION SETUP SCRIPT
# Handles all dependencies, wheel files, and fixes
# Tested on: Jetson Nano 2GB, JetPack 4.5, L4T R32.5.0
# ============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

WHEELS_DIR="$HOME/.jetson_wheels"
LOG_FILE="$HOME/setup_log_$(date +%Y%m%d_%H%M%S).txt"

# Target versions
TORCH_VERSION="1.7.0"
TORCHVISION_VERSION="0.2.2.post3"
NUMPY_VERSION="1.19.5"

# Torch wheel details
TORCH_WHEEL_NAME="torch-1.7.0-cp36-cp36m-linux_aarch64.whl"
TORCH_WHEEL_URL="https://nvidia.box.com/shared/static/cs3xn3td6sfgtene6jdvsxlr366m2dhq.whl"

log() {
    echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[OK]${NC} $1" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

fail() {
    echo -e "${RED}[FAIL]${NC} $1" | tee -a "$LOG_FILE"
}

header() {
    echo "" | tee -a "$LOG_FILE"
    echo -e "${BOLD}============================================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${BOLD}  $1${NC}" | tee -a "$LOG_FILE"
    echo -e "${BOLD}============================================================${NC}" | tee -a "$LOG_FILE"
}

# ============================================================
# SYSTEM CHECKS
# ============================================================
header "SYSTEM INFORMATION"

log "Hostname: $(hostname)"
log "Date: $(date)"
log "User: $(whoami)"

# Check if Jetson
if [ -f /etc/nv_tegra_release ]; then
    NV_RELEASE=$(cat /etc/nv_tegra_release)
    log "Tegra Release: $NV_RELEASE"
    success "Jetson platform detected"
else
    warn "Not a Jetson platform or nv_tegra_release not found"
fi

# Check architecture
ARCH=$(uname -m)
log "Architecture: $ARCH"
if [ "$ARCH" != "aarch64" ]; then
    fail "Expected aarch64 architecture, got $ARCH"
    fail "This script is for Jetson Nano (ARM64) only"
    exit 1
fi
success "Architecture: aarch64"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
log "Python version: $PYTHON_VERSION"
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -eq 6 ]; then
    success "Python 3.6 detected (compatible)"
else
    warn "Python $PYTHON_VERSION detected. This script targets Python 3.6"
    warn "Some wheel files may not be compatible"
fi

# Check disk space
AVAIL_MB=$(df -m / | tail -1 | awk '{print $4}')
log "Available disk space: ${AVAIL_MB}MB"
if [ "$AVAIL_MB" -lt 500 ]; then
    fail "Less than 500MB free. Need more space."
    echo ""
    echo "Run these to free space:"
    echo "  sudo apt-get clean"
    echo "  sudo apt-get autoremove -y"
    echo "  rm -rf ~/.cache/pip"
    echo "  sudo rm -rf /usr/local/cuda-10.2/samples"
    echo "  sudo rm -rf /usr/local/cuda-10.2/doc"
    exit 1
fi
success "Disk space: ${AVAIL_MB}MB available"

# ============================================================
# CUDA CHECKS
# ============================================================
header "CUDA ENVIRONMENT"

# Find CUDA
CUDA_HOME=""
if [ -d "/usr/local/cuda-10.2" ]; then
    CUDA_HOME="/usr/local/cuda-10.2"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_HOME="/usr/local/cuda"
fi

if [ -n "$CUDA_HOME" ]; then
    success "CUDA found at: $CUDA_HOME"
else
    fail "CUDA not found at /usr/local/cuda-10.2 or /usr/local/cuda"
    warn "PyTorch CUDA support may not work"
fi

# Check for libcurand
CURAND=$(find /usr/local -name "libcurand.so*" 2>/dev/null | head -1)
if [ -n "$CURAND" ]; then
    success "libcurand found: $CURAND"
else
    warn "libcurand not found. Attempting to install..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq libcurand-10-2 2>/dev/null || warn "Could not install libcurand"
    CURAND=$(find /usr/local -name "libcurand.so*" 2>/dev/null | head -1)
    if [ -n "$CURAND" ]; then
        success "libcurand installed: $CURAND"
    else
        fail "libcurand still not found"
    fi
fi

# Check for libcudnn
CUDNN=$(find /usr -name "libcudnn.so*" 2>/dev/null | head -1)
if [ -n "$CUDNN" ]; then
    success "libcudnn found: $CUDNN"
else
    warn "libcudnn not found. Attempting to install..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq libcudnn8 2>/dev/null || warn "Could not install libcudnn8"
    CUDNN=$(find /usr -name "libcudnn.so*" 2>/dev/null | head -1)
    if [ -n "$CUDNN" ]; then
        success "libcudnn installed: $CUDNN"
    else
        fail "libcudnn still not found. PyTorch may not load."
        fail "Try: sudo apt-get install libcudnn8"
    fi
fi

# Set environment variables
log "Setting CUDA environment variables..."

export CUDA_HOME="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:${CUDA_HOME}/targets/aarch64-linux/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export CPATH="${CUDA_HOME}/targets/aarch64-linux/include:${CPATH}"

# Make permanent
BASHRC="$HOME/.bashrc"
add_to_bashrc() {
    if ! grep -q "$1" "$BASHRC" 2>/dev/null; then
        echo "$1" >> "$BASHRC"
        log "Added to .bashrc: $1"
    fi
}

add_to_bashrc "export CUDA_HOME=${CUDA_HOME}"
add_to_bashrc 'export PATH=${CUDA_HOME}/bin:${PATH}'
add_to_bashrc 'export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:${CUDA_HOME}/targets/aarch64-linux/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}'
add_to_bashrc 'export CPATH=${CUDA_HOME}/targets/aarch64-linux/include:${CPATH}'

success "CUDA environment configured"

# ============================================================
# SYSTEM PACKAGES
# ============================================================
header "SYSTEM PACKAGES"

log "Updating package lists..."
sudo apt-get update -qq 2>/dev/null

SYSTEM_PACKAGES=(
    python3-pip
    python3-dev
    python3-numpy
    python3-matplotlib
    python3-scipy
    libopenblas-base
    libopenblas-dev
    libjpeg-dev
    zlib1g-dev
    libpython3-dev
    libavcodec-dev
    libavformat-dev
    libswscale-dev
    libfreetype6-dev
)

for pkg in "${SYSTEM_PACKAGES[@]}"; do
    if dpkg -l | grep -q "^ii  $pkg "; then
        success "$pkg already installed"
    else
        log "Installing $pkg..."
        sudo apt-get install -y -qq "$pkg" 2>/dev/null
        if dpkg -l | grep -q "^ii  $pkg "; then
            success "$pkg installed"
        else
            warn "Could not install $pkg"
        fi
    fi
done

# ============================================================
# PIP SETUP
# ============================================================
header "PIP SETUP"

# Upgrade pip
log "Checking pip..."
PIP_VERSION=$(pip3 --version 2>&1 | awk '{print $2}')
log "Current pip version: $PIP_VERSION"
pip3 install --upgrade pip 2>/dev/null || warn "Could not upgrade pip"

# Create wheels directory
mkdir -p "$WHEELS_DIR"
log "Wheels directory: $WHEELS_DIR"

# ============================================================
# NUMPY
# ============================================================
header "NUMPY ($NUMPY_VERSION)"

check_numpy() {
    python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null
}

CURRENT_NUMPY=$(check_numpy)

if [ "$CURRENT_NUMPY" = "$NUMPY_VERSION" ]; then
    success "NumPy $NUMPY_VERSION already installed"
    # Verify it actually works (no illegal instruction)
    if python3 -c "import numpy; numpy.array([1,2,3])" 2>/dev/null; then
        success "NumPy functional test passed"
    else
        fail "NumPy installed but crashes (wrong architecture?)"
        log "Reinstalling NumPy..."
        pip3 uninstall numpy -y 2>/dev/null
        CURRENT_NUMPY=""
    fi
fi

if [ "$CURRENT_NUMPY" != "$NUMPY_VERSION" ]; then
    log "Current NumPy: ${CURRENT_NUMPY:-not installed}"
    log "Removing incompatible numpy versions..."
    pip3 uninstall numpy -y 2>/dev/null

    # Remove any cached bad numpy
    rm -rf "$HOME/.local/lib/python3.6/site-packages/numpy"* 2>/dev/null

    log "Installing NumPy $NUMPY_VERSION..."

    # Method 1: pip install specific version
    if pip3 install "numpy==$NUMPY_VERSION" --no-cache-dir 2>/dev/null; then
        if python3 -c "import numpy; numpy.array([1,2,3]); print('OK')" 2>/dev/null; then
            success "NumPy $NUMPY_VERSION installed via pip"
        else
            fail "NumPy from pip crashes. Trying system package..."
            pip3 uninstall numpy -y 2>/dev/null
            rm -rf "$HOME/.local/lib/python3.6/site-packages/numpy"* 2>/dev/null

            # Method 2: Use system numpy
            sudo apt-get install -y python3-numpy 2>/dev/null
            export PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH}"
            add_to_bashrc 'export PYTHONPATH=/usr/lib/python3/dist-packages:${PYTHONPATH}'

            if python3 -c "import numpy; print('OK')" 2>/dev/null; then
                success "NumPy installed via system package"
            else
                fail "NumPy installation failed"
            fi
        fi
    else
        warn "pip install numpy failed, using system package"
        sudo apt-get install -y python3-numpy 2>/dev/null
        export PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH}"
        add_to_bashrc 'export PYTHONPATH=/usr/lib/python3/dist-packages:${PYTHONPATH}'
    fi
fi

FINAL_NUMPY=$(check_numpy)
log "Final NumPy version: ${FINAL_NUMPY:-FAILED}"

# ============================================================
# PYTORCH
# ============================================================
header "PYTORCH ($TORCH_VERSION)"

check_torch() {
    python3 -c "import torch; print(torch.__version__)" 2>/dev/null
}

check_torch_cuda() {
    python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null
}

CURRENT_TORCH=$(check_torch)

if [ "$CURRENT_TORCH" = "$TORCH_VERSION" ]; then
    success "PyTorch $TORCH_VERSION already installed"
    CUDA_STATUS=$(check_torch_cuda)
    if [ "$CUDA_STATUS" = "True" ]; then
        success "PyTorch CUDA: available"
    else
        warn "PyTorch CUDA: not available"
    fi
else
    log "Current PyTorch: ${CURRENT_TORCH:-not installed}"

    # Remove wrong versions
    if [ -n "$CURRENT_TORCH" ] && [ "$CURRENT_TORCH" != "$TORCH_VERSION" ]; then
        log "Removing PyTorch $CURRENT_TORCH..."
        pip3 uninstall torch -y 2>/dev/null
        rm -rf "$HOME/.local/lib/python3.6/site-packages/torch"* 2>/dev/null
    fi

    # Check if wheel exists
    WHEEL_PATH="$WHEELS_DIR/$TORCH_WHEEL_NAME"

    if [ -f "$WHEEL_PATH" ]; then
        success "Torch wheel found: $WHEEL_PATH"
    else
        # Search common locations
        FOUND_WHEEL=$(find "$HOME" -name "$TORCH_WHEEL_NAME" -o -name "cs3xn3td6sfgtene6jdvsxlr366m2dhq.whl" 2>/dev/null | head -1)

        if [ -n "$FOUND_WHEEL" ]; then
            log "Found wheel at: $FOUND_WHEEL"
            cp "$FOUND_WHEEL" "$WHEEL_PATH"
            # Rename if it has box.com name
            if echo "$FOUND_WHEEL" | grep -q "cs3xn3td6sf"; then
                cp "$FOUND_WHEEL" "$WHEEL_PATH"
            fi
            success "Wheel copied to: $WHEEL_PATH"
        else
            log "Downloading PyTorch $TORCH_VERSION wheel..."
            log "URL: $TORCH_WHEEL_URL"
            wget -q --show-progress -O "$WHEEL_PATH" "$TORCH_WHEEL_URL" 2>&1 | tee -a "$LOG_FILE"

            if [ -f "$WHEEL_PATH" ] && [ -s "$WHEEL_PATH" ]; then
                success "Download complete: $(du -h "$WHEEL_PATH" | awk '{print $1}')"
            else
                fail "Download failed"
                # Try alternative
                log "Trying NVIDIA developer download..."
                ALT_URL="https://developer.download.nvidia.com/compute/redist/jp/v45/pytorch/torch-1.8.0-cp36-cp36m-linux_aarch64.whl"
                TORCH_VERSION="1.8.0"
                TORCH_WHEEL_NAME="torch-1.8.0-cp36-cp36m-linux_aarch64.whl"
                WHEEL_PATH="$WHEELS_DIR/$TORCH_WHEEL_NAME"
                wget -q --show-progress -O "$WHEEL_PATH" "$ALT_URL" 2>&1 | tee -a "$LOG_FILE"
            fi
        fi
    fi

    # Install wheel
    if [ -f "$WHEEL_PATH" ] && [ -s "$WHEEL_PATH" ]; then
        log "Installing PyTorch from wheel..."
        pip3 install "$WHEEL_PATH" 2>&1 | tee -a "$LOG_FILE"

        INSTALLED_TORCH=$(check_torch)
        if [ -n "$INSTALLED_TORCH" ]; then
            success "PyTorch $INSTALLED_TORCH installed"
            CUDA_STATUS=$(check_torch_cuda)
            log "CUDA available: $CUDA_STATUS"
        else
            fail "PyTorch installation failed"
            log "Checking for errors..."
            python3 -c "import torch" 2>&1 | tee -a "$LOG_FILE"
        fi
    else
        fail "No valid wheel file found"
        log "Please manually download PyTorch wheel for Jetson"
        log "Place it at: $WHEEL_PATH"
    fi
fi

# ============================================================
# TORCHVISION
# ============================================================
header "TORCHVISION ($TORCHVISION_VERSION)"

check_torchvision() {
    python3 -c "import torchvision; print(torchvision.__version__)" 2>/dev/null
}

CURRENT_TV=$(check_torchvision)

if [ "$CURRENT_TV" = "$TORCHVISION_VERSION" ]; then
    success "TorchVision $TORCHVISION_VERSION already installed"
else
    log "Current TorchVision: ${CURRENT_TV:-not installed}"

    if [ -n "$CURRENT_TV" ]; then
        log "Removing TorchVision $CURRENT_TV..."
        pip3 uninstall torchvision -y 2>/dev/null
    fi

    log "Installing TorchVision $TORCHVISION_VERSION..."
    pip3 install "torchvision==$TORCHVISION_VERSION" --no-cache-dir 2>&1 | tee -a "$LOG_FILE"

    INSTALLED_TV=$(check_torchvision)
    if [ -n "$INSTALLED_TV" ]; then
        success "TorchVision $INSTALLED_TV installed"
    else
        fail "TorchVision installation failed"
    fi
fi

# ============================================================
# OPENCV CHECK
# ============================================================
header "OPENCV"

check_opencv() {
    python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null
}

CURRENT_CV=$(check_opencv)
if [ -n "$CURRENT_CV" ]; then
    success "OpenCV $CURRENT_CV installed"
else
    log "Installing OpenCV..."
    sudo apt-get install -y python3-opencv 2>/dev/null
    pip3 install opencv-python-headless 2>/dev/null
    CURRENT_CV=$(check_opencv)
    if [ -n "$CURRENT_CV" ]; then
        success "OpenCV $CURRENT_CV installed"
    else
        fail "OpenCV installation failed"
    fi
fi

# ============================================================
# ADDITIONAL PYTHON PACKAGES
# ============================================================
header "ADDITIONAL PACKAGES"

PIP_PACKAGES=(
    "pandas"
    "pyyaml"
    "tqdm"
    "flask"
    "matplotlib"
    "seaborn"
    "scipy"
    "Pillow"
)

for pkg in "${PIP_PACKAGES[@]}"; do
    if python3 -c "import ${pkg%%=*}" 2>/dev/null; then
        VER=$(python3 -c "import ${pkg%%=*}; print(getattr(${pkg%%=*}, '__version__', 'unknown'))" 2>/dev/null)
        success "$pkg ($VER) already available"
    else
        log "Installing $pkg..."
        pip3 install "$pkg" --no-cache-dir 2>/dev/null
        if python3 -c "import ${pkg%%=*}" 2>/dev/null; then
            success "$pkg installed"
        else
            warn "Could not install $pkg"
        fi
    fi
done

# Fix flask import name
python3 -c "import flask" 2>/dev/null || pip3 install flask --no-cache-dir 2>/dev/null
# Fix PIL import name
python3 -c "from PIL import Image" 2>/dev/null || pip3 install Pillow --no-cache-dir 2>/dev/null
# Fix yaml import name
python3 -c "import yaml" 2>/dev/null || pip3 install pyyaml --no-cache-dir 2>/dev/null

# ============================================================
# YOLOV5 NMS FIX (torchvision.ops.nms not in 0.2.2)
# ============================================================
header "YOLOV5 COMPATIBILITY FIX"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERAL_PY="$SCRIPT_DIR/utils/general.py"

if [ -f "$GENERAL_PY" ]; then
    if grep -q "torchvision.ops.nms" "$GENERAL_PY"; then
        log "Patching torchvision.ops.nms in general.py..."

        # Backup
        cp "$GENERAL_PY" "${GENERAL_PY}.backup_$(date +%s)"

        # Add custom NMS function after imports
        # Check if already patched
        if grep -q "def nms_pytorch" "$GENERAL_PY"; then
            success "NMS patch already applied"
        else
            # Insert nms_pytorch function after the os.environ line
            sed -i "/os.environ\['NUMEXPR_MAX_THREADS'\]/a\\
\\
def nms_pytorch(boxes, scores, iou_thres):\\
    x1 = boxes[:, 0]\\
    y1 = boxes[:, 1]\\
    x2 = boxes[:, 2]\\
    y2 = boxes[:, 3]\\
    areas = (x2 - x1) * (y2 - y1)\\
    order = scores.argsort(descending=True)\\
    keep = []\\
    while order.numel() > 0:\\
        if order.numel() == 1:\\
            keep.append(order.item())\\
            break\\
        i = order[0].item()\\
        keep.append(i)\\
        xx1 = torch.max(x1[order[1:]], x1[i])\\
        yy1 = torch.max(y1[order[1:]], y1[i])\\
        xx2 = torch.min(x2[order[1:]], x2[i])\\
        yy2 = torch.min(y2[order[1:]], y2[i])\\
        w = torch.clamp(xx2 - xx1, min=0)\\
        h = torch.clamp(yy2 - yy1, min=0)\\
        inter = w * h\\
        iou = inter / (areas[i] + areas[order[1:]] - inter)\\
        inds = torch.where(iou <= iou_thres)[0]\\
        order = order[inds + 1]\\
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)\\
" "$GENERAL_PY"

            # Replace the call
            sed -i 's/torchvision\.ops\.nms(boxes, scores, iou_thres)/nms_pytorch(boxes, scores, iou_thres)/g' "$GENERAL_PY"

            if grep -q "nms_pytorch" "$GENERAL_PY"; then
                success "NMS patch applied successfully"
            else
                fail "NMS patch failed"
            fi
        fi
    else
        if grep -q "nms_pytorch" "$GENERAL_PY"; then
            success "NMS already patched (no torchvision.ops.nms found)"
        else
            success "general.py does not use torchvision.ops.nms"
        fi
    fi
else
    warn "utils/general.py not found at $GENERAL_PY"
fi

# ============================================================
# YOLOV5 WEIGHTS
# ============================================================
header "YOLOV5 WEIGHTS"

WEIGHTS_DIR="$SCRIPT_DIR/weights"
mkdir -p "$WEIGHTS_DIR"

if [ -f "$WEIGHTS_DIR/yolov5s.pt" ]; then
    SIZE=$(du -h "$WEIGHTS_DIR/yolov5s.pt" | awk '{print $1}')
    success "yolov5s.pt found ($SIZE)"
else
    log "Downloading yolov5s.pt..."
    wget -q --show-progress -O "$WEIGHTS_DIR/yolov5s.pt" \
        "https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt" 2>&1 | tee -a "$LOG_FILE"
    if [ -f "$WEIGHTS_DIR/yolov5s.pt" ]; then
        success "yolov5s.pt downloaded"
    else
        fail "Could not download yolov5s.pt"
    fi
fi

# ============================================================
# ELEMENTS MODULE CHECK
# ============================================================
header "YOLOV5 ELEMENTS MODULE"

if [ -f "$SCRIPT_DIR/elements/yolo.py" ]; then
    success "elements/yolo.py found"
else
    fail "elements/yolo.py not found"
    warn "Make sure you cloned the JetsonYoloV5 repository"
fi

if [ -d "$SCRIPT_DIR/models" ]; then
    success "models/ directory found"
else
    fail "models/ directory not found"
fi

if [ -d "$SCRIPT_DIR/utils" ]; then
    success "utils/ directory found"
else
    fail "utils/ directory not found"
fi

# ============================================================
# CAMERA CHECK
# ============================================================
header "CAMERA"

if [ -e "/dev/video0" ]; then
    success "Camera detected at /dev/video0"
else
    warn "No camera at /dev/video0"
    ls /dev/video* 2>/dev/null && success "Camera found at: $(ls /dev/video*)" || warn "No camera devices found. Connect USB camera."
fi

# ============================================================
# FINAL VERIFICATION
# ============================================================
header "FINAL VERIFICATION"

echo ""
log "Running comprehensive import test..."

python3 << 'PYTEST'
import sys

tests = []

def test(name, func):
    try:
        result = func()
        print("  [OK]   {} -> {}".format(name, result))
        tests.append((name, True, result))
    except Exception as e:
        print("  [FAIL] {} -> {}".format(name, str(e)[:60]))
        tests.append((name, False, str(e)[:60]))

# Core
test("Python", lambda: sys.version.split()[0])
test("NumPy", lambda: __import__('numpy').__version__)
test("OpenCV", lambda: __import__('cv2').__version__)

# PyTorch
test("PyTorch", lambda: __import__('torch').__version__)
test("CUDA Available", lambda: str(__import__('torch').cuda.is_available()))

# ML
test("TorchVision", lambda: __import__('torchvision').__version__)
test("Pandas", lambda: __import__('pandas').__version__)
test("Matplotlib", lambda: __import__('matplotlib').__version__)
test("YAML", lambda: __import__('yaml').__version__)
test("Flask", lambda: __import__('flask').__version__)
test("PIL", lambda: __import__('PIL').__version__)
test("SciPy", lambda: __import__('scipy').__version__)

# Functional tests
test("NumPy Array", lambda: str(__import__('numpy').array([1,2,3]).shape))
test("Torch Tensor", lambda: str(__import__('torch').tensor([1,2,3]).shape))

# Camera
def cam_test():
    import cv2
    cap = cv2.VideoCapture(0)
    opened = cap.isOpened()
    cap.release()
    return "OK" if opened else "NOT AVAILABLE"
test("Camera", cam_test)

# Summary
passed = sum(1 for _, ok, _ in tests if ok)
total = len(tests)
print("")
print("  Results: {}/{} passed".format(passed, total))

failed = [(n, r) for n, ok, r in tests if not ok]
if failed:
    print("")
    print("  Failed tests:")
    for name, reason in failed:
        print("    - {}: {}".format(name, reason))
    sys.exit(1)
else:
    sys.exit(0)
PYTEST

RESULT=$?

echo ""

if [ $RESULT -eq 0 ]; then
    header "SETUP COMPLETE"
    echo ""
    echo -e "${GREEN}${BOLD}  All checks passed!${NC}"
    echo ""
    echo "  To run YOLOv5 detection:"
    echo "    cd $(pwd)"
    echo "    python3 JetsonYolo.py"
    echo ""
    echo "  To run posture detection:"
    echo "    python3 posture.py"
    echo "    Open browser: http://$(hostname -I | awk '{print $1}'):5000"
    echo ""
    echo "  Log file: $LOG_FILE"
    echo ""
else
    header "SETUP INCOMPLETE"
    echo ""
    echo -e "${YELLOW}  Some tests failed. Check log: $LOG_FILE${NC}"
    echo ""
    echo "  Common fixes:"
    echo "    - Reconnect camera if camera test failed"
    echo "    - Run 'source ~/.bashrc' then re-run this script"
    echo "    - Check disk space: df -h"
    echo ""
fi

echo -e "${BOLD}============================================================${NC}"
echo ""
