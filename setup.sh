#!/bin/bash
# SAM3 Linear Probing - Setup Script
# This script sets up the environment for local training

set -e  # Exit on error

echo "======================================================================"
echo "SAM3 Linear Probing - Setup Script"
echo "======================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Select Python interpreter (prefer 3.11 for SAM3 compatibility)
if [ -n "${PYTHON_BIN:-}" ]; then
    if ! command -v "$PYTHON_BIN" &> /dev/null; then
        echo -e "${RED}Error: PYTHON_BIN='$PYTHON_BIN' not found${NC}"
        exit 1
    fi
elif command -v python3.11 &> /dev/null; then
    PYTHON_BIN="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
else
    echo -e "${RED}Error: No python interpreter found${NC}"
    exit 1
fi

# Check Python version
echo "Checking Python version..."
python_version=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
python_mm=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python executable: $(command -v $PYTHON_BIN)"
echo "  Python version: $python_version"

if ! $PYTHON_BIN -c "import sys; exit(0 if (3, 8) <= sys.version_info[:2] <= (3, 12) else 1)"; then
    echo -e "${RED}Error: Python 3.8-3.12 required (3.11 recommended)${NC}"
    echo -e "${YELLOW}Hint: run with PYTHON_BIN=python3.11 ./setup.sh${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Check CUDA availability
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected - will use CPU (much slower)${NC}"
fi
echo ""

# Install system build dependencies required for native Python packages
echo "Checking system build dependencies..."
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
elif command -v sudo &> /dev/null; then
    SUDO="sudo"
else
    SUDO=""
    echo -e "${YELLOW}⚠ Not running as root and 'sudo' is unavailable.${NC}"
    echo -e "${YELLOW}  System dependencies may fail to install automatically.${NC}"
fi

if command -v apt-get &> /dev/null; then
    echo "  Using apt package manager"
    $SUDO apt-get update
    python_dev_pkg="python${python_mm}-dev"
    if ! $SUDO apt-get install -y "$python_dev_pkg"; then
        echo -e "${YELLOW}⚠ ${python_dev_pkg} unavailable, falling back to python3-dev${NC}"
        $SUDO apt-get install -y python3-dev
    fi
    $SUDO apt-get install -y \
        build-essential \
        pkg-config \
        meson \
        ninja-build
    echo -e "${GREEN}✓ Installed Python headers and build toolchain${NC}"
elif command -v dnf &> /dev/null; then
    echo "  Using dnf package manager"
    python_dev_pkg="python${python_mm}-devel"
    if ! $SUDO dnf install -y "$python_dev_pkg"; then
        echo -e "${YELLOW}⚠ ${python_dev_pkg} unavailable, falling back to python3-devel${NC}"
        $SUDO dnf install -y python3-devel
    fi
    $SUDO dnf install -y \
        gcc \
        gcc-c++ \
        make \
        pkgconf-pkg-config \
        meson \
        ninja-build
    echo -e "${GREEN}✓ Installed Python headers and build toolchain${NC}"
elif command -v yum &> /dev/null; then
    echo "  Using yum package manager"
    python_dev_pkg="python${python_mm}-devel"
    if ! $SUDO yum install -y "$python_dev_pkg"; then
        echo -e "${YELLOW}⚠ ${python_dev_pkg} unavailable, falling back to python3-devel${NC}"
        $SUDO yum install -y python3-devel
    fi
    $SUDO yum install -y \
        gcc \
        gcc-c++ \
        make \
        pkgconfig \
        meson \
        ninja-build
    echo -e "${GREEN}✓ Installed Python headers and build toolchain${NC}"
else
    echo -e "${YELLOW}⚠ Could not detect apt/dnf/yum. Install Python headers manually:${NC}"
    echo "    Debian/Ubuntu: sudo apt-get install python${python_mm}-dev build-essential meson ninja-build"
    echo "    Fedora/RHEL:   sudo dnf install python${python_mm}-devel gcc gcc-c++ make meson ninja-build"
fi
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "sam3_env" ]; then
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf sam3_env
        $PYTHON_BIN -m venv sam3_env
        echo -e "${GREEN}✓ Virtual environment recreated${NC}"
    fi
else
    $PYTHON_BIN -m venv sam3_env
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source sam3_env/bin/activate
echo "  Venv Python: $(python --version 2>&1)"
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade packaging toolchain
echo "Upgrading packaging toolchain..."
python -m pip install --upgrade pip wheel
python -m pip install "setuptools==81.0.0"
echo -e "${GREEN}✓ pip/setuptools/wheel upgraded${NC}"
echo ""

# Install PyTorch
echo "Installing PyTorch with CUDA support..."
echo "  Detecting CUDA version..."

if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "  CUDA version: $cuda_version"
    
    if [[ "$cuda_version" == "12.2" ]]; then
        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu122
    elif [[ "$cuda_version" == "12.1" ]]; then
        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$cuda_version" == "11.8" ]]; then
        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo -e "${YELLOW}⚠ Unknown CUDA version, installing default PyTorch${NC}"
        python -m pip install torch torchvision
    fi
else
    echo -e "${YELLOW}⚠ nvcc not found, installing CPU-only PyTorch${NC}"
    python -m pip install torch torchvision
fi
echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# Verify PyTorch CUDA
echo "Verifying PyTorch CUDA..."
python << 'EOF'
import torch
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
EOF
echo ""

# Clone and install SAM3
echo "Installing SAM3..."
if [ -d "sam3" ]; then
    echo -e "${YELLOW}⚠ SAM3 directory already exists${NC}"
    read -p "Skip SAM3 installation? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        rm -rf sam3
        git clone https://github.com/facebookresearch/sam3.git
        cd sam3
        python -m pip install -e ".[notebooks]"
        cd ..
        echo -e "${GREEN}✓ SAM3 installed${NC}"
    fi
else
    git clone https://github.com/facebookresearch/sam3.git
    cd sam3
    python -m pip install -e ".[notebooks]"
    cd ..
    echo -e "${GREEN}✓ SAM3 installed${NC}"
fi
echo ""

# Install other dependencies
echo "Installing additional dependencies..."
python -m pip install nibabel scikit-learn matplotlib tqdm
echo -e "${GREEN}✓ Additional dependencies installed${NC}"
echo ""

# Re-pin setuptools (some installs may upgrade it)
echo "baby"
python -m pip install --force-reinstall "setuptools==81.0.0"

# Verify installation
echo "Verifying installation..."
python << 'EOF'
import sys
packages = {
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'nibabel': 'NiBabel',
    'sklearn': 'scikit-learn',
    'matplotlib': 'Matplotlib',
    'tqdm': 'tqdm',
    'setuptools': 'setuptools',
    'pkg_resources': 'pkg_resources'
}

all_ok = True
for package, name in packages.items():
    try:
        __import__(package)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} - FAILED")
        all_ok = False

# Check SAM3
try:
    sys.path.insert(0, 'sam3')
    from sam3 import build_sam3_image_model
    print(f"  ✓ SAM3")
except ImportError as e:
    print(f"  ✗ SAM3 - FAILED: {e}")
    all_ok = False

if all_ok:
    print("\n✓ All dependencies installed successfully!")
else:
    print("\n✗ Some dependencies failed to install")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Installation verified${NC}"
else
    echo -e "${RED}✗ Installation verification failed${NC}"
    exit 1
fi
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p checkpoints
mkdir -p preprocessed_cache
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Summary
echo "======================================================================"
echo "SETUP COMPLETE!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   ${GREEN}source sam3_env/bin/activate${NC}"
echo ""
echo "2. Update paths in the training script (or use command line args):"
echo "   - DATA_ROOT: Path to your BraTS2020 data"
echo "   - BPE_PATH: Path to sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
echo ""
echo "3. Run training:"
echo "   ${GREEN}python sam3_linear_probing_local.py \\${NC}"
echo "   ${GREEN}    --data_root /path/to/BraTS2020 \\${NC}"
echo "   ${GREEN}    --bpe_path sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \\${NC}"
echo "   ${GREEN}    --epochs 20${NC}"
echo ""
echo "For optimized training (45-50x faster):"
echo "   ${GREEN}python sam3_linear_probing_optimized.py ...${NC}"
echo ""
echo "For more information, see README_LOCAL_SETUP.md"
echo "======================================================================"
