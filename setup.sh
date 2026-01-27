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

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "${RED}Error: Python 3.8+ required${NC}"
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

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "sam3_env" ]; then
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf sam3_env
        python3 -m venv sam3_env
        echo -e "${GREEN}✓ Virtual environment recreated${NC}"
    fi
else
    python3 -m venv sam3_env
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source sam3_env/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Install PyTorch
echo "Installing PyTorch with CUDA support..."
echo "  Detecting CUDA version..."

if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "  CUDA version: $cuda_version"
    
    if [[ "$cuda_version" == "12.2" ]]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu122
    elif [[ "$cuda_version" == "12.1" ]]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$cuda_version" == "11.8" ]]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo -e "${YELLOW}⚠ Unknown CUDA version, installing default PyTorch${NC}"
        pip install torch torchvision
    fi
else
    echo -e "${YELLOW}⚠ nvcc not found, installing CPU-only PyTorch${NC}"
    pip install torch torchvision
fi
echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# Verify PyTorch CUDA
echo "Verifying PyTorch CUDA..."
python3 << 'EOF'
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
        pip install -e ".[notebooks]"
        cd ..
        echo -e "${GREEN}✓ SAM3 installed${NC}"
    fi
else
    git clone https://github.com/facebookresearch/sam3.git
    cd sam3
    pip install -e ".[notebooks]"
    cd ..
    echo -e "${GREEN}✓ SAM3 installed${NC}"
fi
echo ""

# Install other dependencies
echo "Installing additional dependencies..."
pip install nibabel scikit-learn matplotlib tqdm
echo -e "${GREEN}✓ Additional dependencies installed${NC}"
echo ""

# Verify installation
echo "Verifying installation..."
python3 << 'EOF'
import sys
packages = {
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'nibabel': 'NiBabel',
    'sklearn': 'scikit-learn',
    'matplotlib': 'Matplotlib',
    'tqdm': 'tqdm'
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