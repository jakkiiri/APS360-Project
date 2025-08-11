#!/bin/bash
# Setup script for creating Python virtual environment for CNN training on HPC

echo "Setting up Python virtual environment for Dual-Backbone CNN training..."

# Set virtual environment name
VENV_NAME="skin_cnn_env"
VENV_PATH="$HOME/$VENV_NAME"

# Check if virtual environment already exists
if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment already exists at $VENV_PATH"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_PATH"
    else
        echo "Using existing virtual environment."
        echo "To activate it, run: source $VENV_PATH/bin/activate"
        exit 0
    fi
fi

# Load Python module if needed (uncomment and adjust for your HPC system)
# module load python/3.9

# Check Python version
echo "Using Python: $(which python3)"
echo "Python version: $(python3 --version)"

# Create virtual environment
echo "Creating virtual environment at $VENV_PATH..."
python3 -m venv "$VENV_PATH"

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (adjust CUDA version as needed for your system)
echo "Installing PyTorch with CUDA support..."
# For CUDA 11.8 (most common on recent HPC systems)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Alternative for CUDA 12.1 (uncomment if needed)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Alternative for CPU only (uncomment if needed)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
fi

echo ""
echo "Checking other packages..."
python -c "import timm; print(f'timm version: {timm.__version__}')"
python -c "import numpy; print(f'numpy version: {numpy.__version__}')"
python -c "import pandas; print(f'pandas version: {pandas.__version__}')"
python -c "import matplotlib; print(f'matplotlib version: {matplotlib.__version__}')"

echo ""
echo "✅ Virtual environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "The virtual environment is located at: $VENV_PATH"
