#!/bin/bash
# Setup script for creating Python virtual environment on HPC
# Run this on the login node before submitting jobs

echo "Setting up Python virtual environment for Swin Transformer training..."

# Load Python module if required (uncomment and adjust for your HPC system)
# module load python/3.9
# module load cuda/11.8  # If you need specific CUDA version

# Check Python availability
echo "Checking Python installation..."
python3 --version || { echo "Python3 not found. Please load the appropriate module."; exit 1; }

# Create virtual environment
ENV_NAME="swin_skin_env"
ENV_PATH="$HOME/$ENV_NAME"

if [ -d "$ENV_PATH" ]; then
    echo "Virtual environment $ENV_NAME already exists at $ENV_PATH"
    read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        rm -rf "$ENV_PATH"
    else
        echo "Keeping existing environment. Exiting..."
        exit 0
    fi
fi

echo "Creating virtual environment at $ENV_PATH..."
python3 -m venv "$ENV_PATH"

# Activate environment
echo "Activating virtual environment..."
source "$ENV_PATH/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (you may need to adjust for your CUDA version)
echo "Installing PyTorch..."
# For CUDA 11.8 (adjust based on your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only (uncomment if no GPU)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo "Installing other dependencies..."
pip install -r requirements.txt

# Test installation
echo "Testing installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import timm; print(f'timm version: {timm.__version__}')"
# python -c "import wandb; print(f'wandb version: {wandb.__version__}')"  # W&B removed
python -c "import matplotlib; print(f'matplotlib version: {matplotlib.__version__}')"

echo ""
echo "✅ Virtual environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "source $ENV_PATH/bin/activate"
echo ""
echo "To update the SLURM script, make sure the path matches:"
echo "source $ENV_PATH/bin/activate"
echo ""
echo "You can now submit your job with: sbatch train_swin_killarney.sh"