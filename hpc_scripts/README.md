# Swin Transformer Training on HPC (Killarney)

This directory contains scripts for training a Swin Transformer on skin disease classification using HPC resources.

## Files Overview

### 1. `train_swin_transformer.py`
Main training script featuring:
- **Pre-trained Swin Transformer** with fine-tuning for 9-class skin disease classification
- **Strong augmentations** using Albumentations (rotation, crops, color jitter, elastic transforms, etc.)
- **Class imbalance handling** with weighted sampling and additional augmentations for minority classes
- **Weights & Biases integration** for experiment tracking
- **Early stopping** and model checkpointing
- **Comprehensive metrics** (accuracy, precision, recall, F1, confusion matrix)
- **HPC-optimized** with headless matplotlib backend (no graphics display)

### 2. `train_swin_killarney.sh`
SLURM batch script for Killarney HPC:
- Configured for GPU training (V100)
- Data extraction and setup
- Environment activation
- Resource allocation (32GB RAM, 8 CPUs)
- 12-hour time limit

### 3. `requirements.txt`
Python dependencies including:
- PyTorch ecosystem (torch, torchvision, timm)
- Albumentations for advanced augmentations
- Weights & Biases for monitoring
- Standard ML libraries (sklearn, numpy, etc.)

### 4. `config.py`
Centralized configuration management:
- Model configurations for different Swin variants
- Training hyperparameters
- Augmentation settings
- HPC resource configurations
- Class definitions and mappings

### 5. `wandb_setup_guide.md`
Comprehensive guide for Weights & Biases authentication:
- Multiple authentication methods (offline mode, API key, secure file)
- HPC-specific recommendations
- Troubleshooting and security best practices
- Step-by-step setup instructions

### 6. `setup_venv.sh`
Automated virtual environment setup script:
- Creates and configures Python virtual environment
- Installs PyTorch with appropriate CUDA support
- Tests all dependencies
- Provides activation instructions

## Quick Start

### 1. Setup Environment

#### Option A: Automated Setup (Recommended)
```bash
# Run the setup script (on login node)
chmod +x setup_venv.sh
./setup_venv.sh
```

#### Option B: Manual Setup
```bash
# Create Python virtual environment
python3 -m venv swin_skin_env
source swin_skin_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Setup Weights & Biases (see wandb_setup_guide.md for details)
# Option A: Get API key and use offline mode (recommended for HPC)
export WANDB_MODE=offline

# Option B: Login directly (for development)
wandb login
```

### 2. Prepare Data
Ensure your dataset follows this structure:
```
DataSplit/
├── train/
│   ├── nevus/
│   ├── melanoma/
│   ├── bcc/
│   ├── seborrheic_keratosis/
│   ├── actinic_keratosis/
│   ├── scc/
│   ├── dermatofibroma/
│   ├── lentigo/
│   └── vascular_lesion/
├── val/
│   └── [same structure]
└── test/
    └── [same structure]
```

### 3. Update SLURM Script
Edit `train_swin_killarney.sh`:
- Update the virtual environment path (should match where you created it)
- Update data archive path and extraction commands
- Uncomment and adjust module loading if required by your HPC system
- Adjust resource requirements if needed

### 4. Submit Job
```bash
# Make script executable
chmod +x train_swin_killarney.sh

# Submit to SLURM
sbatch train_swin_killarney.sh
```

## Model Configuration

### Available Swin Transformer Variants
- **swin_tiny**: Fastest, smallest model
- **swin_small**: Balanced speed/performance
- **swin_base**: Default, good performance (recommended)
- **swin_large**: Best performance, requires more resources

### Key Features for Class Imbalance
1. **Weighted Random Sampling**: Balances class representation in each batch
2. **Class-weighted Loss**: CrossEntropyLoss with inverse class frequencies
3. **Minority Class Augmentation**: Additional transforms for underrepresented classes
4. **Strong Augmentations**: Comprehensive Albumentations pipeline

### Minority Classes (Extra Augmentation)
- melanoma (malignant)
- actinic_keratosis (pre-cancerous)
- scc (squamous cell carcinoma)
- dermatofibroma (benign but rare)
- lentigo (benign but rare)
- vascular_lesion (rare)

## Monitoring

Training metrics are logged to Weights & Biases:
- Real-time loss and accuracy curves
- Confusion matrices
- Learning rate schedules
- System metrics

Access your dashboard at: https://wandb.ai/

## Outputs

The training produces:
- `best_model.pth`: Best model checkpoint
- `confusion_matrix.png`: Final confusion matrix
- Training logs in `logs/` directory
- W&B experiment tracking online

## Customization

### Adjust Hyperparameters
Edit the SLURM script or modify `config.py`:
```bash
python train_swin_transformer.py \
    --data_dir /path/to/data \
    --model_name swin_small_patch4_window7_224 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --num_epochs 100
```

### Resource Requirements
- **Minimum**: 16GB RAM, 1 GPU, 4 CPUs
- **Recommended**: 32GB RAM, 1 V100/A100 GPU, 8 CPUs
- **Large models**: 64GB RAM, 1 A100 GPU, 12 CPUs

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce `batch_size` or `image_size`
2. **Data not found**: Check data extraction and paths in SLURM script
3. **Environment issues**: 
   - Use `setup_venv.sh` for automated setup
   - Ensure virtual environment is activated
   - Check if modules need to be loaded on your HPC system
4. **W&B authentication**: Run `wandb login` before training or use offline mode
5. **Graphics/Display errors**: Script is configured for headless operation with matplotlib backend 'Agg'
6. **Python/PyTorch not found**: Load appropriate modules before creating venv

### Performance Tips
1. Use `$SLURM_TMPDIR` for faster I/O (already configured)
2. Increase `num_workers` if CPU cores are available
3. Use mixed precision training for larger batch sizes
4. Consider gradient accumulation for effective larger batches

## Expected Results

With proper training, you should expect:
- **Validation Accuracy**: 75-85% (depending on data quality)
- **Training Time**: 6-12 hours for 50 epochs
- **Best Performance**: Usually achieved around epoch 30-40
- **Class Balance**: Monitor per-class metrics in W&B