# HPC Scripts for Dual-Backbone CNN Training

This directory contains scripts for training a dual-backbone CNN model (ConvNeXt-Tiny + DeiT-III) for skin disease classification on HPC systems like Killarney.

## Files Overview

- `train_skin_cnn_hpc.py` - Main training script adapted for HPC usage
- `train_skin_cnn_killarney.sh` - SLURM batch script for job submission
- `requirements.txt` - Python dependencies
- `setup_venv.sh` - Virtual environment setup script
- `README.md` - This file

## Model Architecture

The model uses a dual-backbone approach:
- **CNN Branch**: ConvNeXt-Tiny with feature extraction and Global Average Pooling
- **ViT Branch**: DeiT-III Small/Tiny with CLS token extraction
- **Classifier**: Multi-layer perceptron that processes concatenated features

## Setup Instructions

### 1. Prepare Your Environment

First, set up the Python virtual environment:

```bash
# Make the setup script executable
chmod +x setup_venv.sh

# Run the setup script
./setup_venv.sh
```

This will:
- Create a virtual environment at `$HOME/skin_cnn_env`
- Install PyTorch with CUDA support
- Install all required dependencies

### 2. Prepare Your Data

Ensure your dataset follows this structure:
```
DataSplit/
├── train/
│   ├── nevus/
│   ├── melanoma/
│   ├── bcc/
│   └── ... (other classes)
└── val/
    ├── nevus/
    ├── melanoma/
    ├── bcc/
    └── ... (other classes)
```

### 3. Submit the Job

Make the batch script executable and submit:

```bash
# Make the script executable
chmod +x train_skin_cnn_killarney.sh

# Submit the job
sbatch train_skin_cnn_killarney.sh
```

## Configuration Options

The training script supports many command-line arguments. Key parameters include:

### Data Parameters
- `--data_dir`: Path to original dataset directory
- `--cache_dir`: Path to cache directory (for faster loading)
- `--output_dir`: Output directory for models and plots

### Model Parameters
- `--deit_variant`: DeiT model variant (default: "deit3_small_patch16_224")
- `--img_size`: Input image size for DeiT (default: 224)

### Training Parameters
- `--batch_size`: Batch size (default: 24)
- `--epochs`: Number of epochs (default: 100)
- `--learning_rate`: Initial learning rate (default: 3e-4)
- `--lr_backbone`: Learning rate for backbone when unfrozen (default: 1e-5)
- `--warmup_epochs`: Number of warmup epochs (default: 5)

### Augmentation Parameters
- `--mixup_alpha`: Mixup alpha parameter (default: 0.4)
- `--cutmix_alpha`: CutMix alpha parameter (default: 1.0)
- `--mixup_burst_len`: Epochs to keep mixup on after resume (default: 8)

## Features

### Advanced Training Techniques
- **Weighted Random Sampling**: Handles class imbalance
- **Mixed Precision Training**: Faster training with AMP
- **Progressive Unfreezing**: Backbones unfrozen after warmup
- **Mixup/CutMix**: Data augmentation with adaptive scheduling
- **Cosine Annealing**: Learning rate scheduling with warmup

### Monitoring and Checkpointing
- **Top-K Checkpoints**: Keeps best models by F1 score
- **Early Stopping**: Based on moving average of macro-F1
- **Comprehensive Metrics**: Per-class and macro metrics
- **CSV Logging**: Detailed metrics for each epoch

### HPC Optimizations
- **Headless Operation**: No display required
- **Efficient Caching**: Preprocessed tensors for faster I/O
- **Memory Management**: Proper cleanup and memory usage
- **Error Handling**: Robust error handling and logging

## Output Files

The training will generate several output files:

### Model Files
- `multi_best_classifier.pt` - Best model checkpoint (by macro-F1)
- `checkpoints/multi_ep*_f1*_rec*.pt` - Top-K checkpoints

### Metrics and Logs
- `multi_metrics.csv` - Training metrics for each epoch
- `multi_val_predictions.pkl` - Best epoch predictions for analysis

### Visualizations
- `confusion_matrix_best_epoch.png` - Confusion matrix from best epoch
- `training_metrics.png` - Training curves plot

## Monitoring Your Job

### Check Job Status
```bash
squeue -u $USER
```

### View Output Logs
```bash
# Real-time output
tail -f logs/dual_backbone_training_<JOB_ID>.out

# Real-time errors
tail -f logs/dual_backbone_training_<JOB_ID>.err
```

### Check Results
```bash
# List output files
ls -la outputs/

# View final metrics
tail -10 outputs/multi_metrics.csv
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `--batch_size` (try 16 or 12)
   - Reduce image size or use smaller model variant

2. **Data Not Found**
   - Check data directory structure
   - Verify paths in the batch script
   - Ensure data extraction completed successfully

3. **Environment Issues**
   - Recreate virtual environment: `./setup_venv.sh`
   - Check module loading in batch script
   - Verify CUDA compatibility

4. **Slow Training**
   - Increase `--num_workers` if CPU allows
   - Use faster storage for cache directory
   - Enable mixed precision with `--use_amp`

### Performance Tips

1. **Use Local Storage**: The script copies data to `$SLURM_TMPDIR` for faster I/O
2. **Caching**: Preprocessed tensors are cached for subsequent runs
3. **Mixed Precision**: Use `--use_amp` for faster training
4. **Batch Size**: Optimize based on GPU memory (24 works well for most systems)

## Customization

### Modifying the Model
Edit `train_skin_cnn_hpc.py` to:
- Change backbone architectures
- Adjust classifier head
- Modify feature fusion strategy

### Adjusting Training
Edit training parameters in the batch script or use command-line args:
- Learning rates and schedules
- Augmentation strategies
- Early stopping criteria

### HPC System Specific
Edit `train_skin_cnn_killarney.sh` to:
- Adjust SLURM parameters for your system
- Change module loading commands
- Modify resource requests

## Class Information

The model supports 9 skin disease classes:
1. `nevus` - Benign moles (most common)
2. `melanoma` - Malignant melanoma (critical minority class)
3. `bcc` - Basal cell carcinoma
4. `keratosis` - Seborrheic keratosis
5. `actinic_keratosis` - Actinic keratosis (pre-cancerous)
6. `scc` - Squamous cell carcinoma
7. `dermatofibroma` - Benign fibrous lesions
8. `lentigo` - Solar lentigo
9. `vascular_lesion` - Vascular lesions

The model uses weighted sampling and augmentation to handle class imbalance, with special attention to minority classes (especially melanoma).
