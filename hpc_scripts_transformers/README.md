# Enhanced Swin Transformer for Severe Class Imbalance - HPC Training

This directory contains scripts for training an **enhanced Swin Transformer** specifically designed to handle severe class imbalance in skin disease classification, optimized for HPC resources.

## Files Overview

### 1. `train_swin_transformer.py`
**🏆 Enhanced training script** with state-of-the-art class imbalance handling:
- **Advanced Loss Functions**: Class-Balanced Focal Loss, LDAM Loss, Enhanced Weighted Cross-Entropy
- **Medical Importance Weighting**: Critical classes (melanoma: 3x, SCC: 2.5x) get priority
- **Aggressive Minority Oversampling**: 3x minority class boosting during training
- **Dermatology-Specific Augmentations**: CLAHE, skin tone variation, lighting simulation
- **Macro F1 Optimization**: Primary metric instead of accuracy for balanced performance
- **Multi-Phase Learning Rate**: Warmup + Cosine Annealing + Plateau reduction
- **Extended Training**: 200 epochs with 20 patience for thorough learning
- **Automatic Mixed Precision**: Faster training with AMP
- **Comprehensive Visualization**: All plots saved to disk (no external dependencies)
- **HPC-Optimized**: Headless operation with robust checkpointing

### 2. `train_swin_killarney.sh`
**🚀 Enhanced SLURM batch script** for extended training:
- Configured for GPU training (H100/V100/A100)
- Extended resource allocation (64GB RAM, 8 CPUs)
- 48-hour time limit for 200-epoch training
- Automated data extraction and setup
- Enhanced logging and result tracking
- W&B removed - all logging via matplotlib

### 3. `requirements.txt`
Python dependencies including:
- PyTorch ecosystem (torch, torchvision, timm)
- Albumentations for dermatology-specific augmentations
- Matplotlib/Seaborn for comprehensive visualization
- Standard ML libraries (sklearn, numpy, pandas, etc.)
- ~~Weights & Biases~~ **(Removed - using matplotlib instead)**

### 4. `config.py`
Centralized configuration management:
- Model configurations for different Swin variants
- Training hyperparameters
- Augmentation settings
- HPC resource configurations
- Class definitions and mappings

### 5. `setup_venv.sh`
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

# No additional setup needed - all logging via matplotlib
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

### 🎯 **Advanced Class Imbalance Handling**
1. **Enhanced Weighted Sampling**: Medical importance + inverse frequency weighting
2. **Advanced Loss Functions**: Class-Balanced Focal Loss (default), LDAM Loss, Enhanced Weighted CE
3. **Aggressive Minority Oversampling**: 3x oversampling with targeted augmentation
4. **Dermatology-Specific Augmentations**: CLAHE, skin tone variation, medical imaging artifacts
5. **Macro F1 Optimization**: Primary metric for balanced minority class performance

### Minority Classes (Extra Augmentation)
- melanoma (malignant)
- actinic_keratosis (pre-cancerous)
- scc (squamous cell carcinoma)
- dermatofibroma (benign but rare)
- lentigo (benign but rare)
- vascular_lesion (rare)

## 📊 **Enhanced Monitoring & Visualization**

All training metrics and visualizations are automatically saved to disk:
- **Real-time CSV logging**: Complete metrics per epoch
- **Training curves**: Loss, accuracy, macro F1, per-class metrics
- **Confusion matrices**: Best F1 and best accuracy models
- **Per-class performance**: Detailed precision/recall/F1 analysis
- **Learning rate tracking**: Multi-phase scheduler visualization

**No external dependencies** - all plots saved to `outputs/plots/`

## 📁 **Enhanced Output Files**

The training produces comprehensive results:

### 🏆 **Model Checkpoints**
- `best_f1_model.pth`: Best model by macro F1 (primary)
- `best_acc_model.pth`: Best model by accuracy (secondary)
- `checkpoint.pth`: Latest checkpoint for resuming training

### 📊 **Visualization & Analysis**
- `plots/confusion_matrix_best_f1.png`: Confusion matrix for best F1 model
- `plots/per_class_metrics_best_f1.png`: Detailed per-class performance
- `plots/training_metrics.png`: Complete training curves
- `plots/final_confusion_matrix.png`: Final validation confusion matrix

### 📈 **Data & Logs**
- `training_metrics.csv`: Epoch-by-epoch metrics data
- `final_summary.json`: Complete results summary
- `best_f1_predictions.pkl`: Predictions for further analysis
- `logs/`: SLURM job output and error logs

## Customization

### 🔧 **Advanced Hyperparameter Configuration**
```bash
python train_swin_transformer.py \
    --data_dir /path/to/data \
    --model_name swin_base_patch4_window7_224 \
    --batch_size 16 \
    --num_epochs 200 \
    --patience 20 \
    --learning_rate 1e-4 \
    --minority_boost_factor 3.0 \
    --dermatology_aug_prob 0.8 \
    --loss_function cb_focal \
    --focal_gamma 2.0 \
    --use_amp
```

### 🎯 **Loss Function Options**
- `cb_focal`: Class-Balanced Focal Loss (recommended for severe imbalance)
- `focal`: Standard Focal Loss
- `ldam`: Label-Distribution-Aware Margin Loss
- `weighted_ce`: Enhanced Weighted Cross-Entropy

### 💻 **Updated Resource Requirements**
- **Minimum**: 32GB RAM, 1 GPU, 4 CPUs (for 200-epoch training)
- **Recommended**: 64GB RAM, 1 H100/A100 GPU, 8 CPUs (current configuration)
- **Extended training**: 96GB RAM, 1 A100 GPU, 12 CPUs (for larger models/batches)

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce `batch_size` or `image_size`
2. **Data not found**: Check data extraction and paths in SLURM script
3. **Environment issues**: 
   - Use `setup_venv.sh` for automated setup
   - Ensure virtual environment is activated
   - Check if modules need to be loaded on your HPC system
4. **Graphics/Display errors**: Script is configured for headless operation with matplotlib backend 'Agg'
5. **Missing plots**: Check `outputs/plots/` directory - all visualizations saved automatically
6. **Python/PyTorch not found**: Load appropriate modules before creating venv

### Performance Tips
1. Use `$SLURM_TMPDIR` for faster I/O (already configured)
2. Increase `num_workers` if CPU cores are available
3. Use mixed precision training for larger batch sizes
4. Consider gradient accumulation for effective larger batches

## 🎯 **Expected Enhanced Results**

With the advanced class imbalance handling, you should expect:
- **Macro F1-Score**: 70-85% (primary optimization target)
- **Validation Accuracy**: 75-90% (depending on data quality) 
- **Minority Class Performance**: Significant improvement in melanoma, SCC detection
- **Training Time**: 24-48 hours for 200 epochs (extended for thorough learning)
- **Best Performance**: Usually achieved around epoch 80-120 with new scheduling
- **Class Balance**: Comprehensive per-class metrics saved automatically to CSV/plots