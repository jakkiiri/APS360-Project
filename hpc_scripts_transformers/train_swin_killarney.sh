#!/bin/bash
#SBATCH --job-name=swin_enhanced_skin_disease
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/swin_training_%j.out
#SBATCH --error=logs/swin_training_%j.err

# Print job information
echo "Starting job $SLURM_JOB_ID on $SLURM_JOB_NODELIST"
echo "Job started at: $(date)"
echo "Working directory: $(pwd)"

# Create necessary directories
mkdir -p logs
mkdir -p data
mkdir -p outputs
mkdir -p temp

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# W&B removed - using matplotlib for logging and visualization
# All plots and metrics are saved directly to disk

# Load Python module (if required on your HPC system)
# module load python/3.9  # Uncomment and adjust if needed

# Activate your Python virtual environment (assuming it's already set up)
source $HOME/swin_skin_env/bin/activate  # Adjust path to your venv location

# Verify environment activation
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# Navigate to working directory
cd $SLURM_TMPDIR || cd /tmp

# Create data directory in temp space for faster I/O
mkdir -p skin_disease_data
cd skin_disease_data

echo "Extracting dataset..."
# Extract your dataset (replace with your actual data archive)
# Assuming the data is in a tar.gz file on the cluster
cp $SLURM_SUBMIT_DIR/data/skin_disease_dataset.tar.gz .
tar -xzf skin_disease_dataset.tar.gz

# Verify data extraction
echo "Dataset extracted. Directory structure:"
ls -la

# If your data has a different structure, adjust the path accordingly
DATA_DIR="$(pwd)/DataSplit"  # Adjust this path based on your actual data structure

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR not found!"
    echo "Available directories:"
    find . -maxdepth 2 -type d
    exit 1
fi

echo "Data directory found: $DATA_DIR"
echo "Checking class directories..."
ls -la "$DATA_DIR/train/" 2>/dev/null || echo "Train directory not found"
ls -la "$DATA_DIR/val/" 2>/dev/null || echo "Val directory not found"
ls -la "$DATA_DIR/test/" 2>/dev/null || echo "Test directory not found"

# Set output directory back to submission directory
OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs"
mkdir -p "$OUTPUT_DIR"

echo "Starting Swin Transformer training..."
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run the training script with appropriate parameters
python "$SLURM_SUBMIT_DIR/train_swin_transformer.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "swin_base_patch4_window7_224" \
    --image_size 512 \
    --batch_size 16 \
    --num_epochs 200 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --patience 20 \
    --num_workers 8 \
    --minority_boost_factor 3.0 \
    --dermatology_aug_prob 0.8 \
    --loss_function "cb_focal" \
    --focal_gamma 2.0 \
    --cb_beta 0.9999 \
    --use_amp \
    --seed 42

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Copy important results back to submission directory
    echo "Copying results back to submission directory..."
    cp -r outputs/* "$OUTPUT_DIR/" 2>/dev/null || echo "No additional outputs to copy"
    
    # Log final model sizes and results
    if [ -f "$OUTPUT_DIR/best_f1_model.pth" ]; then
        echo "Best F1 model saved: $(ls -lh $OUTPUT_DIR/best_f1_model.pth)"
    fi
    if [ -f "$OUTPUT_DIR/best_acc_model.pth" ]; then
        echo "Best accuracy model saved: $(ls -lh $OUTPUT_DIR/best_acc_model.pth)"
    fi
    if [ -f "$OUTPUT_DIR/training_metrics.csv" ]; then
        echo "Training metrics saved: $(ls -lh $OUTPUT_DIR/training_metrics.csv)"
    fi
    if [ -d "$OUTPUT_DIR/plots" ]; then
        echo "Generated plots: $(ls -1 $OUTPUT_DIR/plots/ | wc -l) files in plots/"
    fi
    
    echo "Job completed successfully at: $(date)"
else
    echo "Training failed with exit code $?"
    exit 1
fi

# Clean up temporary files (optional)
echo "Cleaning up temporary files..."
cd $SLURM_SUBMIT_DIR
rm -rf $SLURM_TMPDIR/skin_disease_data 2>/dev/null || echo "No temp files to clean"

# Display final summary
echo "="*60
echo "TRAINING SUMMARY:"
echo "- Enhanced Swin Transformer with Class Imbalance Handling"
echo "- 200 epochs with 20 patience, Macro F1 optimization"
echo "- Class-Balanced Focal Loss with minority oversampling"
echo "- Dermatology-specific augmentations"
echo "- All visualizations saved to: $OUTPUT_DIR/plots/"
echo "="*60

echo "Job finished at: $(date)"