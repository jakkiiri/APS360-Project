#!/bin/bash
#SBATCH --job-name=dual_backbone_cnn_extended
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/dual_backbone_training_%j.out
#SBATCH --error=logs/dual_backbone_training_%j.err

# Print job information
echo "Starting job $SLURM_JOB_ID on $SLURM_JOB_NODELIST"
echo "Job started at: $(date)"
echo "Working directory: $(pwd)"

# Create necessary directories
mkdir -p logs
mkdir -p data
mkdir -p outputs
mkdir -p temp
mkdir -p cache

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Load Python module (adjust according to your HPC system)
# module load python/3.9  # Uncomment and adjust if needed

# Activate your Python virtual environment
source $HOME/skin_cnn_env/bin/activate  # Adjust path to your venv location

# Verify environment activation
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# Navigate to working directory (use fast local storage)
cd $SLURM_TMPDIR || cd /tmp

# Create data directory in temp space for faster I/O
mkdir -p skin_disease_data
cd skin_disease_data

echo "Extracting dataset..."
# Extract your dataset (replace with your actual data archive)
# Option 1: Copy from shared storage
cp $SLURM_SUBMIT_DIR/data/skin_disease_dataset.tar.gz . 2>/dev/null || \
cp /path/to/shared/storage/skin_disease_dataset.tar.gz . 2>/dev/null || \
echo "Warning: Dataset archive not found, assuming data is already available"

# Extract if archive exists
if [ -f "skin_disease_dataset.tar.gz" ]; then
    tar -xzf skin_disease_dataset.tar.gz
    echo "Dataset extracted successfully"
else
    echo "No archive found, looking for existing data structure..."
fi

# Verify data extraction and find data directory
echo "Looking for data directory..."
DATA_DIR=""

# Common possible data directory structures
for possible_dir in "DataSplit" "DataSplit2" "data" "dataset" "skin_disease_data"; do
    if [ -d "$possible_dir" ]; then
        DATA_DIR="$(pwd)/$possible_dir"
        echo "Found data directory: $DATA_DIR"
        break
    fi
done

# If no data found in temp, try using data from submission directory
if [ -z "$DATA_DIR" ] || [ ! -d "$DATA_DIR" ]; then
    echo "No data found in temp storage, checking submission directory..."
    for possible_dir in "$SLURM_SUBMIT_DIR/DataSplit" "$SLURM_SUBMIT_DIR/DataSplit2" "$SLURM_SUBMIT_DIR/../Data Preprocessing/DataSplit" "$SLURM_SUBMIT_DIR/../DataSplit2"; do
        if [ -d "$possible_dir" ]; then
            DATA_DIR="$possible_dir"
            echo "Using data from submission directory: $DATA_DIR"
            break
        fi
    done
fi

# Final check for data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found!"
    echo "Please ensure your dataset is available in one of these locations:"
    echo "  - $SLURM_SUBMIT_DIR/data/skin_disease_dataset.tar.gz"
    echo "  - $SLURM_SUBMIT_DIR/DataSplit/"
    echo "  - $SLURM_SUBMIT_DIR/DataSplit2/"
    echo ""
    echo "Current directory contents:"
    ls -la
    echo ""
    echo "Submission directory contents:"
    ls -la "$SLURM_SUBMIT_DIR/"
    exit 1
fi

echo "Data directory confirmed: $DATA_DIR"
echo "Checking data structure..."
ls -la "$DATA_DIR/"

# Verify train/val splits exist
if [ ! -d "$DATA_DIR/train" ]; then
    echo "Error: Train directory not found in $DATA_DIR"
    echo "Available directories:"
    ls -la "$DATA_DIR/"
    exit 1
fi

if [ ! -d "$DATA_DIR/val" ]; then
    echo "Error: Validation directory not found in $DATA_DIR"
    echo "Available directories:"
    ls -la "$DATA_DIR/"
    exit 1
fi

echo "Train classes:"
ls -la "$DATA_DIR/train/" 2>/dev/null || echo "Could not list train classes"
echo "Val classes:"
ls -la "$DATA_DIR/val/" 2>/dev/null || echo "Could not list val classes"

# Set cache and output directories
CACHE_DIR="$SLURM_TMPDIR/cache"
OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs"
mkdir -p "$CACHE_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Cache directory: $CACHE_DIR"
echo "Output directory: $OUTPUT_DIR"

echo "Starting Dual-Backbone CNN training..."
echo "Data directory: $DATA_DIR"
echo "Cache directory: $CACHE_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run the training script with appropriate parameters
python "$SLURM_SUBMIT_DIR/train_skin_cnn_hpc.py" \
    --data_dir "$DATA_DIR" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --deit_variant "deit3_small_patch16_224" \
    --img_size 224 \
    --batch_size 24 \
    --epochs 200 \
    --learning_rate 3e-4 \
    --lr_backbone 1e-5 \
    --patience 20 \
    --num_workers 8 \
    --warmup_epochs 8 \
    --minority_boost_factor 2.5 \
    --mixup_alpha 0.4 \
    --cutmix_alpha 1.0 \
    --mixup_off_pct 0.2 \
    --mixup_burst_len 12 \
    --finetune_lr_drop 0.2 \
    --topk_checkpoints 5 \
    --seed 42 \
    --use_amp

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Copy cache back to submission directory (optional, for reuse)
    echo "Copying cache to submission directory for future use..."
    cp -r "$CACHE_DIR" "$SLURM_SUBMIT_DIR/cache_backup" 2>/dev/null || echo "Cache copy failed (not critical)"
    
    # Log final model size and results
    if [ -f "$OUTPUT_DIR/multi_best_classifier.pt" ]; then
        echo "Best model saved: $(ls -lh $OUTPUT_DIR/multi_best_classifier.pt)"
    fi
    
    if [ -f "$OUTPUT_DIR/multi_metrics.csv" ]; then
        echo "Metrics logged to: $OUTPUT_DIR/multi_metrics.csv"
        echo "Final metrics:"
        tail -5 "$OUTPUT_DIR/multi_metrics.csv"
    fi
    
    if [ -f "$OUTPUT_DIR/confusion_matrix_best_epoch.png" ]; then
        echo "Confusion matrix saved: $OUTPUT_DIR/confusion_matrix_best_epoch.png"
    fi
    
    if [ -f "$OUTPUT_DIR/training_metrics.png" ]; then
        echo "Training metrics plot saved: $OUTPUT_DIR/training_metrics.png"
    fi
    
    echo "Job completed successfully at: $(date)"
else
    echo "Training failed with exit code $?"
    echo "Check the error log for details: logs/dual_backbone_training_${SLURM_JOB_ID}.err"
    exit 1
fi

# Clean up temporary files (optional)
echo "Cleaning up temporary files..."
cd $SLURM_SUBMIT_DIR
rm -rf $SLURM_TMPDIR/skin_disease_data 2>/dev/null || echo "No temp files to clean"

echo "Job finished at: $(date)"

# Optional: Print summary of outputs
echo ""
echo "=== JOB SUMMARY ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Generated files:"
ls -la "$OUTPUT_DIR/" 2>/dev/null || echo "No output files found"
echo "=================="
