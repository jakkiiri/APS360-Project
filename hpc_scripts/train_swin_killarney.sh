#!/bin/bash
#SBATCH --job-name=swin_skin_disease
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

# Set W&B API key (REPLACE WITH YOUR ACTUAL KEY OR USE ALTERNATIVE METHODS BELOW)
# Option 1: Set API key directly (NOT RECOMMENDED for security)
# export WANDB_API_KEY="your_wandb_api_key_here"

# Option 2: Use offline mode and sync later (RECOMMENDED for HPC)
export WANDB_MODE=offline

# Option 3: Read from a secure file (create .wandb_key file with your key)
# if [ -f "$HOME/.wandb_key" ]; then
#     export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
# fi

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
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --patience 15 \
    --num_workers 8 \
    --seed 42

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Copy important results back to submission directory
    echo "Copying results back to submission directory..."
    cp -r outputs/* "$OUTPUT_DIR/" 2>/dev/null || echo "No additional outputs to copy"
    
    # Log final model size
    if [ -f "$OUTPUT_DIR/best_model.pth" ]; then
        echo "Best model saved: $(ls -lh $OUTPUT_DIR/best_model.pth)"
    fi
    
    # Sync W&B runs if using offline mode
    if [ "$WANDB_MODE" = "offline" ]; then
        echo "Syncing W&B offline runs..."
        wandb sync "$SLURM_SUBMIT_DIR"/wandb/ || echo "W&B sync failed or no runs to sync"
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

echo "Job finished at: $(date)"