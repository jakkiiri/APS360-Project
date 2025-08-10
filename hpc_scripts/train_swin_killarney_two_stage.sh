#!/bin/bash
#SBATCH --job-name=swin_two_stage
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/swin_two_stage_%j.out
#SBATCH --error=logs/swin_two_stage_%j.err

echo "Starting job $SLURM_JOB_ID on $SLURM_JOB_NODELIST"
echo "Job started at: $(date)"
echo "Working directory: $(pwd)"

mkdir -p logs data outputs_two_stage temp

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_LAUNCH_BLOCKING=1

# Ensure no W&B logging (script does not use it, but keep env clean)
unset WANDB_API_KEY
export WANDB_MODE=disabled

# Activate venv
source $HOME/swin_skin_env/bin/activate
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "nvidia-smi:"
nvidia-smi || echo "nvidia-smi not available"

# Use node-local scratch for faster IO
cd $SLURM_TMPDIR || cd /tmp
mkdir -p skin_disease_data
cd skin_disease_data

echo "Extracting dataset..."
# Replace with your actual dataset archive
cp $SLURM_SUBMIT_DIR/data/skin_disease_dataset.tar.gz .
tar -xzf skin_disease_dataset.tar.gz

echo "Dataset extracted. Directory structure:"
ls -la

DATA_DIR="$(pwd)/DataSplit"
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR not found!"
    find . -maxdepth 2 -type d
    exit 1
fi

echo "Data directory found: $DATA_DIR"
ls -la "$DATA_DIR/train/" 2>/dev/null || echo "Train directory not found"
ls -la "$DATA_DIR/val/" 2>/dev/null || echo "Val directory not found"
ls -la "$DATA_DIR/test/" 2>/dev/null || echo "Test directory not found"

OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs_two_stage"
mkdir -p "$OUTPUT_DIR"

echo "Starting Two-Stage Swin training..."
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"

python "$SLURM_SUBMIT_DIR/train_swin_transformer_two_stage.py" \
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

if [ $? -eq 0 ]; then
  echo "Two-stage training completed successfully!"
  echo "Copying results back to submission directory..."
  cp -r outputs_two_stage/* "$OUTPUT_DIR/" 2>/dev/null || echo "No additional outputs to copy"
  echo "Job completed successfully at: $(date)"
else
  echo "Training failed with exit code $?"
  exit 1
fi

echo "Cleaning up temporary files..."
cd $SLURM_SUBMIT_DIR
rm -rf $SLURM_TMPDIR/skin_disease_data 2>/dev/null || echo "No temp files to clean"
echo "Job finished at: $(date)"


