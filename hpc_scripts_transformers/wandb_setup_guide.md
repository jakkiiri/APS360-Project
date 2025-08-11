# Weights & Biases (W&B) Setup Guide for HPC

This guide explains how to handle W&B authentication on Killarney HPC for the Swin Transformer training.

## 🔑 Authentication Options

### Option 1: Offline Mode (Recommended for HPC)
**Pros:** No internet required during training, secure, works on any HPC
**Cons:** Requires manual sync after training

```bash
# Activate your virtual environment
source swin_skin_env/bin/activate

# In your SLURM script or terminal
export WANDB_MODE=offline

# Train your model (creates local wandb logs)
python train_swin_transformer.py --data_dir /path/to/data

# After training, sync from login node (with internet access)
wandb sync wandb/
```

### Option 2: Environment Variable
**Pros:** Automatic syncing during training
**Cons:** API key visible in process list, requires internet on compute nodes

```bash
# Activate your virtual environment first
source swin_skin_env/bin/activate

# Get your API key from https://wandb.ai/authorize
export WANDB_API_KEY="your_api_key_here"
python train_swin_transformer.py --data_dir /path/to/data
```

### Option 3: Secure File Method
**Pros:** More secure than environment variable
**Cons:** Requires file management

```bash
# Store your API key in a secure file
echo "your_api_key_here" > ~/.wandb_key
chmod 600 ~/.wandb_key

# SLURM script will read from this file
# See train_swin_killarney.sh for implementation
```

### Option 4: Disable W&B Completely
**Pros:** No authentication needed, simple
**Cons:** No experiment tracking

```bash
# Activate your virtual environment
source swin_skin_env/bin/activate

python train_swin_transformer.py --data_dir /path/to/data --disable_wandb
```

## 🚀 Quick Setup Steps

### 1. Get Your W&B API Key
1. Go to https://wandb.ai/authorize
2. Copy your API key

### 2. Choose Your Method

#### For Offline Mode (Recommended):
```bash
# Edit train_swin_killarney.sh
export WANDB_MODE=offline  # Already set in the script

# Run training
sbatch train_swin_killarney.sh

# After training, sync from login node
wandb sync wandb/
```

#### For Direct Authentication:
```bash
# Option A: Edit the SLURM script
export WANDB_API_KEY="your_key_here"  # Uncomment in script

# Option B: Use secure file
echo "your_key_here" > ~/.wandb_key
chmod 600 ~/.wandb_key
# Uncomment the secure file section in train_swin_killarney.sh
```

## 📊 Understanding W&B Modes

### Online Mode (Default)
- Syncs in real-time during training
- Requires internet connection on compute nodes
- Shows live plots and metrics

### Offline Mode
- Stores data locally in `wandb/` directory
- No internet required during training
- Sync manually after training

### Disabled Mode
- No W&B logging at all
- Use `--disable_wandb` flag
- Only local matplotlib plots are generated

## 🔄 Syncing Offline Runs

If you used offline mode, sync your runs after training:

```bash
# From login node (with internet)
cd /path/to/your/project

# Sync all offline runs
wandb sync wandb/

# Sync specific run
wandb sync wandb/offline-run-20231201_123456-abc123

# Check sync status
wandb status
```

## 🛠 Troubleshooting

### "wandb: ERROR Unable to create run"
**Solution:** Check authentication or use offline mode
```bash
export WANDB_MODE=offline
```

### "wandb: ERROR API key not found"
**Solutions:**
1. Set `WANDB_API_KEY` environment variable
2. Run `wandb login` on login node
3. Use offline mode
4. Use `--disable_wandb` flag

### "wandb: ERROR Network connection failed"
**Solution:** Use offline mode (recommended for HPC)
```bash
export WANDB_MODE=offline
```

### Sync fails after offline training
**Solutions:**
1. Ensure you're on login node with internet
2. Check wandb directory exists: `ls wandb/`
3. Try syncing specific runs: `wandb sync wandb/offline-run-*`

## 🔒 Security Best Practices

1. **Never hardcode API keys in scripts**
2. **Use offline mode when possible**
3. **Store keys in secure files with restricted permissions**
4. **Use environment variables only in secure environments**
5. **Consider using `--disable_wandb` for sensitive data**

## 📈 What Gets Logged

When W&B is enabled, you'll see:
- Real-time training and validation metrics
- Learning rate schedules
- System metrics (GPU utilization, memory usage)
- Model configuration and hyperparameters
- Confusion matrices and other plots

Access your dashboard at: https://wandb.ai/

## 🎯 Recommended Workflow for HPC

1. **Development:** Use online mode with your API key
2. **Production/HPC:** Use offline mode for training
3. **Analysis:** Sync offline runs and analyze online

This gives you the best of both worlds: secure HPC training and rich online analysis.