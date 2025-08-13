# Swin Transformer Inference Scripts & GUI

This directory contains scripts and a GUI application for running inference on your trained Swin Transformer model for skin disease classification.

## Files

### Command Line Scripts
- `swin_inference.py` - Main inference script with comprehensive evaluation metrics
- `run_inference.py` - Easy-to-use wrapper script with auto-detection

### GUI Application  
- `single_image_gui.py` - Interactive GUI for single image classification
- `run_gui.py` - GUI launcher with dependency checking
- `gui_demo.py` - Demo and test suite for the GUI

### Threshold Tuning & Optimization
- `threshold_tuning.py` - Coordinate ascent threshold optimization module
- `threshold_workflow.py` - Complete workflow for tuning thresholds
- `model_utils.py` - Model loading utilities with PyTorch compatibility

### Analysis & Utilities
- `analyze_results.py` - Advanced results analysis
- `test_inference_setup.py` - Environment verification
- `requirements.txt` - Package dependencies
- `README.md` - This file

## Quick Start

### GUI Application (Recommended for Single Images)

Launch the interactive GUI for easy single-image classification:

```bash
# Simple launch
python run_gui.py

# Or directly
python single_image_gui.py
```

**GUI Features:**
- 📂 **Load Model**: Browse and load your trained .pth model weights
- 🖼️ **Load Image**: Select any skin lesion image (JPG, PNG, etc.)
- 🔍 **Run Inference**: One-click classification with confidence scores
- 📊 **Results Visualization**: Class probabilities with color-coded risk levels
- 🎯 **Medical Context**: Risk levels and descriptions for each disease class
- 🎯 **Threshold Tuning**: Load optimized thresholds for improved F1 performance

### Command Line Scripts

#### Option 1: Easy Runner (For Test Sets)

The `run_inference.py` script automatically detects your trained model and data:

```bash
# Auto-detect everything
python run_inference.py

# Specify custom paths
python run_inference.py --checkpoint /path/to/best_f1_model.pth --data_dir /path/to/DataSplit

# Use different batch size
python run_inference.py --batch_size 16

# Enable threshold tuning
python run_inference.py --tune_thresholds

# Check GPU availability
python run_inference.py --gpu
```

#### Option 2: Direct Script

For more control, use the main inference script directly:

```bash
python swin_inference.py \
    --checkpoint_path /path/to/model.pth \
    --data_dir /path/to/DataSplit \
    --output_dir ./test_results \
    --batch_size 32
```

## Requirements

The inference scripts use the same dependencies as the training script:

- PyTorch
- timm
- albumentations
- scikit-learn
- matplotlib
- seaborn
- pandas
- tqdm
- PIL

## Data Structure

Your data directory should have this structure:

```
DataSplit/
├── train/
│   ├── nevus/
│   ├── melanoma/
│   ├── bcc/
│   ├── keratosis/
│   ├── actinic_keratosis/
│   ├── scc/
│   ├── dermatofibroma/
│   ├── lentigo/
│   └── vascular_lesion/
├── val/
│   └── ... (same structure)
└── test/          # <- This is what inference uses
    ├── nevus/
    ├── melanoma/
    ├── bcc/
    ├── keratosis/
    ├── actinic_keratosis/
    ├── scc/
    ├── dermatofibroma/
    ├── lentigo/
    └── vascular_lesion/
```

## Model Checkpoints

The scripts look for these checkpoint files (in order of preference):

1. `outputs/best_f1_model.pth` - Best macro F1 score model
2. `outputs/best_acc_model.pth` - Best accuracy model
3. `outputs/checkpoints/best_f1_model.pth` - Alternative location

## Output

The inference script generates comprehensive results:

### Files Created

- `test_results.json` - Complete metrics and analysis
- `test_predictions.pkl` - Raw predictions for further analysis
- `plots/test_confusion_matrix.png` - Confusion matrix visualization
- `plots/test_per_class_metrics.png` - Per-class precision, recall, F1
- `plots/test_class_distribution.png` - Test set class distribution
- `plots/misclassification_analysis.png` - Most common misclassifications
- `plots/per_class_metrics_best_f1.csv` - Detailed metrics in CSV format
- `plots/misclassification_analysis_detailed.csv` - Detailed misclassification data

### Metrics Reported

- **Overall Metrics**: Accuracy, Precision, Recall, F1-Score (both macro and weighted)
- **Per-Class Metrics**: Precision, Recall, F1-Score for each disease class
- **Confusion Matrix**: Visual representation of classification performance
- **Misclassification Analysis**: Most common classification errors

## Class Mapping & Risk Levels

Both the scripts and GUI use the exact same class mapping as the training script:

```python
CLASS_NAMES = [
    'nevus',           # 0 - Benign mole (Low risk)
    'melanoma',        # 1 - Malignant melanoma (Critical risk)
    'bcc',             # 2 - Basal cell carcinoma (High risk)
    'keratosis',       # 3 - Seborrheic keratosis (Low risk)
    'actinic_keratosis', # 4 - Actinic keratosis (Moderate risk)
    'scc',             # 5 - Squamous cell carcinoma (High risk)
    'dermatofibroma',  # 6 - Dermatofibroma (Low risk)
    'lentigo',         # 7 - Solar lentigo (Low risk)
    'vascular_lesion'  # 8 - Vascular lesion (Low risk)
]
```

### GUI Risk Level Color Coding

The GUI uses color coding to indicate medical risk levels:
- 🔴 **Red**: Critical risk (melanoma)
- 🟠 **Orange**: High risk (BCC, SCC)
- 🟡 **Yellow**: Moderate risk (actinic keratosis)
- 🟢 **Green**: Low risk (nevus, keratosis, dermatofibroma, lentigo, vascular lesion)

## Model Architecture

The inference script automatically reconstructs the same model architecture used in training:

- **Backbone**: Swin Transformer (configurable model name)
- **Classifier**: 2-layer MLP with dropout (512 hidden units)
- **Input Size**: 224x224 (configurable)
- **Number of Classes**: 9

## Example Usage

### GUI Application (Single Images)

1. **Launch GUI:**
   ```bash
   cd inference_gui
   python run_gui.py
   ```

2. **Load Model:**
   - Click "📂 Load Model Weights"
   - Select your trained `.pth` file (e.g., `best_f1_model.pth`)
   - GUI will show model info and validation metrics

3. **Load Image:**
   - Click "📂 Load Image" 
   - Select any skin lesion image (JPG, PNG, etc.)
   - Image will be displayed in the preview panel

4. **Run Classification:**
   - Click "🚀 Run Classification"
   - View results with confidence scores and risk levels
   - See probability distribution for all classes

### Command Line Scripts (Test Sets)

#### Basic inference with auto-detection:
```bash
cd inference_gui
python run_inference.py
```

#### Custom paths:
```bash
python run_inference.py \
    --checkpoint ../hpc_scripts_transformers/outputs/best_f1_model.pth \
    --data_dir ../DataSplit \
    --output_dir ./my_test_results
```

#### Direct script with specific model:
```bash
python swin_inference.py \
    --checkpoint_path /path/to/model.pth \
    --data_dir /path/to/DataSplit \
    --model_name swin_base_patch4_window7_224 \
    --image_size 224 \
    --batch_size 16 \
    --output_dir ./results
```

## 🎯 Threshold Tuning for Improved Performance

The inference system includes advanced **threshold tuning** capabilities that can significantly improve F1 scores, especially for imbalanced datasets like medical classification.

### What is Threshold Tuning?

Instead of using standard argmax (0.5 threshold for all classes), threshold tuning optimizes per-class decision thresholds using coordinate ascent to maximize your chosen metric (typically F1-score).

### Complete Workflow

#### 1. Tune Thresholds on Validation Set
```bash
python threshold_workflow.py \
    --checkpoint_path /path/to/best_model.pth \
    --data_dir /path/to/DataSplit \
    --output_dir ./threshold_results \
    --metric f1 \
    --passes 3
```

This will:
- ✅ Tune thresholds using validation set
- ✅ Evaluate on test set with both standard and optimized thresholds
- ✅ Save optimized thresholds for GUI use
- ✅ Generate comprehensive performance report

#### 2. Use in Batch Inference
```bash
python swin_inference.py \
    --checkpoint_path /path/to/model.pth \
    --data_dir /path/to/DataSplit \
    --tune_thresholds \
    --output_dir ./results
```

#### 3. Use in GUI
1. Run threshold workflow to generate `optimized_thresholds.json`
2. Launch GUI: `python run_gui.py`
3. Load model and thresholds
4. Check "Use optimized thresholds" for improved predictions

### Expected Improvements

Threshold tuning typically provides:
- 📈 **F1-Score**: +0.02 to +0.10 improvement (macro average)
- 🎯 **Minority Classes**: Significant recall improvements for rare diseases
- ⚖️ **Precision/Recall Balance**: Better trade-offs for medical diagnosis

### Example Results
```
Standard (argmax) Results:
   F1 (macro): 0.7234
   Accuracy:   0.8156

Threshold-Tuned Results:
   F1 (macro): 0.7891 (+0.0657)
   Accuracy:   0.8203 (+0.0047)

Optimized Thresholds:
          nevus: 0.524
       melanoma: 0.312  # Lower threshold for critical class
            bcc: 0.445
      keratosis: 0.591
actinic_keratosis: 0.398  # Lower for pre-cancerous
            scc: 0.378  # Lower for malignant
  dermatofibroma: 0.612
        lentigo: 0.456
 vascular_lesion: 0.518
```

## Troubleshooting

### Common Issues

1. **"Checkpoint not found"**: Make sure your model was saved properly during training
2. **"Test directory not found"**: Ensure your data has a `test/` subfolder with class directories
3. **"CUDA out of memory"**: Reduce batch size with `--batch_size 8` or `--batch_size 4`
4. **Model architecture mismatch**: Ensure `--model_name` and `--image_size` match training

### Debugging

Add verbose output by running the scripts with Python's verbose flag:
```bash
python -v run_inference.py
```

Check GPU availability:
```bash
python run_inference.py --gpu
```

## Integration with Training

The inference scripts are designed to work seamlessly with models trained using:
- `../hpc_scripts_transformers/train_swin_transformer.py`
- `../hpc_scripts_transformers/train_swin_transformer_two_stage.py`

Make sure the model name and image size match your training configuration.
