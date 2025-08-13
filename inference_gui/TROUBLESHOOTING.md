# Troubleshooting Guide

## Common Issues and Solutions

### 1. PyTorch Model Loading Error (`weights_only` parameter)

**Error Message:**
```
Failed to load model:
Weights only load failed. This file can still be loaded, to do so you have two options.
(1) In PyTorch 2.6, we changed the default value of the 'weights_only' argument...
```

**Cause:** 
This error occurs due to security changes in newer PyTorch versions (2.6+). The `torch.load()` function now requires explicit specification of the `weights_only` parameter.

**Solution:**
The inference scripts have been updated to handle this automatically. If you still encounter issues:

1. **Update your scripts** - Make sure you're using the latest versions with the fix
2. **Alternative loading** - If the issue persists, you can manually set the parameter:

```python
# Instead of:
checkpoint = torch.load(model_path)

# Use:
checkpoint = torch.load(model_path, weights_only=False)
```

**Why `weights_only=False`?**
Your trained model checkpoints contain more than just model weights - they include optimizer states, scheduler states, training metrics, etc. Setting `weights_only=False` allows loading these complete checkpoints.

### 2. CUDA Out of Memory

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. **Reduce batch size** in command line scripts:
   ```bash
   python swin_inference.py --batch_size 8  # or even smaller
   ```
2. **Use CPU** if GPU memory is limited:
   ```bash
   # Force CPU usage by not using CUDA
   export CUDA_VISIBLE_DEVICES=""
   python single_image_gui.py
   ```

### 3. Missing Dependencies

**Error Message:**
```
ImportError: No module named 'timm'
```

**Solution:**
Install all required packages:
```bash
pip install torch torchvision timm albumentations pillow matplotlib numpy
# Or use the requirements file:
pip install -r requirements.txt
```

### 4. tkinter Not Available (Linux)

**Error Message:**
```
ImportError: No module named 'tkinter'
```

**Solution:**
Install tkinter for your system:

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-tk
```

**CentOS/RHEL:**
```bash
sudo yum install tkinter
# or
sudo dnf install python3-tkinter
```

### 5. Model Architecture Mismatch

**Error Message:**
```
RuntimeError: Error(s) in loading state_dict...
size mismatch for classifier.3.weight
```

**Cause:** 
The model architecture doesn't match the saved checkpoint.

**Solution:**
1. **Check model name** - Ensure you're using the same model architecture:
   ```python
   # Make sure this matches your training:
   model_name = 'swin_base_patch4_window7_224'
   ```

2. **Check number of classes** - Verify you have 9 classes:
   ```python
   num_classes = 9  # Should match your training
   ```

3. **Check image size** - Ensure image size matches training:
   ```python
   image_size = 224  # Should match your training
   ```

### 6. Image Loading Issues

**Error Message:**
```
Cannot identify image file
```

**Solutions:**
1. **Check file format** - Ensure image is in supported format (JPG, PNG, JPEG)
2. **Check file corruption** - Try opening the image in another program
3. **Convert format** if needed:
   ```python
   from PIL import Image
   img = Image.open('image.webp').convert('RGB')
   img.save('image.jpg')
   ```

### 7. GUI Not Responding

**Symptoms:** GUI freezes during inference

**Solutions:**
1. **Wait for inference** - Large models may take time
2. **Check console** for error messages
3. **Restart GUI** if completely frozen
4. **Use smaller images** to reduce processing time

### 8. Inconsistent Class Predictions

**Issue:** Predictions don't match expected medical knowledge

**Debugging Steps:**
1. **Check class mapping** - Verify classes are correctly ordered:
   ```python
   print(CLASS_NAMES)
   # Should output: ['nevus', 'melanoma', 'bcc', ...]
   ```

2. **Verify image preprocessing** - Ensure same normalization as training
3. **Check model checkpoint** - Confirm you're loading the correct model

### 9. Permission Errors

**Error Message:**
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**
1. **Check file permissions** - Ensure you can read the model file
2. **Run from correct directory** - Make sure you have write access
3. **Use full paths** instead of relative paths

### 10. Performance Issues

**Symptoms:** Very slow inference

**Solutions:**
1. **Use GPU** if available:
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```
2. **Enable mixed precision** (if supported):
   ```python
   with torch.cuda.amp.autocast():
       output = model(input)
   ```
3. **Optimize batch size** for your hardware

## Getting Help

If you encounter issues not covered here:

1. **Check console output** for detailed error messages
2. **Run test scripts** to verify your environment:
   ```bash
   python test_gui_setup.py
   python test_inference_setup.py
   ```
3. **Verify PyTorch installation**:
   ```python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```

## Environment Verification

Run this quick check to verify your setup:

```python
import torch
import timm
import albumentations
from PIL import Image
import matplotlib.pyplot as plt

print("✅ All packages imported successfully")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# Test model creation
model = timm.create_model('swin_base_patch4_window7_224', pretrained=False)
print("✅ Model creation successful")
```

## Contact and Support

For additional help:
- Check the main README.md for usage examples
- Verify your training script compatibility
- Ensure all file paths are correct
