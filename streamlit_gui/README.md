# 🔬 AI Derm — Intelligent Skin Lesion Analysis

A modern, professional web interface for AI-powered skin lesion classification using Swin Transformer deep learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)

## ✨ Features

- **🎨 Ocean Blue Theme** — Professional dark UI with calming blue accents
- **🔬 Instant AI Analysis** — Get predictions in seconds with confidence scores
- **🖼️ Sample Gallery** — Click-to-try example images from all 9 categories
- **📊 Batch Processing** — Analyze multiple images at once
- **⚠️ Risk Assessment** — Color-coded risk levels (Critical/High/Moderate/Low)
- **📈 Interactive Charts** — Beautiful probability visualizations with Plotly
- **🚀 GPU Acceleration** — Automatic CUDA detection and usage

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd streamlit_gui
pip install -r requirements.txt
```

### 2. Run the App

**Option A — Double-click the batch file:**
```
run.bat
```

**Option B — Command line:**
```bash
streamlit run Home.py
```

The app opens automatically at `http://localhost:8501`

## 📁 Project Structure

```
streamlit_gui/
├── Home.py                   # Main application
├── model_utils.py            # Model loading utilities
├── threshold_tuning.py       # Threshold optimization
├── requirements.txt          # Python dependencies
├── run.bat                   # Windows launcher
├── README.md                 # This file
├── sample_images/            # Example images (10 per class)
│   ├── nevus/
│   ├── melanoma/
│   ├── bcc/
│   └── ... (9 classes)
└── pages/
    ├── 1_Batch_Inference.py
    └── 2_About.py
```

## 🩺 Supported Conditions

| Condition | Risk Level | Description |
|-----------|------------|-------------|
| Nevus | 🟢 Low | Benign mole |
| Melanoma | 🔴 Critical | Malignant skin cancer |
| BCC | 🟠 High | Basal cell carcinoma |
| Keratosis | 🟢 Low | Seborrheic keratosis |
| Actinic Keratosis | 🟡 Moderate | Pre-cancerous lesion |
| SCC | 🟠 High | Squamous cell carcinoma |
| Dermatofibroma | 🟢 Low | Benign fibrous nodule |
| Lentigo | 🟢 Low | Solar lentigo (age spot) |
| Vascular Lesion | 🟢 Low | Blood vessel marking |

## 🏗️ Model Architecture

- **Backbone:** Swin Transformer Base (patch 4, window 7)
- **Input:** 512×512 RGB images
- **Parameters:** ~88 million
- **Output:** 9 skin condition classes

## 💻 System Requirements

- Python 3.8+
- 4GB+ RAM (8GB recommended)
- GPU with CUDA (optional, for faster inference)

## ⚠️ Medical Disclaimer

This AI tool is for **educational and research purposes only**. It is NOT a substitute for professional medical diagnosis. Always consult a qualified dermatologist or healthcare provider.

## 📝 License

Part of the APS360 Project — University of Toronto
