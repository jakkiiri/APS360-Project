#!/usr/bin/env python3
"""
Batch Inference - Ocean Blue Theme
"""

import sys
import os
import warnings
import logging

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel(logging.ERROR)
for name in ['torch', 'torchvision', 'timm']:
    logging.getLogger(name).setLevel(logging.ERROR)

import streamlit as st

st.set_page_config(
    page_title="Batch Analysis | AI Derm",
    page_icon="📊",
    layout="wide"
)

import torch
torch.set_warn_always(False)

import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

# ============================================================================
# OCEAN BLUE THEME CSS
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #0ea5e9;
    --primary-dark: #0284c7;
    --primary-light: #38bdf8;
    --dark-bg: #0c1222;
    --dark-surface: #131a2e;
    --dark-card: #1a2540;
    --dark-border: #2a3a5c;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
}

* { font-family: 'Plus Jakarta Sans', sans-serif; }

.block-container { max-width: 1100px; padding-top: 1rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1222 0%, #131a2e 100%) !important;
    border-right: 1px solid var(--dark-border) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

.sidebar-header {
    padding: 1.5rem 1.25rem 1.5rem;
    border-bottom: 1px solid var(--dark-border);
    margin-bottom: 1rem;
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.sidebar-logo-icon {
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, var(--primary) 0%, #06b6d4 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
}

.sidebar-logo-text {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary-light) 0%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sidebar-tagline {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
    padding-left: 0.25rem;
}

[data-testid="stSidebarNav"] a {
    color: var(--text-secondary) !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 0.875rem 1.25rem !important;
    border-radius: 12px !important;
    margin: 0.25rem 0.5rem !important;
    border-left: 3px solid transparent !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebarNav"] a:hover {
    background: rgba(14, 165, 233, 0.1) !important;
    color: var(--primary-light) !important;
    border-left-color: var(--primary) !important;
}

[data-testid="stSidebarNav"] a[aria-selected="true"] {
    background: rgba(14, 165, 233, 0.15) !important;
    color: var(--primary-light) !important;
    border-left-color: var(--primary) !important;
}

/* Hero */
.page-hero {
    background: linear-gradient(135deg, #0c1222 0%, #1a2540 100%);
    border-radius: 20px;
    padding: 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid var(--dark-border);
}

.page-hero h1 {
    font-size: 2.25rem;
    font-weight: 800;
    color: var(--primary-light);
    margin: 0 0 0.5rem 0;
}

.page-hero p {
    font-size: 1.1rem;
    color: var(--text-secondary);
    margin: 0;
    max-width: 600px;
}

/* Section */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2rem 0 1rem;
}

.section-icon {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
}

.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--dark-surface) !important;
    border: 1px solid var(--dark-border) !important;
    border-radius: 14px !important;
    padding: 1.25rem !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--primary-light) !important;
}

/* Buttons */
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    border: none !important;
}

/* File uploader */
[data-testid="stFileUploader"] > div {
    border: 2px dashed var(--dark-border) !important;
    border-radius: 16px !important;
    background: var(--dark-surface) !important;
}

[data-testid="stFileUploader"] > div:hover {
    border-color: var(--primary) !important;
}

/* Placeholder */
.placeholder {
    background: var(--dark-surface);
    border: 2px dashed var(--dark-border);
    border-radius: 16px;
    padding: 3rem;
    text-align: center;
}

.placeholder .icon { font-size: 3rem; margin-bottom: 1rem; }
.placeholder h4 { font-size: 1.2rem; color: var(--text-primary); margin: 0 0 0.5rem 0; }
.placeholder p { color: var(--text-secondary); margin: 0; }

hr {
    margin: 1.5rem 0 !important;
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--dark-border), transparent) !important;
}
</style>
""", unsafe_allow_html=True)

# Model setup
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from model_utils import safe_torch_load
except ImportError:
    def safe_torch_load(checkpoint_path, map_location=None):
        try:
            return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(checkpoint_path, map_location=map_location)

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR.parent / "best_model" / "best_acc_model.pth"

CLASS_NAMES = ['nevus', 'melanoma', 'bcc', 'keratosis', 'actinic_keratosis', 'scc', 'dermatofibroma', 'lentigo', 'vascular_lesion']
RISK_INFO = {'nevus': 'Low', 'melanoma': 'Critical', 'bcc': 'High', 'keratosis': 'Low',
             'actinic_keratosis': 'Moderate', 'scc': 'High', 'dermatofibroma': 'Low', 'lentigo': 'Low', 'vascular_lesion': 'Low'}
DISPLAY_NAMES = {'nevus': 'Nevus', 'melanoma': 'Melanoma', 'bcc': 'BCC', 'keratosis': 'Keratosis',
                 'actinic_keratosis': 'Actinic Keratosis', 'scc': 'SCC', 'dermatofibroma': 'Dermatofibroma',
                 'lentigo': 'Lentigo', 'vascular_lesion': 'Vascular Lesion'}

class SwinTransformerClassifier(nn.Module):
    def __init__(self, num_classes=9, model_name='swin_base_patch4_window7_224', pretrained=True, image_size=224):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=image_size)
        feature_dim = self.backbone.num_features if hasattr(self.backbone, 'num_features') else 1024
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(feature_dim, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, num_classes))
    def forward(self, x):
        return self.classifier(self.backbone(x))

@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwinTransformerClassifier(num_classes=len(CLASS_NAMES), pretrained=False, image_size=512)
    checkpoint = safe_torch_load(str(MODEL_PATH), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.to(device).eval()
    return model, device

def get_transform():
    return A.Compose([A.Resize(512, 512), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])

# Sidebar header
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">🔬</div>
            <div class="sidebar-logo-text">AI Derm</div>
        </div>
        <div class="sidebar-tagline">AI Skin Lesion Analysis</div>
    </div>
    """, unsafe_allow_html=True)

# Main
st.markdown("""
<div class="page-hero">
    <h1>📊 Batch Analysis</h1>
    <p>Process multiple skin lesion images at once and get comprehensive analysis with statistics.</p>
</div>
""", unsafe_allow_html=True)

try:
    model, device = load_model()
    st.success(f"Model loaded — {'🚀 GPU' if torch.cuda.is_available() else '💻 CPU'}")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.divider()

st.markdown('<div class="section-header"><div class="section-icon">📤</div><div class="section-title">Upload Images</div></div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload multiple images", type=['jpg', 'jpeg', 'png', 'webp'], accept_multiple_files=True, label_visibility="collapsed")

if uploaded_files:
    st.info(f"📁 **{len(uploaded_files)} images** ready")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔬 ANALYZE ALL", type="primary", use_container_width=True):
            results = []
            transform = get_transform()
            progress = st.progress(0)
            
            for i, f in enumerate(uploaded_files):
                progress.progress((i + 1) / len(uploaded_files), f"Processing {f.name}...")
                try:
                    img = Image.open(f).convert('RGB')
                    tensor = transform(image=np.array(img))['image'].unsqueeze(0).to(device)
                    with torch.no_grad():
                        probs = F.softmax(model(tensor), dim=1)
                        pred_idx = torch.argmax(probs, dim=1).item()
                        conf = probs[0][pred_idx].item()
                    results.append({'Filename': f.name, 'Prediction': DISPLAY_NAMES[CLASS_NAMES[pred_idx]], 
                                   'Confidence': f"{conf:.1%}", 'Risk': RISK_INFO[CLASS_NAMES[pred_idx]]})
                except:
                    results.append({'Filename': f.name, 'Prediction': 'Error', 'Confidence': '-', 'Risk': '-'})
            
            progress.empty()
            st.session_state.batch_results = results
            st.rerun()

st.divider()

if 'batch_results' in st.session_state and st.session_state.batch_results:
    results = st.session_state.batch_results
    df = pd.DataFrame(results)
    valid = df[df['Prediction'] != 'Error']
    
    st.markdown('<div class="section-header"><div class="section-icon">📈</div><div class="section-title">Summary</div></div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(results))
    c2.metric("Processed", len(valid))
    c3.metric("High Risk", len(valid[valid['Risk'].isin(['Critical', 'High'])]))
    c4.metric("Low Risk", len(valid[~valid['Risk'].isin(['Critical', 'High'])]))
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header"><div class="section-icon">🎯</div><div class="section-title">Predictions</div></div>', unsafe_allow_html=True)
        if len(valid) > 0:
            dist = valid['Prediction'].value_counts()
            fig = go.Figure(go.Pie(labels=dist.index, values=dist.values, hole=0.5, 
                                   marker=dict(colors=['#0ea5e9', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#f97316', '#14b8a6'])))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.markdown('<div class="section-header"><div class="section-icon">⚠️</div><div class="section-title">Risk Levels</div></div>', unsafe_allow_html=True)
        if len(valid) > 0:
            risk_dist = valid['Risk'].value_counts().reindex(['Critical', 'High', 'Moderate', 'Low']).dropna()
            colors = {'Critical': '#ef4444', 'High': '#f97316', 'Moderate': '#f59e0b', 'Low': '#10b981'}
            fig = go.Figure(go.Bar(x=risk_dist.index, y=risk_dist.values, 
                                   marker=dict(color=[colors.get(r, '#0ea5e9') for r in risk_dist.index]),
                                   text=risk_dist.values, textposition='outside'))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=50), paper_bgcolor='rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(tickfont=dict(color='#e2e8f0')),
                            yaxis=dict(tickfont=dict(color='#94a3b8'), gridcolor='rgba(255,255,255,0.05)'))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.divider()
    st.markdown('<div class="section-header"><div class="section-icon">📋</div><div class="section-title">Results</div></div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📥 Download CSV", df.to_csv(index=False), "results.csv", "text/csv", use_container_width=True)
    with c2:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.batch_results = None
            st.rerun()
else:
    st.markdown('<div class="placeholder"><div class="icon">☁️</div><h4>No Images Uploaded</h4><p>Upload images above to begin</p></div>', unsafe_allow_html=True)

st.divider()
st.caption("AI Derm · Swin Transformer · APS360")
