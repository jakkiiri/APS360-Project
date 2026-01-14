#!/usr/bin/env python3
"""
DermAI - Skin Lesion Analysis
Ocean Blue Theme
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
    page_title="DermAI - Skin Lesion Analysis",
    page_icon="🔬",
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
    --accent: #06b6d4;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --dark-bg: #0c1222;
    --dark-surface: #131a2e;
    --dark-card: #1a2540;
    --dark-border: #2a3a5c;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
}

* {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ============ SIDEBAR ============ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1222 0%, #131a2e 100%) !important;
    border-right: 1px solid var(--dark-border) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

/* Custom sidebar header */
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
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
}

.sidebar-logo-text {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary-light) 0%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.sidebar-tagline {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
    padding-left: 0.25rem;
}

[data-testid="stSidebarNav"] {
    padding-top: 0.5rem;
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

[data-testid="stSidebarNav"] span {
    font-size: 1.15rem !important;
}

/* ============ MAIN CONTAINER ============ */
.block-container {
    max-width: 1100px;
    padding-top: 1rem;
}

/* ============ HERO ============ */
.hero {
    background: linear-gradient(135deg, #0c1222 0%, #1a2540 50%, #0f172a 100%);
    border-radius: 24px;
    padding: 3rem;
    margin-bottom: 2rem;
    border: 1px solid var(--dark-border);
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: -100px;
    right: -100px;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(14, 165, 233, 0.15) 0%, transparent 70%);
    pointer-events: none;
}

.hero::after {
    content: '';
    position: absolute;
    bottom: -50px;
    left: -50px;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(6, 182, 212, 0.1) 0%, transparent 70%);
    pointer-events: none;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(14, 165, 233, 0.15);
    border: 1px solid rgba(14, 165, 233, 0.3);
    border-radius: 100px;
    padding: 10px 20px;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--primary-light);
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 1rem;
    line-height: 1.2;
    position: relative;
}

.hero-title span {
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: var(--text-secondary);
    max-width: 600px;
    line-height: 1.7;
    position: relative;
}

.hero-stats {
    display: flex;
    gap: 3rem;
    margin-top: 2rem;
    position: relative;
}

.hero-stat-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--primary-light);
}

.hero-stat-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ============ STEP INDICATOR ============ */
.step-box {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 2rem 0 1.25rem;
    padding: 1.25rem 1.5rem;
    background: var(--dark-surface);
    border: 1px solid var(--dark-border);
    border-radius: 16px;
}

.step-num {
    width: 44px;
    height: 44px;
    min-width: 44px;
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    font-weight: 800;
    color: white;
}

.step-text h3 {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.25rem 0;
}

.step-text p {
    font-size: 0.95rem;
    color: var(--text-secondary);
    margin: 0;
}

/* ============ CATEGORY DISPLAY ============ */
.category-display {
    background: linear-gradient(135deg, var(--dark-surface) 0%, var(--dark-card) 100%);
    border: 1px solid var(--dark-border);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    display: flex;
    align-items: center;
    gap: 1.25rem;
}

.category-display .emoji {
    font-size: 3rem;
}

.category-display .info h4 {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.5rem 0;
}

.category-display .info p {
    font-size: 1rem;
    color: var(--text-secondary);
    margin: 0 0 0.5rem 0;
}

/* ============ RISK BADGES ============ */
.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-low { background: rgba(16, 185, 129, 0.15); color: #10b981; }
.badge-moderate { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
.badge-high { background: rgba(249, 115, 22, 0.15); color: #f97316; }
.badge-critical { background: rgba(239, 68, 68, 0.15); color: #ef4444; }

/* ============ RESULTS ============ */
.results-box {
    background: linear-gradient(135deg, #0c1222 0%, #1a2540 100%);
    border: 1px solid var(--dark-border);
    border-radius: 20px;
    padding: 2rem;
    margin: 1.5rem 0;
}

.results-comparison {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
}

.result-card {
    background: var(--dark-surface);
    border: 2px solid var(--dark-border);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    min-width: 180px;
}

.result-card.ground-truth {
    border-color: rgba(16, 185, 129, 0.4);
}

.result-card.ai-prediction {
    border-color: rgba(14, 165, 233, 0.4);
}

.result-card .label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-secondary);
    margin-bottom: 0.75rem;
}

.result-card .emoji {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.result-card .name {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
}

.result-arrow {
    font-size: 2rem;
    color: var(--primary);
}

/* ============ PLACEHOLDER ============ */
.placeholder-box {
    background: var(--dark-surface);
    border: 2px dashed var(--dark-border);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
}

.placeholder-box .icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
}

.placeholder-box h4 {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 0.5rem 0;
}

.placeholder-box p {
    color: var(--text-secondary);
    margin: 0;
}

/* ============ BUTTONS ============ */
.stButton > button {
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 0.875rem 1.75rem !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}

.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    border: none !important;
    color: white !important;
}

.stButton > button[data-testid="baseButton-primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(14, 165, 233, 0.3) !important;
}

.stButton > button[data-testid="baseButton-secondary"] {
    background: transparent !important;
    border: 2px solid var(--dark-border) !important;
    color: var(--text-primary) !important;
}

.stButton > button[data-testid="baseButton-secondary"]:hover {
    border-color: var(--primary) !important;
    color: var(--primary-light) !important;
}

/* ============ TABS ============ */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: var(--dark-surface);
    padding: 8px;
    border-radius: 14px;
    border: 1px solid var(--dark-border);
}

.stTabs [data-baseweb="tab"] {
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.875rem 1.5rem !important;
    border-radius: 10px !important;
    background: transparent !important;
    color: var(--text-secondary) !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(14, 165, 233, 0.1) !important;
    color: var(--primary-light) !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
}

/* ============ METRICS ============ */
[data-testid="stMetric"] {
    background: var(--dark-surface) !important;
    border: 1px solid var(--dark-border) !important;
    border-radius: 14px !important;
    padding: 1.25rem !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--primary-light) !important;
}

/* ============ EXPANDER ============ */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    background: var(--dark-surface) !important;
    border-radius: 12px !important;
}

/* ============ ALERTS ============ */
.stAlert > div {
    border-radius: 12px !important;
}

/* ============ FILE UPLOADER ============ */
[data-testid="stFileUploader"] > div {
    border: 2px dashed var(--dark-border) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    background: var(--dark-surface) !important;
}

[data-testid="stFileUploader"] > div:hover {
    border-color: var(--primary) !important;
}

/* ============ SELECTBOX ============ */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: var(--dark-card) !important;
    border-color: var(--dark-border) !important;
    color: var(--text-primary) !important;
}

/* ============ DIVIDER ============ */
hr {
    margin: 2rem 0 !important;
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--dark-border), transparent) !important;
}

/* ============ IMAGES ============ */
[data-testid="stImage"] {
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# IMPORTS & CONFIG
# ============================================================================
try:
    from model_utils import safe_torch_load
except ImportError:
    def safe_torch_load(checkpoint_path, map_location=None):
        try:
            return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(checkpoint_path, map_location=map_location)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR.parent / "best_model" / "best_acc_model.pth"
SAMPLE_IMAGES_DIR = BASE_DIR / "sample_images"

CLASS_NAMES = [
    'nevus', 'melanoma', 'bcc', 'keratosis',
    'actinic_keratosis', 'scc', 'dermatofibroma', 'lentigo', 'vascular_lesion'
]

CLASS_INFO = {
    'nevus': {'display': 'Nevus', 'full': 'Melanocytic Nevus', 'risk': 'low', 'emoji': '🟢', 'desc': 'Common benign mole'},
    'melanoma': {'display': 'Melanoma', 'full': 'Malignant Melanoma', 'risk': 'critical', 'emoji': '🔴', 'desc': 'Dangerous skin cancer'},
    'bcc': {'display': 'BCC', 'full': 'Basal Cell Carcinoma', 'risk': 'high', 'emoji': '🟠', 'desc': 'Common skin cancer'},
    'keratosis': {'display': 'Keratosis', 'full': 'Seborrheic Keratosis', 'risk': 'low', 'emoji': '🟢', 'desc': 'Harmless growth'},
    'actinic_keratosis': {'display': 'Actinic Keratosis', 'full': 'Actinic Keratosis', 'risk': 'moderate', 'emoji': '🟡', 'desc': 'Pre-cancerous lesion'},
    'scc': {'display': 'SCC', 'full': 'Squamous Cell Carcinoma', 'risk': 'high', 'emoji': '🟠', 'desc': 'Skin cancer'},
    'dermatofibroma': {'display': 'Dermatofibroma', 'full': 'Dermatofibroma', 'risk': 'low', 'emoji': '🟢', 'desc': 'Benign nodule'},
    'lentigo': {'display': 'Lentigo', 'full': 'Solar Lentigo', 'risk': 'low', 'emoji': '🟢', 'desc': 'Age spot'},
    'vascular_lesion': {'display': 'Vascular Lesion', 'full': 'Vascular Lesion', 'risk': 'low', 'emoji': '🟢', 'desc': 'Blood vessel marking'}
}

# ============================================================================
# MODEL
# ============================================================================
class SwinTransformerClassifier(nn.Module):
    def __init__(self, num_classes=9, model_name='swin_base_patch4_window7_224', 
                 pretrained=True, image_size=224):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=image_size)
        feature_dim = self.backbone.num_features if hasattr(self.backbone, 'num_features') else 1024
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feature_dim, 512), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwinTransformerClassifier(num_classes=len(CLASS_NAMES), pretrained=False, image_size=512)
    checkpoint = safe_torch_load(str(MODEL_PATH), map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval()
    return model, device

def get_transform():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def run_inference(model, image, device):
    transform = get_transform()
    img_array = np.array(image.convert('RGB'))
    tensor = transform(image=img_array)['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
        all_probs = probs[0].cpu().numpy()
    return pred_idx, confidence, all_probs

def get_sample_images():
    samples = {}
    if SAMPLE_IMAGES_DIR.exists():
        for class_name in CLASS_NAMES:
            class_dir = SAMPLE_IMAGES_DIR / class_name
            if class_dir.exists():
                images = sorted(list(class_dir.glob("*.jpg")))[:10]
                if images:
                    samples[class_name] = images
    return samples

def create_confidence_gauge(confidence):
    color = '#10b981' if confidence > 0.7 else '#f59e0b' if confidence > 0.4 else '#ef4444'
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={'suffix': '%', 'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#64748b', 'tickfont': {'color': '#64748b'}},
            'bar': {'color': color},
            'bgcolor': '#1a2540',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.15)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
            ]
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#94a3b8'}
    )
    return fig

def create_prob_chart(probs):
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    sorted_names = [CLASS_INFO[CLASS_NAMES[i]]['display'] for i in sorted_idx]
    
    colors = []
    for i in sorted_idx:
        risk = CLASS_INFO[CLASS_NAMES[i]]['risk']
        if risk == 'critical': colors.append('#ef4444')
        elif risk == 'high': colors.append('#f97316')
        elif risk == 'moderate': colors.append('#f59e0b')
        else: colors.append('#10b981')
    
    fig = go.Figure(go.Bar(
        y=sorted_names[::-1],
        x=sorted_probs[::-1],
        orientation='h',
        marker=dict(color=colors[::-1], line=dict(width=0)),
        text=[f'{p:.1%}' for p in sorted_probs[::-1]],
        textposition='outside',
        textfont=dict(size=13, color='#e2e8f0')
    ))
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=70, t=20, b=20),
        xaxis=dict(range=[0, 1.1], tickformat='.0%', tickfont=dict(color='#64748b'), gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(automargin=True, tickfont=dict(size=13, color='#e2e8f0')),
        bargap=0.3,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Session state
    if 'selected_class' not in st.session_state:
        st.session_state.selected_class = 'nevus'
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'image_source' not in st.session_state:
        st.session_state.image_source = None
    
    # Sidebar header
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-logo">
                <div class="sidebar-logo-icon">🔬</div>
                <div class="sidebar-logo-text">DermAI</div>
            </div>
            <div class="sidebar-tagline">AI Skin Lesion Analysis</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🔬 AI-Powered Skin Analysis</div>
        <h1 class="hero-title"><span>DermAI</span> Skin Lesion Classifier</h1>
        <p class="hero-subtitle">
            State-of-the-art deep learning model using Swin Transformer architecture 
            to classify skin lesions into 9 categories with high accuracy.
        </p>
        <div class="hero-stats">
            <div><div class="hero-stat-value">9</div><div class="hero-stat-label">Classes</div></div>
            <div><div class="hero-stat-value">88M</div><div class="hero-stat-label">Parameters</div></div>
            <div><div class="hero-stat-value">512px</div><div class="hero-stat-label">Resolution</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    try:
        model, device = load_model()
        mode = "🚀 GPU" if torch.cuda.is_available() else "💻 CPU"
        st.success(f"Model loaded successfully — {mode}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
    
    st.divider()
    
    # Tabs
    tab1, tab2 = st.tabs(["🖼️ Sample Images", "📤 Upload Your Own"])
    
    # ==================== TAB 1 ====================
    with tab1:
        samples = get_sample_images()
        
        # Step 1
        st.markdown("""
        <div class="step-box">
            <div class="step-num">1</div>
            <div class="step-text">
                <h3>Select Ground Truth Category</h3>
                <p>Choose the known diagnosis to compare against AI prediction</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(3)
        for idx, class_name in enumerate(CLASS_NAMES):
            info = CLASS_INFO[class_name]
            with cols[idx % 3]:
                is_selected = st.session_state.selected_class == class_name
                btn_type = "primary" if is_selected else "secondary"
                if st.button(f"{info['emoji']} {info['display']}", key=f"cat_{class_name}", 
                           use_container_width=True, type=btn_type):
                    st.session_state.selected_class = class_name
                    st.session_state.selected_image = None
                    st.session_state.prediction = None
                    st.rerun()
        
        # Show selected category
        if st.session_state.selected_class:
            info = CLASS_INFO[st.session_state.selected_class]
            badge_class = f"badge-{info['risk']}"
            st.markdown(f"""
            <div class="category-display">
                <div class="emoji">{info['emoji']}</div>
                <div class="info">
                    <h4>{info['full']}</h4>
                    <p>{info['desc']}</p>
                    <span class="badge {badge_class}">{info['risk']} Risk</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Step 2
        st.markdown("""
        <div class="step-box">
            <div class="step-num">2</div>
            <div class="step-text">
                <h3>Select an Image</h3>
                <p>Click on a sample image to select it for analysis</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.selected_class and st.session_state.selected_class in samples:
            cols = st.columns(5)
            for idx, img_path in enumerate(samples[st.session_state.selected_class]):
                with cols[idx % 5]:
                    img = Image.open(img_path)
                    is_selected = (st.session_state.selected_image == img_path and 
                                   st.session_state.image_source == 'sample')
                    st.image(img, use_container_width=True)
                    btn_label = "✓ Selected" if is_selected else "Select"
                    if st.button(btn_label, key=f"img_{idx}", use_container_width=True,
                                type="primary" if is_selected else "secondary"):
                        st.session_state.selected_image = img_path
                        st.session_state.image_source = 'sample'
                        st.session_state.prediction = None
                        st.rerun()
        
        st.divider()
        
        # Step 3
        st.markdown("""
        <div class="step-box">
            <div class="step-num">3</div>
            <div class="step-text">
                <h3>Run AI Analysis</h3>
                <p>Click the button to classify the selected image</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.selected_image and st.session_state.image_source == 'sample':
            selected_img = Image.open(st.session_state.selected_image)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(selected_img, caption="Selected Image", use_container_width=True)
                if st.button("🔬 ANALYZE IMAGE", type="primary", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        pred_idx, confidence, all_probs = run_inference(model, selected_img, device)
                        st.session_state.prediction = {
                            'class_idx': pred_idx,
                            'class_name': CLASS_NAMES[pred_idx],
                            'confidence': confidence,
                            'probs': all_probs
                        }
                    st.rerun()
        else:
            st.markdown("""
            <div class="placeholder-box">
                <div class="icon">☝️</div>
                <h4>No Image Selected</h4>
                <p>Select an image from the gallery above</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Step 4 - Results
        if st.session_state.prediction and st.session_state.image_source == 'sample':
            st.divider()
            
            st.markdown("""
            <div class="step-box">
                <div class="step-num">4</div>
                <div class="step-text">
                    <h3>Results</h3>
                    <p>Comparison of ground truth vs AI prediction</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            pred = st.session_state.prediction
            pred_info = CLASS_INFO[pred['class_name']]
            gt_info = CLASS_INFO[st.session_state.selected_class]
            is_correct = pred['class_name'] == st.session_state.selected_class
            
            # Results comparison
            st.markdown(f"""
            <div class="results-box">
                <div class="results-comparison">
                    <div class="result-card ground-truth">
                        <div class="label">Ground Truth</div>
                        <div class="emoji">{gt_info['emoji']}</div>
                        <div class="name">{gt_info['display']}</div>
                    </div>
                    <div class="result-arrow">→</div>
                    <div class="result-card ai-prediction">
                        <div class="label">AI Prediction</div>
                        <div class="emoji">{pred_info['emoji']}</div>
                        <div class="name">{pred_info['display']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Use native Streamlit for verdict (guaranteed to work)
            if is_correct:
                st.success("✅ **CORRECT PREDICTION** — The AI correctly identified the skin lesion!")
            else:
                st.error("❌ **INCORRECT PREDICTION** — The AI misclassified the skin lesion.")
            
            # Confidence
            st.markdown("#### Confidence Score")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.plotly_chart(create_confidence_gauge(pred['confidence']), 
                               use_container_width=True, config={'displayModeBar': False})
            
            # Probabilities
            with st.expander("📊 All Class Probabilities", expanded=True):
                st.plotly_chart(create_prob_chart(pred['probs']), 
                               use_container_width=True, config={'displayModeBar': False})
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Try Another Image", use_container_width=True):
                    st.session_state.selected_image = None
                    st.session_state.prediction = None
                    st.rerun()
    
    # ==================== TAB 2 ====================
    with tab2:
        st.markdown("""
        <div class="step-box">
            <div class="step-num">1</div>
            <div class="step-text">
                <h3>Upload Image</h3>
                <p>Select a skin lesion image from your device</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png', 'webp'], 
                                         label_visibility="collapsed")
        
        if uploaded_file:
            uploaded_img = Image.open(uploaded_file)
            st.session_state.image_source = 'upload'
            
            st.markdown("""
            <div class="step-box">
                <div class="step-num">2</div>
                <div class="step-text">
                    <h3>Analyze</h3>
                    <p>Run the AI model on your image</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(uploaded_img, caption="Your Image", use_container_width=True)
                if st.button("🔬 ANALYZE", type="primary", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        pred_idx, confidence, all_probs = run_inference(model, uploaded_img, device)
                        st.session_state.upload_pred = {
                            'class_idx': pred_idx,
                            'class_name': CLASS_NAMES[pred_idx],
                            'confidence': confidence,
                            'probs': all_probs
                        }
                    st.rerun()
            
            if 'upload_pred' in st.session_state and st.session_state.upload_pred:
                st.divider()
                pred = st.session_state.upload_pred
                pred_info = CLASS_INFO[pred['class_name']]
                badge_class = f"badge-{pred_info['risk']}"
                
                st.markdown("### Analysis Results")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 2rem; background: var(--dark-surface); 
                                border-radius: 16px; border: 1px solid var(--dark-border);">
                        <div style="font-size: 4rem; margin-bottom: 0.5rem;">{pred_info['emoji']}</div>
                        <div style="font-size: 1.75rem; font-weight: 700; color: #f1f5f9;">{pred_info['display']}</div>
                        <div style="margin: 1rem 0;">
                            <span class="badge {badge_class}" style="padding: 8px 18px; font-size: 0.85rem;">
                                {pred_info['risk'].upper()} RISK
                            </span>
                        </div>
                        <div style="color: #94a3b8;">{pred_info['full']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.plotly_chart(create_confidence_gauge(pred['confidence']), 
                                   use_container_width=True, config={'displayModeBar': False})
                
                with st.expander("📊 All Probabilities", expanded=True):
                    st.plotly_chart(create_prob_chart(pred['probs']), 
                                   use_container_width=True, config={'displayModeBar': False})
    
    # Disclaimer
    st.divider()
    st.warning("""
    **⚠️ Medical Disclaimer:** This AI tool is for **educational purposes only**. 
    It is NOT a substitute for professional medical diagnosis. Always consult a qualified dermatologist.
    """)
    st.caption("Built with Swin Transformer · APS360 · University of Toronto")

if __name__ == "__main__":
    main()
