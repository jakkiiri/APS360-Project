#!/usr/bin/env python3
"""
About DermAI - Ocean Blue Theme
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="About | DermAI", page_icon="ℹ️", layout="wide")

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
.about-hero {
    background: linear-gradient(135deg, #0c1222 0%, #1a2540 100%);
    border-radius: 24px;
    padding: 3rem;
    margin-bottom: 2rem;
    border: 1px solid var(--dark-border);
    text-align: center;
    position: relative;
    overflow: hidden;
}

.about-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: 50%;
    transform: translateX(-50%);
    width: 80%;
    height: 80%;
    background: radial-gradient(circle, rgba(14, 165, 233, 0.1) 0%, transparent 60%);
}

.about-hero .logo { font-size: 4rem; margin-bottom: 1rem; position: relative; }

.about-hero h1 {
    font-size: 2.75rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0 0 1rem 0;
    position: relative;
}

.about-hero h1 span {
    background: linear-gradient(135deg, var(--primary) 0%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.about-hero p {
    font-size: 1.15rem;
    color: var(--text-secondary);
    max-width: 650px;
    margin: 0 auto;
    line-height: 1.7;
    position: relative;
}

/* Sections */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2.5rem 0 1.25rem;
}

.section-icon {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
}

.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
}

/* Cards */
.info-card {
    background: var(--dark-surface);
    border: 1px solid var(--dark-border);
    border-radius: 16px;
    padding: 1.75rem;
    height: 100%;
}

.info-card h3 {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--primary-light);
    margin: 0 0 1.25rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.info-card ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

.info-card li {
    padding: 0.5rem 0;
    color: var(--text-secondary);
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.info-card li::before {
    content: '→';
    color: var(--primary);
}

/* Condition cards */
.condition-card {
    background: var(--dark-surface);
    border-radius: 14px;
    padding: 1.25rem;
    border-left: 4px solid;
    margin-bottom: 0.75rem;
}

.condition-card.low { border-left-color: #10b981; }
.condition-card.moderate { border-left-color: #f59e0b; }
.condition-card.high { border-left-color: #f97316; }
.condition-card.critical { border-left-color: #ef4444; }

.condition-card .header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}

.condition-card .emoji { font-size: 1.5rem; }
.condition-card .name { font-size: 1.05rem; font-weight: 700; color: var(--text-primary); }

.condition-card .badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 100px;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.badge-low { background: rgba(16, 185, 129, 0.15); color: #10b981; }
.badge-moderate { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
.badge-high { background: rgba(249, 115, 22, 0.15); color: #f97316; }
.badge-critical { background: rgba(239, 68, 68, 0.15); color: #ef4444; }

.condition-card .desc { color: var(--text-secondary); font-size: 0.9rem; line-height: 1.5; margin: 0; }

/* Tech grid */
.tech-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}

.tech-card {
    background: var(--dark-surface);
    border: 1px solid var(--dark-border);
    border-radius: 14px;
    padding: 1.25rem 0.75rem;
    text-align: center;
    transition: all 0.2s ease;
}

.tech-card:hover {
    border-color: var(--primary);
    transform: translateY(-3px);
}

.tech-card .icon { font-size: 2rem; margin-bottom: 0.5rem; }
.tech-card .name { font-size: 0.85rem; font-weight: 600; color: var(--text-primary); }

/* Disclaimer */
.disclaimer {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.25);
    border-radius: 16px;
    padding: 1.75rem;
    margin: 2rem 0;
}

.disclaimer h4 {
    color: #f59e0b;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 0 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.disclaimer p {
    color: var(--text-secondary);
    font-size: 0.95rem;
    line-height: 1.7;
    margin: 0;
}

/* Footer */
.footer {
    text-align: center;
    padding: 1.5rem 0;
    color: #64748b;
    font-size: 0.9rem;
}

.footer strong { color: var(--primary-light); }

hr {
    margin: 2rem 0 !important;
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--dark-border), transparent) !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONTENT
# ============================================================================

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

st.markdown("""
<div class="about-hero">
    <div class="logo">🔬</div>
    <h1>About <span>DermAI</span></h1>
    <p>Advanced skin lesion classification powered by state-of-the-art deep learning, 
    built for APS360 at the University of Toronto.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

st.markdown('<div class="section-header"><div class="section-icon">🎯</div><div class="section-title">Project Overview</div></div>', unsafe_allow_html=True)

st.markdown("""
**DermAI** uses the **Swin Transformer** architecture—a hierarchical vision transformer with state-of-the-art 
image classification performance. Trained on **HAM10000**, **BCN20000**, and **PAD-UFES-20** datasets, 
it classifies skin lesions into **9 categories** with confidence scores and risk assessments.
""")

st.divider()

st.markdown('<div class="section-header"><div class="section-icon">🏗️</div><div class="section-title">Architecture & Training</div></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>🧠 Architecture</h3>
        <ul>
            <li>Swin Transformer Base backbone</li>
            <li>4×4 patch embeddings</li>
            <li>7×7 window attention</li>
            <li>512×512 input resolution</li>
            <li>9 output classes</li>
            <li>~88 million parameters</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>⚙️ Training</h3>
        <ul>
            <li>AdamW optimizer</li>
            <li>Cosine annealing scheduler</li>
            <li>Focal Loss for imbalance</li>
            <li>Albumentations augmentation</li>
            <li>Two-stage fine-tuning</li>
            <li>Mixed precision (FP16)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.markdown('<div class="section-header"><div class="section-icon">🩺</div><div class="section-title">Skin Conditions</div></div>', unsafe_allow_html=True)

conditions = [
    ("🟢", "Nevus", "low", "Common benign mole"),
    ("🔴", "Melanoma", "critical", "Dangerous skin cancer"),
    ("🟠", "BCC", "high", "Basal cell carcinoma"),
    ("🟢", "Keratosis", "low", "Seborrheic keratosis"),
    ("🟡", "Actinic Keratosis", "moderate", "Pre-cancerous lesion"),
    ("🟠", "SCC", "high", "Squamous cell carcinoma"),
    ("🟢", "Dermatofibroma", "low", "Benign nodule"),
    ("🟢", "Lentigo", "low", "Solar age spot"),
    ("🟢", "Vascular Lesion", "low", "Blood vessel marking"),
]

cols = st.columns(3)
for i, (emoji, name, risk, desc) in enumerate(conditions):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="condition-card {risk}">
            <div class="header">
                <span class="emoji">{emoji}</span>
                <span class="name">{name}</span>
            </div>
            <span class="badge badge-{risk}">{risk} Risk</span>
            <p class="desc">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

st.markdown('<div class="section-header"><div class="section-icon">🛠️</div><div class="section-title">Technology</div></div>', unsafe_allow_html=True)

st.markdown("""
<div class="tech-grid">
    <div class="tech-card"><div class="icon">🔥</div><div class="name">PyTorch</div></div>
    <div class="tech-card"><div class="icon">🤗</div><div class="name">timm</div></div>
    <div class="tech-card"><div class="icon">🎨</div><div class="name">Streamlit</div></div>
    <div class="tech-card"><div class="icon">📊</div><div class="name">Plotly</div></div>
    <div class="tech-card"><div class="icon">🖼️</div><div class="name">Albumentations</div></div>
    <div class="tech-card"><div class="icon">🐍</div><div class="name">Python</div></div>
</div>
""", unsafe_allow_html=True)

st.divider()

st.markdown("""
<div class="disclaimer">
    <h4>⚠️ Medical Disclaimer</h4>
    <p>This AI tool is for <strong>educational purposes only</strong>. It is NOT a substitute for professional 
    medical advice. Always consult a qualified dermatologist for proper evaluation and treatment.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <strong>DermAI</strong> · APS360 Applied Fundamentals of Deep Learning · University of Toronto
</div>
""", unsafe_allow_html=True)
