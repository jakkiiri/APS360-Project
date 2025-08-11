"""
Configuration file for Swin Transformer skin disease classification
Contains hyperparameters, paths, and model configurations
"""

import os

# =============================================================================
# Dataset Configuration
# =============================================================================

# Class names and mapping
CLASS_NAMES = [
    'nevus',                # 0 - Most common (benign moles)
    'melanoma',             # 1 - Malignant (minority class)
    'bcc',                  # 2 - Basal cell carcinoma
    'seborrheic_keratosis', # 3 - Benign lesions
    'actinic_keratosis',    # 4 - Pre-cancerous (minority class)
    'scc',                  # 5 - Squamous cell carcinoma (minority class)
    'dermatofibroma',       # 6 - Benign (minority class)
    'lentigo',              # 7 - Benign (minority class)
    'vascular_lesion'       # 8 - Vascular (minority class)
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Minority classes that need extra augmentation
MINORITY_CLASSES = [1, 4, 5, 6, 7, 8]  # melanoma, actinic_keratosis, scc, dermatofibroma, lentigo, vascular_lesion

# =============================================================================
# Model Configuration
# =============================================================================

# Available Swin Transformer variants
SWIN_MODELS = {
    'swin_tiny': 'swin_tiny_patch4_window7_224',
    'swin_small': 'swin_small_patch4_window7_224', 
    'swin_base': 'swin_base_patch4_window7_224',
    'swin_large': 'swin_large_patch4_window7_224'
}

# Default model configuration
DEFAULT_MODEL_CONFIG = {
    'model_name': 'swin_base_patch4_window7_224',
    'num_classes': 9,
    'pretrained': True,
    'dropout_rate': 0.3,
    'hidden_dim': 512
}

# =============================================================================
# Training Configuration
# =============================================================================

# Default training hyperparameters
DEFAULT_TRAINING_CONFIG = {
    # Data
    'image_size': 512,
    'batch_size': 16,
    'num_workers': 8,
    
    # Training
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'patience': 15,
    'max_grad_norm': 1.0,
    
    # Scheduler
    'scheduler_type': 'cosine_warm_restarts',
    'T_0': 10,
    'T_mult': 2,
    
    # Seeds
    'seed': 42
}

# =============================================================================
# Augmentation Configuration
# =============================================================================

# Base augmentation probabilities
BASE_AUG_CONFIG = {
    'horizontal_flip': 0.5,
    'vertical_flip': 0.3,
    'random_rotate_90': 0.5,
    'rotate_limit': 30,
    'rotate_prob': 0.7,
    'brightness_contrast_prob': 0.7,
    'color_jitter_prob': 0.6,
    'gaussian_noise_prob': 0.3,
    'gaussian_blur_prob': 0.3,
    'random_crop_prob': 0.8,
    'elastic_transform_prob': 0.3,
    'grid_distortion_prob': 0.3,
    'coarse_dropout_prob': 0.3
}

# Enhanced augmentation for minority classes
MINORITY_AUG_CONFIG = {
    'random_gamma_prob': 0.5,
    'clahe_prob': 0.5,
    'sharpen_prob': 0.3,
    'emboss_prob': 0.3
}

# =============================================================================
# HPC Configuration
# =============================================================================

# SLURM configuration for different resource requirements
SLURM_CONFIGS = {
    'quick_test': {
        'time': '2:00:00',
        'mem': '16G',
        'cpus_per_task': 4,
        'gres': 'gpu:v100:1'
    },
    'standard': {
        'time': '12:00:00',
        'mem': '32G',
        'cpus_per_task': 8,
        'gres': 'gpu:v100:1'
    },
    'long_training': {
        'time': '24:00:00',
        'mem': '64G',
        'cpus_per_task': 12,
        'gres': 'gpu:a100:1'
    }
}

# =============================================================================
# Weights & Biases Configuration
# =============================================================================

WANDB_CONFIG = {
    'project': 'skin-disease-swin-transformer',
    'entity': None,  # Set this to your W&B username/team
    'tags': ['swin-transformer', 'skin-disease', 'classification', 'medical-imaging'],
    'notes': 'Swin Transformer for 9-class skin disease classification with class imbalance handling'
}

# Metrics to track
WANDB_METRICS = [
    'train_loss', 'train_accuracy',
    'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1',
    'learning_rate', 'epoch'
]

# =============================================================================
# Path Configuration
# =============================================================================

# Default paths (modify these for your setup)
DEFAULT_PATHS = {
    'data_dir': '/path/to/skin_disease_data',
    'output_dir': './outputs',
    'log_dir': './logs',
    'cache_dir': './cache'
}

# =============================================================================
# Utility Functions
# =============================================================================

def get_model_config(model_size='base'):
    """Get model configuration for specified size"""
    config = DEFAULT_MODEL_CONFIG.copy()
    if model_size in SWIN_MODELS:
        config['model_name'] = SWIN_MODELS[f'swin_{model_size}']
    return config

def get_training_config(quick_test=False):
    """Get training configuration, optionally for quick testing"""
    config = DEFAULT_TRAINING_CONFIG.copy()
    if quick_test:
        config.update({
            'num_epochs': 5,
            'batch_size': 8,
            'patience': 3
        })
    return config

def get_slurm_config(config_type='standard'):
    """Get SLURM configuration for specified type"""
    return SLURM_CONFIGS.get(config_type, SLURM_CONFIGS['standard'])

def create_wandb_config(model_config, training_config):
    """Create W&B configuration from model and training configs"""
    wandb_config = WANDB_CONFIG.copy()
    wandb_config['config'] = {**model_config, **training_config}
    return wandb_config

# =============================================================================
# Class Weight Calculation
# =============================================================================

def calculate_inverse_class_weights(class_counts):
    """
    Calculate inverse class weights for handling imbalance
    
    Args:
        class_counts: dict mapping class_idx -> count
        
    Returns:
        dict mapping class_idx -> weight
    """
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    weights = {}
    for class_idx in range(num_classes):
        if class_idx in class_counts and class_counts[class_idx] > 0:
            weights[class_idx] = total_samples / (num_classes * class_counts[class_idx])
        else:
            weights[class_idx] = 1.0
    
    return weights

# =============================================================================
# Validation Functions
# =============================================================================

def validate_config(config):
    """Validate configuration parameters"""
    required_keys = ['image_size', 'batch_size', 'num_epochs', 'learning_rate']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    if config['image_size'] not in [224, 384, 512]:
        print(f"Warning: Unusual image size {config['image_size']}. Common sizes are 224, 384, 512")
    
    if config['batch_size'] < 1:
        raise ValueError("Batch size must be positive")
    
    if config['learning_rate'] <= 0:
        raise ValueError("Learning rate must be positive")
    
    return True

# =============================================================================
# Export commonly used configurations
# =============================================================================

# Quick access to common configurations
QUICK_TEST_CONFIG = get_training_config(quick_test=True)
STANDARD_CONFIG = get_training_config()
BASE_MODEL_CONFIG = get_model_config('base')
SMALL_MODEL_CONFIG = get_model_config('small')