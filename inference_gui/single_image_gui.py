#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swin Transformer Single Image Inference GUI
Interactive application for loading model weights and running inference on individual images
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkFont
import threading
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# Import our utility functions
try:
    from model_utils import safe_torch_load, print_model_info
    from threshold_tuning import ThresholdTuner, adjusted_argmax
except ImportError:
    # Fallback if utility modules are not available
    def safe_torch_load(checkpoint_path, map_location=None):
        try:
            return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(checkpoint_path, map_location=map_location)
    
    def print_model_info(model, checkpoint=None):
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
    
    # Minimal threshold tuning fallback
    ThresholdTuner = None
    adjusted_argmax = None

# Set matplotlib backend for GUI
plt.style.use('default')

# Class mapping - MUST match training script exactly
CLASS_NAMES = [
    'nevus', 'melanoma', 'bcc', 'keratosis',
    'actinic_keratosis', 'scc', 'dermatofibroma', 'lentigo', 'vascular_lesion'
]

# Class descriptions for better understanding
CLASS_DESCRIPTIONS = {
    'nevus': 'Benign mole (non-cancerous)',
    'melanoma': 'Malignant melanoma (dangerous cancer)',
    'bcc': 'Basal cell carcinoma (common skin cancer)', 
    'keratosis': 'Seborrheic keratosis (benign growth)',
    'actinic_keratosis': 'Actinic keratosis (pre-cancerous)',
    'scc': 'Squamous cell carcinoma (skin cancer)',
    'dermatofibroma': 'Dermatofibroma (benign fibrous nodule)',
    'lentigo': 'Solar lentigo (age spot)',
    'vascular_lesion': 'Vascular lesion (blood vessel related)'
}

# Risk levels for medical context
RISK_LEVELS = {
    'nevus': 'Low',
    'melanoma': 'Critical',
    'bcc': 'High',
    'keratosis': 'Low',
    'actinic_keratosis': 'Moderate',
    'scc': 'High',
    'dermatofibroma': 'Low',
    'lentigo': 'Low',
    'vascular_lesion': 'Low'
}

class SwinTransformerClassifier(nn.Module):
    """Swin Transformer model for skin disease classification - matches training script"""
    
    def __init__(self, num_classes=9, model_name='swin_base_patch4_window7_224', pretrained=True, image_size=224):
        super(SwinTransformerClassifier, self).__init__()
        
        # Load pre-trained Swin Transformer
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=image_size)
        
        # Get feature dimension
        if hasattr(self.backbone, 'num_features'):
            feature_dim = self.backbone.num_features
        else:
            # Fallback to forward pass probing in eval mode
            self.backbone.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, image_size, image_size)
                feature_dim = self.backbone(dummy_input).shape[1]
            self.backbone.train()
        
        # Classification head with dropout - matches training script
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class SkinDiseaseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Swin Transformer - Skin Disease Classification")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_image = None
        self.current_image_path = None
        self.model_loaded = False
        self.threshold_tuner = None
        self.use_tuned_thresholds = False
        
        # Setup image preprocessing
        self.setup_preprocessing()
        
        # Setup GUI
        self.setup_styles()
        self.create_widgets()
        self.update_status()
        
    def setup_preprocessing(self):
        """Setup image preprocessing pipeline"""
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def setup_styles(self):
        """Setup custom styles for the GUI"""
        self.style = ttk.Style()
        
        # Configure styles
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        self.style.configure('Heading.TLabel', font=('Arial', 12, 'bold'))
        self.style.configure('Info.TLabel', font=('Arial', 10))
        self.style.configure('Success.TLabel', font=('Arial', 10), foreground='green')
        self.style.configure('Error.TLabel', font=('Arial', 10), foreground='red')
        self.style.configure('Warning.TLabel', font=('Arial', 10), foreground='orange')
        
    def create_widgets(self):
        """Create and layout GUI widgets"""
        # Main title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(title_frame, text="🔬 Swin Transformer Skin Disease Classifier", 
                 style='Title.TLabel').pack()
        ttk.Label(title_frame, text="Load model weights and classify skin lesion images", 
                 style='Info.TLabel').pack()
        
        # Create main container with three columns
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Model and Image loading
        left_frame = ttk.LabelFrame(main_frame, text="📁 Load Model & Image", padding=10)
        left_frame.pack(side='left', fill='y', padx=(0, 5))
        
        self.create_model_section(left_frame)
        self.create_image_section(left_frame)
        self.create_inference_section(left_frame)
        
        # Middle panel - Image display
        middle_frame = ttk.LabelFrame(main_frame, text="🖼️ Image Preview", padding=10)
        middle_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        self.create_image_display(middle_frame)
        
        # Right panel - Results
        right_frame = ttk.LabelFrame(main_frame, text="📊 Classification Results", padding=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        self.create_results_section(right_frame)
        
        # Status bar
        self.create_status_bar()
    
    def create_model_section(self, parent):
        """Create model loading section"""
        model_frame = ttk.LabelFrame(parent, text="🤖 Model", padding=5)
        model_frame.pack(fill='x', pady=(0, 10))
        
        # Model file selection
        ttk.Button(model_frame, text="📂 Load Model Weights", 
                  command=self.load_model, width=20).pack(pady=2)
        
        # Model info
        self.model_info_var = tk.StringVar(value="No model loaded")
        self.model_info_label = ttk.Label(model_frame, textvariable=self.model_info_var,
                                         style='Info.TLabel', wraplength=200)
        self.model_info_label.pack(pady=2)
        
        # Device info
        device_text = f"Device: {self.device}"
        if torch.cuda.is_available():
            device_text += f"\nGPU: {torch.cuda.get_device_name(0)}"
        ttk.Label(model_frame, text=device_text, style='Info.TLabel').pack(pady=2)
    
    def create_image_section(self, parent):
        """Create image loading section"""
        image_frame = ttk.LabelFrame(parent, text="🖼️ Image", padding=5)
        image_frame.pack(fill='x', pady=(0, 10))
        
        # Image file selection
        ttk.Button(image_frame, text="📂 Load Image", 
                  command=self.load_image, width=20).pack(pady=2)
        
        # Image info
        self.image_info_var = tk.StringVar(value="No image loaded")
        self.image_info_label = ttk.Label(image_frame, textvariable=self.image_info_var,
                                         style='Info.TLabel', wraplength=200)
        self.image_info_label.pack(pady=2)
        
        # Supported formats
        ttk.Label(image_frame, text="Supported: JPG, PNG, JPEG", 
                 style='Info.TLabel').pack(pady=2)
    
    def create_inference_section(self, parent):
        """Create inference control section"""
        inference_frame = ttk.LabelFrame(parent, text="🔍 Inference", padding=5)
        inference_frame.pack(fill='x', pady=(0, 10))
        
        # Inference button
        self.inference_button = ttk.Button(inference_frame, text="🚀 Run Classification", 
                                          command=self.run_inference, width=20, state='disabled')
        self.inference_button.pack(pady=2)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(inference_frame, variable=self.progress_var, 
                                           mode='indeterminate')
        self.progress_bar.pack(fill='x', pady=2)
        
        # Threshold tuning controls
        if ThresholdTuner is not None:
            threshold_frame = ttk.LabelFrame(inference_frame, text="🎯 Thresholds", padding=2)
            threshold_frame.pack(fill='x', pady=2)
            
            # Checkbox for using tuned thresholds
            self.use_thresholds_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(threshold_frame, text="Use optimized thresholds", 
                           variable=self.use_thresholds_var,
                           command=self.on_threshold_toggle).pack(anchor='w')
            
            # Load/save threshold buttons
            thresh_btn_frame = ttk.Frame(threshold_frame)
            thresh_btn_frame.pack(fill='x', pady=1)
            
            ttk.Button(thresh_btn_frame, text="📂 Load", width=8,
                      command=self.load_thresholds).pack(side='left', padx=1)
            ttk.Button(thresh_btn_frame, text="💾 Save", width=8,
                      command=self.save_thresholds).pack(side='left', padx=1)
        
        # Clear results button
        ttk.Button(inference_frame, text="🗑️ Clear Results", 
                  command=self.clear_results, width=20).pack(pady=2)
    
    def create_image_display(self, parent):
        """Create image display area"""
        # Image canvas
        self.image_canvas = tk.Canvas(parent, bg='white', width=400, height=400)
        self.image_canvas.pack(expand=True, fill='both', pady=10)
        
        # Placeholder text
        self.image_canvas.create_text(200, 200, text="No image loaded\nClick 'Load Image' to start",
                                     fill='gray', font=('Arial', 12), justify='center')
    
    def create_results_section(self, parent):
        """Create results display section"""
        # Prediction result
        self.prediction_frame = ttk.LabelFrame(parent, text="🎯 Prediction", padding=5)
        self.prediction_frame.pack(fill='x', pady=(0, 10))
        
        self.prediction_var = tk.StringVar(value="No prediction yet")
        self.prediction_label = ttk.Label(self.prediction_frame, textvariable=self.prediction_var,
                                         style='Heading.TLabel', wraplength=300)
        self.prediction_label.pack(pady=2)
        
        self.confidence_var = tk.StringVar(value="")
        self.confidence_label = ttk.Label(self.prediction_frame, textvariable=self.confidence_var,
                                         style='Info.TLabel')
        self.confidence_label.pack(pady=2)
        
        self.description_var = tk.StringVar(value="")
        self.description_label = ttk.Label(self.prediction_frame, textvariable=self.description_var,
                                          style='Info.TLabel', wraplength=300)
        self.description_label.pack(pady=2)
        
        # Probability distribution
        prob_frame = ttk.LabelFrame(parent, text="📊 Class Probabilities", padding=5)
        prob_frame.pack(fill='both', expand=True)
        
        # Create matplotlib figure for probability plot
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.fig.patch.set_facecolor('white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, prob_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Initial empty plot
        self.clear_probability_plot()
    
    def create_status_bar(self):
        """Create status bar at bottom"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill='x', side='bottom')
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                     style='Info.TLabel')
        self.status_label.pack(side='left', padx=10, pady=5)
        
        # Add class count info
        ttk.Label(status_frame, text=f"Classes: {len(CLASS_NAMES)}", 
                 style='Info.TLabel').pack(side='right', padx=10, pady=5)
    
    def load_model(self):
        """Load model weights from file"""
        file_path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[
                ("PyTorch Models", "*.pth *.pt"),
                ("All Files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.status_var.set("Loading model...")
            self.root.update()
            
            # Create model
            self.model = SwinTransformerClassifier(
                num_classes=len(CLASS_NAMES),
                model_name='swin_base_patch4_window7_224',
                pretrained=False,  # Don't load pretrained when loading checkpoint
                image_size=224
            )
            
            # Load checkpoint using safe loading function
            checkpoint = safe_torch_load(file_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                epoch_info = f" (Epoch {checkpoint.get('epoch', '?')})"
                val_f1 = checkpoint.get('best_val_f1_macro', None)
                if val_f1:
                    epoch_info += f"\nVal F1: {val_f1:.3f}"
            else:
                self.model.load_state_dict(checkpoint)
                epoch_info = ""
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Update UI
            self.model_loaded = True
            model_name = os.path.basename(file_path)
            self.model_info_var.set(f"✅ {model_name}{epoch_info}")
            self.status_var.set("Model loaded successfully")
            
            self.update_inference_button()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.status_var.set("Model loading failed")
            self.model_loaded = False
    
    def load_image(self):
        """Load image from file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("PNG Files", "*.png"),
                ("All Files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Load and display image
            self.current_image = Image.open(file_path).convert('RGB')
            self.current_image_path = file_path
            
            # Display image
            self.display_image()
            
            # Update UI
            image_name = os.path.basename(file_path)
            image_size = self.current_image.size
            self.image_info_var.set(f"✅ {image_name}\nSize: {image_size[0]}x{image_size[1]}")
            self.status_var.set("Image loaded successfully")
            
            self.update_inference_button()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            self.status_var.set("Image loading failed")
    
    def display_image(self):
        """Display loaded image in canvas"""
        if self.current_image is None:
            return
        
        # Clear canvas
        self.image_canvas.delete("all")
        
        # Calculate display size (maintain aspect ratio)
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not yet drawn, use default size
            canvas_width, canvas_height = 400, 400
        
        # Calculate scaling to fit canvas
        img_width, img_height = self.current_image.size
        scale_w = (canvas_width - 20) / img_width
        scale_h = (canvas_height - 20) / img_height
        scale = min(scale_w, scale_h)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        display_image = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(display_image)
        
        # Center image in canvas
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        self.image_canvas.create_image(x, y, anchor='nw', image=self.photo)
    
    def update_inference_button(self):
        """Update inference button state"""
        if self.model_loaded and self.current_image is not None:
            self.inference_button.config(state='normal')
        else:
            self.inference_button.config(state='disabled')
    
    def run_inference(self):
        """Run inference on loaded image"""
        if not self.model_loaded or self.current_image is None:
            return
        
        # Run inference in separate thread to prevent UI freezing
        thread = threading.Thread(target=self._inference_worker)
        thread.daemon = True
        thread.start()
    
    def _inference_worker(self):
        """Worker function for inference (runs in separate thread)"""
        try:
            # Update UI
            self.root.after(0, lambda: self.status_var.set("Running inference..."))
            self.root.after(0, lambda: self.progress_bar.start())
            self.root.after(0, lambda: self.inference_button.config(state='disabled'))
            
            # Preprocess image
            image_array = np.array(self.current_image)
            augmented = self.transform(image=image_array)
            image_tensor = augmented['image'].unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                all_probs = probabilities[0].cpu().numpy()
                
                # Apply threshold tuning if available and enabled
                if self.use_tuned_thresholds and self.threshold_tuner is not None:
                    # Use tuned thresholds
                    tuned_preds = self.threshold_tuner.predict(all_probs.reshape(1, -1), use_tuned=True)
                    predicted_class = tuned_preds[0]
                    confidence = all_probs[predicted_class]
                else:
                    # Standard argmax
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
            
            # Update UI with results
            self.root.after(0, lambda: self._update_results(predicted_class, confidence, all_probs))
            
        except Exception as e:
            error_msg = f"Inference failed: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.root.after(0, lambda: self.status_var.set("Inference failed"))
        finally:
            self.root.after(0, lambda: self.progress_bar.stop())
            self.root.after(0, lambda: self.inference_button.config(state='normal'))
    
    def _update_results(self, predicted_class, confidence, all_probs):
        """Update results display with predictions"""
        # Get class information
        class_name = CLASS_NAMES[predicted_class]
        description = CLASS_DESCRIPTIONS[class_name]
        risk_level = RISK_LEVELS[class_name]
        
        # Update prediction display
        threshold_indicator = " (T)" if self.use_tuned_thresholds else ""
        self.prediction_var.set(f"🎯 {class_name.upper()}{threshold_indicator}")
        self.confidence_var.set(f"Confidence: {confidence:.1%}")
        
        description_text = f"{description}\nRisk Level: {risk_level}"
        if self.use_tuned_thresholds:
            description_text += "\n(Using optimized thresholds)"
        self.description_var.set(description_text)
        
        # Update probability plot
        self.update_probability_plot(all_probs)
        
        # Update status
        threshold_suffix = " (tuned)" if self.use_tuned_thresholds else ""
        self.status_var.set(f"Prediction: {class_name} ({confidence:.1%}){threshold_suffix}")
        
        # Color code prediction label based on risk
        if risk_level == 'Critical':
            self.prediction_label.config(style='Error.TLabel')
        elif risk_level in ['High', 'Moderate']:
            self.prediction_label.config(style='Warning.TLabel')
        else:
            self.prediction_label.config(style='Success.TLabel')
    
    def update_probability_plot(self, probabilities):
        """Update probability distribution plot"""
        self.ax.clear()
        
        # Create horizontal bar plot
        y_pos = np.arange(len(CLASS_NAMES))
        bars = self.ax.barh(y_pos, probabilities, alpha=0.7)
        
        # Color code bars based on risk levels
        for i, (bar, class_name) in enumerate(zip(bars, CLASS_NAMES)):
            risk = RISK_LEVELS[class_name]
            if risk == 'Critical':
                bar.set_color('red')
            elif risk == 'High':
                bar.set_color('orange')
            elif risk == 'Moderate':
                bar.set_color('yellow')
            else:
                bar.set_color('lightblue')
        
        # Customize plot
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels([name.replace('_', ' ').title() for name in CLASS_NAMES])
        self.ax.set_xlabel('Probability')
        self.ax.set_title('Class Probabilities')
        self.ax.set_xlim(0, 1)
        
        # Add probability values on bars
        for i, prob in enumerate(probabilities):
            if prob > 0.01:  # Only show if probability > 1%
                self.ax.text(prob + 0.01, i, f'{prob:.1%}', 
                           va='center', fontsize=8)
        
        # Highlight predicted class
        max_idx = np.argmax(probabilities)
        bars[max_idx].set_alpha(1.0)
        bars[max_idx].set_edgecolor('black')
        bars[max_idx].set_linewidth(2)
        
        self.ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        self.canvas.draw()
    
    def clear_probability_plot(self):
        """Clear probability plot"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'No prediction yet\nRun inference to see results', 
                    transform=self.ax.transAxes, ha='center', va='center',
                    fontsize=12, color='gray')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.canvas.draw()
    
    def clear_results(self):
        """Clear all results"""
        self.prediction_var.set("No prediction yet")
        self.confidence_var.set("")
        self.description_var.set("")
        self.prediction_label.config(style='Info.TLabel')
        self.clear_probability_plot()
        self.status_var.set("Results cleared")
    
    def on_threshold_toggle(self):
        """Handle threshold toggle checkbox"""
        self.use_tuned_thresholds = self.use_thresholds_var.get()
        if self.use_tuned_thresholds and self.threshold_tuner is None:
            messagebox.showwarning("Warning", 
                                 "No thresholds loaded. Please load threshold file first.")
            self.use_thresholds_var.set(False)
            self.use_tuned_thresholds = False
        
        status_suffix = " (tuned)" if self.use_tuned_thresholds else ""
        self.status_var.set(f"Ready for inference{status_suffix}")
    
    def load_thresholds(self):
        """Load optimized thresholds from file"""
        if ThresholdTuner is None:
            messagebox.showerror("Error", "Threshold tuning not available")
            return
        
        file_path = filedialog.askopenfilename(
            title="Load Threshold File",
            filetypes=[
                ("JSON Files", "*.json"),
                ("Pickle Files", "*.pkl"),
                ("All Files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.threshold_tuner = ThresholdTuner(CLASS_NAMES)
            self.threshold_tuner.load(file_path)
            
            messagebox.showinfo("Success", 
                              f"Thresholds loaded successfully!\n"
                              f"F1 score improvement: {self.threshold_tuner.tuning_report.get('f1_macro', 0) - 0.5:.3f}")
            
            # Enable the checkbox
            self.use_thresholds_var.set(True)
            self.use_tuned_thresholds = True
            self.status_var.set("Ready for inference (tuned)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load thresholds:\n{str(e)}")
    
    def save_thresholds(self):
        """Save current thresholds to file"""
        if self.threshold_tuner is None:
            messagebox.showwarning("Warning", "No thresholds to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Threshold File",
            defaultextension=".json",
            filetypes=[
                ("JSON Files", "*.json"),
                ("Pickle Files", "*.pkl"),
                ("All Files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.threshold_tuner.save(file_path)
            messagebox.showinfo("Success", f"Thresholds saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save thresholds:\n{str(e)}")
    
    def update_status(self):
        """Update status based on current state"""
        if not self.model_loaded and self.current_image is None:
            self.status_var.set("Load model and image to start")
        elif not self.model_loaded:
            self.status_var.set("Load model weights")
        elif self.current_image is None:
            self.status_var.set("Load image for classification")
        else:
            threshold_suffix = " (tuned)" if self.use_tuned_thresholds else ""
            self.status_var.set(f"Ready for inference{threshold_suffix}")

def main():
    """Main function to run the GUI"""
    # Check if required packages are available
    try:
        import torch
        import timm
        import albumentations
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Install requirements with: pip install -r requirements.txt")
        return
    
    # Create and run GUI
    root = tk.Tk()
    app = SkinDiseaseGUI(root)
    
    # Handle window closing
    def on_closing():
        root.quit()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nGUI interrupted by user")
    except Exception as e:
        print(f"GUI error: {e}")

if __name__ == "__main__":
    main()
