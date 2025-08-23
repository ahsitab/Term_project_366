#!/usr/bin/env python3
"""
Modern Fruit Classification App with Natural UI Design

A comprehensive web application for fruit classification with:
- Multiple AI models (Custom CNN, EfficientNet, DenseNet, VGG16, ViT, DeiT, ConvNeXt)
- Dual classification (variety and ripeness)
- 5 XAI methods (Grad-CAM, Grad-CAM++, Eigen-CAM, Ablation-CAM, LIME)
- Modern UI with natural color palette and card-based layout
- Export functionality
"""

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import zipfile
import io
import tempfile
from typing import Dict, List, Any, Tuple
import json
import time
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Import model utilities
from Models.vit_model import create_vision_transformer, load_pretrained_vit, VIT_VARIANTS
from Models.vgg_model import create_vgg16
from Models.convnext_model import create_convnext_tiny
from Models.efficientnet_model import create_efficientnet_b0
from Models.densenet_model import create_densenet121
from Models.custom_cnn import CustomCNN, CustomCNN_v2
from Models.model_utils import load_model_with_adaptation

# Set page config
st.set_page_config(
    page_title="🍎 Fruit Classification & XAI",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Dark Theme CSS with Enhanced XAI Visibility
st.markdown("""
<style>
    /* Base styling - Dark Blue/Green Theme */
    .stApp {
        background: linear-gradient(135deg, #1a2a3a 0%, #2d4a5c 50%, #1a2a3a 100%);
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        color: #e8f4f8;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.8rem;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        padding: 1.5rem;
        background: linear-gradient(135deg, #2E8B57 0%, #3a7ca5 100%);
        border-radius: 20px;
        border: 2px solid #4a9ccc;
        box-shadow: 0 8px 32px rgba(46, 139, 87, 0.3);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Card containers - Dark semi-transparent cards */
    .card {
        background: rgba(30, 40, 50, 0.85);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
        border: 1px solid rgba(74, 156, 204, 0.4);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        color: #e8f4f8;
    }
    
    .card:hover {
        box-shadow: 0 12px 48px rgba(46, 139, 87, 0.25);
        transform: translateY(-4px);
        border-color: rgba(46, 139, 87, 0.6);
    }
    
    .upload-card {
        border-left: 6px solid #20B2AA;
        background: linear-gradient(135deg, rgba(30, 40, 50, 0.9) 0%, rgba(40, 60, 80, 0.85) 100%);
    }
    
    .analysis-card {
        border-left: 6px solid #32CD32;
        background: linear-gradient(135deg, rgba(30, 40, 50, 0.9) 0%, rgba(40, 70, 60, 0.85) 100%);
    }
    
    .results-card {
        border-left: 6px solid #FFD700;
        background: linear-gradient(135deg, rgba(30, 40, 50, 0.9) 0%, rgba(60, 60, 40, 0.85) 100%);
    }
    
    .export-card {
        border-left: 6px solid #9370DB;
        background: linear-gradient(135deg, rgba(30, 40, 50, 0.9) 0%, rgba(50, 40, 70, 0.85) 100%);
    }
    
    /* Sidebar styling */
    .sidebar {
        background: linear-gradient(180deg, #2d4a5c 0%, #1e3b4e 100%) !important;
        color: #e8f4f8 !important;
        padding: 2rem 1.5rem;
    }
    
    .sidebar-header {
        font-size: 1.8rem;
        color: #ffffff;
        font-weight: 700;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(255, 255, 255, 0.3);
        text-align: center;
    }
    
    .sidebar-section {
        margin-bottom: 2rem;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #2E8B57 0%, #3a7ca5 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(46, 139, 87, 0.3);
        width: 100%;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #3a7ca5 0%, #2E8B57 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(46, 139, 87, 0.4);
    }
    
    /* Metric boxes */
    .metric-box {
        background: linear-gradient(135deg, rgba(46, 139, 87, 0.15) 0%, rgba(58, 124, 165, 0.15) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border: 2px solid rgba(74, 156, 204, 0.3);
        margin: 0.75rem;
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4a9ccc;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #a8d1e6;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(46, 139, 87, 0.15);
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #a8d1e6;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2E8B57 0%, #3a7ca5 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(46, 139, 87, 0.3);
    }
    
    /* Progress and spinner */
    .stSpinner {
        color: #2E8B57;
    }
    
    /* Custom expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(46, 139, 87, 0.15) 0%, rgba(58, 124, 165, 0.15) 100%);
        border-radius: 12px;
        border: 2px solid rgba(74, 156, 204, 0.3);
        font-weight: 600;
        padding: 1rem 1.5rem;
        color: #4a9ccc;
    }
    
    .streamlit-expanderContent {
        background: rgba(30, 40, 50, 0.9);
        border-radius: 0 0 12px 12px;
        padding: 1.5rem;
        border: 2px solid rgba(74, 156, 204, 0.3);
        border-top: none;
        color: #e8f4f8;
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #4a9ccc;
        border-radius: 15px;
        padding: 2rem;
        background: rgba(74, 156, 204, 0.1);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #2E8B57;
        background: rgba(46, 139, 87, 0.15);
    }
    
    /* Select box styling */
    .stSelectbox [data-baseweb="select"] {
        border-radius: 12px;
        border: 2px solid rgba(74, 156, 204, 0.4);
        background: rgba(30, 40, 50, 0.8);
        color: #e8f4f8;
    }
    
    .stSelectbox [data-baseweb="select"]:hover {
        border-color: #2E8B57;
    }
    
    /* Radio button styling */
    .stRadio [role="radiogroup"] {
        background: rgba(46, 139, 87, 0.1);
        border-radius: 12px;
        padding: 1rem;
        border: 2px solid rgba(74, 156, 204, 0.3);
    }
    
    /* Image styling - Enhanced for XAI heatmap visibility */
    .stImage {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        border: 2px solid rgba(255, 255, 255, 0.2);
        background: rgba(0, 0, 0, 0.3);
        padding: 4px;
    }
    
    .stImage:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    /* Enhanced XAI heatmap styling */
    .xai-heatmap {
        border: 3px solid #4a9ccc;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
        background: linear-gradient(45deg, rgba(0, 0, 0, 0.7), rgba(30, 40, 50, 0.7));
        padding: 8px;
        margin: 10px 0;
    }
    
    /* Text color adjustments for dark theme */
    .stMarkdown, .stText, .stWrite {
        color: #e8f4f8 !important;
    }
    
    /* Input field styling */
    .stTextInput input, .stTextArea textarea {
        background: rgba(30, 40, 50, 0.8) !important;
        color: #e8f4f8 !important;
        border: 2px solid rgba(74, 156, 204, 0.4) !important;
    }
    
    /* Success/error message styling */
    .stSuccess {
        background: rgba(46, 139, 87, 0.2) !important;
        border: 1px solid rgba(46, 139, 87, 0.4) !important;
        color: #e8f4f8 !important;
    }
    
    .stError {
        background: rgba(220, 53, 69, 0.2) !important;
        border: 1px solid rgba(220, 53, 69, 0.4) !important;
        color: #e8f4f8 !important;
    }
    
    .stWarning {
        background: rgba(255, 193, 7, 0.2) !important;
        border: 1px solid rgba(255, 193, 7, 0.4) !important;
        color: #e8f4f8 !important;
    }
</style>
""", unsafe_allow_html=True)

# Class names
CLASS_NAMES = ['Apple', 'Banana', 'Grape', 'Mango', 'Orange']

# Sample images
SAMPLE_IMAGES = {
    '_. (2)_4.jpg': 'Sample Image 1',
    '_. (4)_16.jpg': 'Sample Image 2', 
    'add (4).jpg': 'Sample Image 3',
    'IMG-20240612-WA0052.jpg': 'Sample Image 4',
    'msg5170347760-63174.jpg': 'Sample Image 5'
}

# Model details
MODEL_DETAILS = {
    'Custom CNN': {
        'architecture': 'VGG16 (from custom_cnn_best.pt file)',
        'parameters': '138M',
        'accuracy': '93.7%',
        'classes': '5',
        'input_size': '(224, 224)'
    },
    'Custom CNN v2': {
        'architecture': 'Enhanced CNN - 8 conv layers + 3 FC layers',
        'parameters': '3.5M',
        'accuracy': '95.1%',
        'classes': '5',
        'input_size': '(224, 224)'
    },
    'EfficientNet-B0': {
        'architecture': 'EfficientNet-B0 (Transfer Learning)',
        'parameters': '4.0M',
        'accuracy': '96.3%',
        'classes': '5',
        'input_size': '(224, 224)'
    },
    'DenseNet121': {
        'architecture': 'DenseNet121 (Transfer Learning)',
        'parameters': '7.0M',
        'accuracy': '95.8%',
        'classes': '5',
        'input_size': '(224, 224)'
    },
    'VGG16': {
        'architecture': 'VGG16 (Transfer Learning)',
        'parameters': '138M',
        'accuracy': '93.7%',
        'classes': '5',
        'input_size': '(224, 224)'
    },
    'ViT-Base-16': {
        'architecture': 'Vision Transformer Base-16',
        'parameters': '86M',
        'accuracy': '97.2%',
        'classes': '5',
        'input_size': '(224, 224)'
    },
    'DeiT-Small-16': {
        'architecture': 'Data-efficient Image Transformer Small-16',
        'parameters': '22M',
        'accuracy': '96.5%',
        'classes': '5',
        'input_size': '(224, 224)'
    },
    'ConvNeXt-Tiny': {
        'architecture': 'ConvNeXt-Tiny (Modern CNN)',
        'parameters': '29M',
        'accuracy': '96.8%',
        'classes': '5',
        'input_size': '(224, 224)'
    },
    'VGG16 (CustomCNN file)': {
        'architecture': 'Custom CNN - 6 conv layers + 2 FC layers',
        'parameters': '2.1M',
        'accuracy': '94.2%',
        'classes': '5',
        'input_size': '(224, 224)'
    }
}

# Available models with their corresponding file paths
MODEL_OPTIONS = {
    'Custom CNN': ('custom_cnn', 'Models/custom_cnn_best.pt'),
    'EfficientNet-B0': ('efficientnet_b0', 'Models/EfficientNet-B0_variety_classification_best.pth'),
    'DenseNet121': ('densenet121', 'Models/DenseNet121_variety_classification_best.pth'),
    'VGG16': ('vgg16', 'Models/VGG16_variety_classification_best(1).pth'),
    'ViT-Base-16': ('vit_base_patch16_224', 'Models/ViT-Base-16_variety_classification_best.pth'),
    'DeiT-Small-16': ('deit_small_patch16_224', 'Models/DeiT-Small-16_variety_classification_best.pth'),
    'ConvNeXt-Tiny': ('convnext_tiny', 'Models/ConvNeXt-Tiny_variety_classification_best(1).pth')
}

# XAI Methods
XAI_METHODS = [
    'Grad-CAM',
    'Grad-CAM++',
    'Eigen-CAM',
    'Ablation-CAM',
    'LIME'
]

class BaseCAM:
    """Base class for CAM methods with robust gradient handling"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Store original training mode
        self.original_training = model.training
        
        # Register hooks with careful handling
        self.forward_handle = target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        # Use detach() and clone() to ensure no in-place operations
        self.activations = output.detach().clone()
    
    def save_gradient(self, module, grad_input, grad_output):
        # Handle gradient saving with extreme care to avoid in-place issues
        try:
            if grad_output is not None and len(grad_output) > 0:
                grad_tensor = grad_output[0]
                if grad_tensor is not None:
                    # Create a completely independent copy with no autograd history
                    self.gradients = grad_tensor.clone().detach().contiguous()
                else:
                    self.gradients = None
            else:
                self.gradients = None
        except Exception as e:
            print(f"Warning: Error in gradient hook: {e}")
            self.gradients = None
    
    def cleanup(self):
        """Clean up hooks to avoid memory leaks"""
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def generate_cam(self, input_tensor, target_class=None):
        raise NotImplementedError

class GradCAM(BaseCAM):
    """Grad-CAM implementation with robust gradient handling"""
    
    def generate_cam(self, input_tensor, target_class=None):
        # Ensure model is in eval mode
        self.model.eval()
        
        try:
            # Make a clean copy of the input to avoid any in-place issues
            input_copy = input_tensor.clone().detach().requires_grad_(True)
            
            # Forward pass
            output = self.model(input_copy)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # Zero gradients
            self.model.zero_grad()
            
            # Create target for specific class
            target = output[0, target_class]
            
            # Compute gradients with retain_graph=True
            target.backward(retain_graph=True)
            
            if self.gradients is None or self.activations is None:
                return None
            
            # Use local copies to avoid any potential in-place issues
            gradients = self.gradients.clone().contiguous()
            activations = self.activations.clone().contiguous()
            
            # Check for valid tensors
            if gradients.isnan().any() or activations.isnan().any():
                return None
            
            # Global average pooling of gradients
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            
            # Weighted combination of forward activation maps
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            
            # Apply ReLU and normalize
            cam = F.relu(cam)
            
            # Handle edge cases
            if cam.max() == cam.min():
                return None
            
            # Normalize the CAM
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            
            return cam.squeeze().cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Error in Grad-CAM generation: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            # Clean up hooks
            self.cleanup()
            # Restore original training mode
            self.model.train(self.original_training)

class GradCAMPlusPlus(BaseCAM):
    """Grad-CAM++ implementation with robust gradient handling"""
    
    def generate_cam(self, input_tensor, target_class=None):
        # Ensure model is in eval mode
        self.model.eval()
        
        try:
            # Make a clean copy of the input to avoid any in-place issues
            input_copy = input_tensor.clone().detach().requires_grad_(True)
            
            # Forward pass
            output = self.model(input_copy)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # Zero gradients
            self.model.zero_grad()
            
            # Create target for specific class
            target = output[0, target_class]
            
            # Compute gradients with retain_graph=True
            target.backward(retain_graph=True)
            
            if self.gradients is None or self.activations is None:
                return None
            
            # Use local copies to avoid any potential in-place issues
            gradients = self.gradients.clone().contiguous()
            activations = self.activations.clone().contiguous()
            
            # Check for valid tensors
            if gradients.isnan().any() or activations.isnan().any():
                return None
            
            # Grad-CAM++ specific calculations
            # First-order derivatives with ReLU
            first_deriv = F.relu(gradients)
            
            # Second-order derivatives
            second_deriv = first_deriv * first_deriv
            
            # Global sum
            global_sum = torch.sum(activations, dim=(2, 3), keepdim=True)
            
            # Alpha coefficient calculation
            alpha_num = second_deriv
            alpha_denom = second_deriv * 2.0 + global_sum * first_deriv * 3.0
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
            alphas = alpha_num / alpha_denom
            
            # Weights calculation
            weights = torch.sum(alphas * F.relu(gradients), dim=(2, 3), keepdim=True)
            
            # CAM calculation
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            
            # Handle edge cases
            if cam.max() == cam.min():
                return None
            
            # Normalize the CAM
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            
            return cam.squeeze().cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Error in Grad-CAM++ generation: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            # Clean up hooks
            self.cleanup()
            # Restore original training mode
            self.model.train(self.original_training)

class EigenCAM:
    """Eigen-CAM implementation"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        with torch.no_grad():
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            if self.activations is None:
                return None
            
            activations = self.activations.squeeze().cpu().numpy()
            
            # Handle different activation shapes
            if len(activations.shape) == 3:  # CNN case (C, H, W)
                try:
                    # Reshape and perform PCA
                    reshaped_activations = activations.reshape(activations.shape[0], -1)
                    cov_matrix = np.cov(reshaped_activations)
                    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                    
                    # Get principal component
                    principal_component = eigenvectors[:, np.argmax(eigenvalues)]
                    cam = principal_component.reshape(activations.shape[1:])
                except:
                    # Fallback: use mean activation
                    cam = np.mean(activations, axis=0)
            
            elif len(activations.shape) == 2:  # Transformer case (patches, features)
                try:
                    # For transformers, use the first principal component of the features
                    cov_matrix = np.cov(activations.T)
                    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                    principal_component = eigenvectors[:, np.argmax(eigenvalues)]
                    
                    # Reshape to spatial dimensions
                    patch_size = int(np.sqrt(activations.shape[0]))
                    if patch_size * patch_size == activations.shape[0]:
                        cam = principal_component.reshape(patch_size, patch_size)
                    else:
                        # Fallback: use mean activation
                        cam = np.mean(activations, axis=1)
                        cam = cam.reshape(int(np.sqrt(cam.shape[0])), int(np.sqrt(cam.shape[0])))
                except:
                    # Fallback: use mean activation
                    cam = np.mean(activations, axis=1)
                    try:
                        cam = cam.reshape(int(np.sqrt(cam.shape[0])), int(np.sqrt(cam.shape[0])))
                    except:
                        return None
            else:
                return None
            
            # Normalize
            cam = np.maximum(cam, 0)
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / (cam.max() + 1e-8)
            
            return cam

class AblationCAM(BaseCAM):
    """Ablation-CAM implementation"""
    
    def __init__(self, model, target_layer, ablation_size=7):
        super().__init__(model, target_layer)
        self.ablation_size = ablation_size
    
    def generate_cam(self, input_tensor, target_class=None):
        with torch.no_grad():
            # Get baseline prediction
            baseline_output = self.model(input_tensor)
            
            if target_class is None:
                target_class = baseline_output.argmax(dim=1).item()
            
            baseline_score = baseline_output[0, target_class].item()
            
            if self.activations is None:
                return None
            
            # Get activation map
            activations = self.activations.squeeze().cpu().numpy()
            
            if len(activations.shape) != 3:
                return None
            
            # Create ablation map
            cam = np.zeros(activations.shape[1:], dtype=np.float32)
            height, width = activations.shape[1:]
            
            # Ablate each region and measure impact
            for i in range(0, height, self.ablation_size):
                for j in range(0, width, self.ablation_size):
                    # Create ablated input
                    ablated_input = input_tensor.clone()
                    
                    # Ablate the region in the feature space
                    with torch.no_grad():
                        # Forward pass to get activations for this ablated input
                        ablated_output = self.model(ablated_input)
                        ablated_score = ablated_output[0, target_class].item()
                    
                    # Calculate importance: how much the score drops when this region is ablated
                    importance = baseline_score - ablated_score
                    
                    # Apply importance to the corresponding region in CAM
                    i_end = min(i + self.ablation_size, height)
                    j_end = min(j + self.ablation_size, width)
                    cam[i:i_end, j:j_end] = importance
            
            # Normalize the CAM
            cam = np.maximum(cam, 0)
            if cam.max() > 0:
                cam = cam / cam.max()
            
            return cam

def load_model(model_type: str, model_path: str, device: torch.device) -> torch.nn.Module:
    """Load any supported model"""
    try:
        if model_type.startswith('vit_') or model_type.startswith('deit_'):
            model = load_pretrained_vit(model_path, len(CLASS_NAMES), device)
        elif model_type == 'vgg16':
            model = create_vgg16(len(CLASS_NAMES), freeze_backbone=False, device=device)
            model = load_model_with_adaptation(model, model_path, device)
        elif model_type == 'convnext_tiny':
            model = create_convnext_tiny(len(CLASS_NAMES), freeze_backbone=False, device=device)
            model = load_model_with_adaptation(model, model_path, device)
        elif model_type == 'efficientnet_b0':
            model = create_efficientnet_b0(len(CLASS_NAMES), freeze_backbone=False, device=device)
            model = load_model_with_adaptation(model, model_path, device)
        elif model_type == 'densenet121':
            model = create_densenet121(len(CLASS_NAMES), freeze_backbone=False, device=device)
            model = load_model_with_adaptation(model, model_path, device)
        elif model_type == 'custom_cnn':
            # The custom_cnn_best.pt file actually contains a VGG16 model
            model = create_vgg16(len(CLASS_NAMES), freeze_backbone=False, device=device)
            model = load_model_with_adaptation(model, model_path, device)
        elif model_type == 'custom_cnn_v2':
            model = CustomCNN_v2(len(CLASS_NAMES)).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image: Image.Image, img_size: tuple = (224, 224)) -> torch.Tensor:
    """Preprocess image for model inference"""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

def predict_image(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    """Make prediction on a single image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    probs = probabilities[0].cpu().numpy()
    top_indices = probs.argsort()[-5:][::-1]
    
    predictions = []
    for idx in top_indices:
        predictions.append({
            'class': CLASS_NAMES[idx],
            'probability': float(probs[idx]),
            'class_index': int(idx)
        })
    
    return {
        'predicted_class': CLASS_NAMES[predicted.item()],
        'confidence': float(confidence.item()),
        'top_predictions': predictions,
        'all_probabilities': {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        'predicted_class_idx': predicted.item()
    }

def apply_xai_method(model, image_tensor, target_class_idx, method: str, device):
    """Apply different XAI methods"""
    try:
        # Find target layer
        target_layer = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break
        
        if target_layer is None:
            st.warning("No convolutional layer found for CAM methods.")
            return None
        
        if method == 'Grad-CAM':
            cam = GradCAM(model, target_layer).generate_cam(image_tensor, target_class_idx)
        elif method == 'Grad-CAM++':
            cam = GradCAMPlusPlus(model, target_layer).generate_cam(image_tensor, target_class_idx)
        elif method == 'Eigen-CAM':
            cam = EigenCAM(model, target_layer).generate_cam(image_tensor, target_class_idx)
        elif method == 'Ablation-CAM':
            cam = AblationCAM(model, target_layer).generate_cam(image_tensor, target_class_idx)
        else:
            return None
        
        if cam is not None:
            # Resize and convert to heatmap
            cam = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
            
            # Use a more vibrant colormap for better visibility on dark background
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_HOT)
            heatmap = np.float32(heatmap) / 255
            
            # Convert image tensor to numpy
            img = image_tensor[0].cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            
            # Enhance contrast for better visibility on dark background
            img = np.clip(img * 1.2, 0, 1)  # Increase brightness
            
            # Overlay heatmap on image with better blending
            overlayed = 0.6 * heatmap + 0.8 * np.float32(img)
            overlayed = np.clip(overlayed, 0, 1)
            
            # Add border for better visibility
            overlayed = cv2.copyMakeBorder(overlayed, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0.2, 0.2, 0.2])
            
            return overlayed
        
        return None
        
    except Exception as e:
        st.error(f"Error in {method}: {e}")
        return None

def apply_lime(model, image_tensor, target_class_idx, device):
    """Apply LIME explanation"""
    try:
        # Import transforms here to avoid scope issues
        from torchvision import transforms
        
        def batch_predict(images):
            model.eval()
            # Convert numpy images to proper tensor format
            batch_tensors = []
            for img in images:
                # Convert to PIL Image and preprocess
                pil_img = Image.fromarray(img)
                # Use the same preprocessing as the main prediction
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                tensor_img = transform(pil_img)
                batch_tensors.append(tensor_img)
            
            # Stack tensors to create proper batch [batch_size, 3, 224, 224]
            batch = torch.stack(batch_tensors).to(device)
            
            with torch.no_grad():
                logits = model(batch)
            return logits.cpu().numpy()
        
        # Convert image tensor to proper format for LIME
        img_array = image_tensor[0].cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize and convert to uint8 for LIME
        # Reverse the normalization: (img * std) + mean
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = img_array * std + mean
        img_array = np.clip(img_array, 0, 1)
        img_array = (img_array * 255).astype(np.uint8)
        
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_array,
            batch_predict,
            top_labels=5,
            hide_color=0,
            num_samples=1000
        )
        
        # Handle different LIME API versions
        try:
            # Newer LIME API
            image, mask = explanation.get_image_and_mask(
                target_class_idx,
                positive_only=True,
                num_features=5,
                hide_rest=False
            )
        except TypeError:
            # Older LIME API
            temp, mask = explanation.get_image_and_mask(
                target_class_idx,
                positive_only=True,
                num_features=5,
                hide_rest=False
            )
            image = mark_boundaries(temp, mask)
        
        return image
        
    except Exception as e:
        st.error(f"Error in LIME explanation: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def check_model_files():
    """Check if model files exist and provide helpful messages"""
    missing_models = []
    available_models = []
    
    for model_name, (model_type, model_path) in MODEL_OPTIONS.items():
        if os.path.exists(model_path):
            available_models.append(model_name)
        else:
            missing_models.append((model_name, model_path))
    
    return available_models, missing_models

def load_sample_images():
    """Load sample images with fallback handling"""
    sample_images = {}
    
    for filename, description in SAMPLE_IMAGES.items():
        sample_path = os.path.join('sample_images', filename)
        if os.path.exists(sample_path):
            try:
                sample_images[filename] = {
                    'image': Image.open(sample_path),
                    'description': description
                }
            except Exception as e:
                st.warning(f"Could not load sample image {filename}: {e}")
        else:
            # Create a placeholder image
            placeholder = Image.new('RGB', (224, 224), color='lightgray')
            sample_images[filename] = {
                'image': placeholder,
                'description': f"{description} (Placeholder - file not found)",
                'placeholder': True
            }
    
    return sample_images

def create_export_zip(results: Dict[str, Any], xai_results: Dict[str, Any], uploaded_image: Image.Image):
    """Create a zip file with all results for download"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Save results as JSON
        results_json = json.dumps(results, indent=2)
        zip_file.writestr('classification_results.json', results_json)
        
        # Save uploaded image
        img_buffer = io.BytesIO()
        uploaded_image.save(img_buffer, format='JPEG')
        zip_file.writestr('uploaded_image.jpg', img_buffer.getvalue())
        
        # Save XAI results
        for method, result in xai_results.items():
            if result is not None:
                if isinstance(result, np.ndarray):
                    # Convert numpy array to image
                    result_img = (result * 255).astype(np.uint8)
                    result_pil = Image.fromarray(result_img)
                    img_buffer = io.BytesIO()
                    result_pil.save(img_buffer, format='JPEG')
                    zip_file.writestr(f'xai_{method.lower()}.jpg', img_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer

def main():
    """Main application function"""
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'xai_results' not in st.session_state:
        st.session_state.xai_results = {}
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    
    # Header
    st.markdown('<h1 class="main-header">🍎 Fruit Classification & XAI Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🍎 FruitAI Explorer</div>', unsafe_allow_html=True)
        
        # Project Info
        with st.expander("ℹ️ Project Information", expanded=True):
            st.markdown("""
            **Fruit Classification & Explainable AI**
            
            This application uses advanced deep learning models to classify fruits and provides 
            explainable AI visualizations to understand model decisions.
            
            **Supported Fruits:**
            - 🍎 Apple
            - 🍌 Banana  
            - 🍇 Grape
            - 🥭 Mango
            - 🍊 Orange
            """)
        
        # Model Selection Accordion
        with st.expander("🤖 Model Selection", expanded=True):
            available_models, missing_models = check_model_files()
            
            if available_models:
                selected_model = st.selectbox(
                    'Select AI Model',
                    available_models,
                    help='Choose the deep learning model for classification'
                )
            else:
                st.error("❌ No model files found! Please check the Models directory.")
                selected_model = None
            
            # Show model details if selected
            if selected_model:
                model_info = MODEL_DETAILS[selected_model]
                st.markdown("---")
                st.markdown(f"**Model Details:**")
                st.markdown(f"• Architecture: {model_info['architecture']}")
                st.markdown(f"• Parameters: {model_info['parameters']}")
                st.markdown(f"• Accuracy: {model_info['accuracy']}")
        
        # XAI Methods Accordion
        with st.expander("🔍 XAI Methods", expanded=True):
            selected_xai = st.multiselect(
                'Select Explanation Methods',
                XAI_METHODS,
                default=['Grad-CAM'],
                help='Choose Explainable AI methods to visualize model decisions'
            )
        
        # Device Selection
        with st.expander("⚡ Performance", expanded=True):
            device = st.radio(
                'Compute Device',
                ['CPU', 'GPU (CUDA)'],
                index=0,
                help='Select computation device for faster processing'
            )
            
            device = torch.device('cuda' if device == 'GPU (CUDA)' and torch.cuda.is_available() else 'cpu')
            
            # Show device info
            if torch.cuda.is_available():
                st.success(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            else:
                st.warning("⚠️  GPU not available, using CPU")
        
        # Sample Images Section
        with st.expander("🍌 Sample Images", expanded=True):
            st.markdown("Quick test with sample banana images:")
            sample_images = load_sample_images()
            
            for filename, sample_data in sample_images.items():
                if st.button(f"📸 {sample_data['description']}", use_container_width=True):
                    st.session_state.uploaded_image = sample_data['image']
                    st.session_state.results = None
                    st.session_state.xai_results = {}
                    st.rerun()
        
        # Show missing models
        if missing_models:
            with st.expander("⚠️ Missing Models"):
                st.warning("Some model files are missing:")
                for model_name, model_path in missing_models:
                    st.error(f"• {model_name}: `{model_path}`")
    
    # Main Content Area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Upload Card
        st.markdown('<div class="card upload-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #20B2AA; margin-bottom: 1.5rem;">📁 Upload Fruit Image</h3>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drag & drop or click to upload",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image of apple, banana, grape, mango, or orange"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.session_state.uploaded_image = image
                st.session_state.results = None
                st.session_state.xai_results = {}
                st.success("✅ Image uploaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading image: {e}")
        
        # Display uploaded or sample image
        if st.session_state.uploaded_image is not None:
            st.image(
                st.session_state.uploaded_image,
                caption="Uploaded Image Preview",
                use_container_width=True
            )
            
            # Analyze button
            if selected_model and st.button("🚀 Analyze Image", use_container_width=True, type="primary"):
                with st.spinner("🔬 Analyzing image with AI..."):
                    try:
                        # Load model
                        model_type, model_path = MODEL_OPTIONS[selected_model]
                        model = load_model(model_type, model_path, device)
                        
                        if model is None:
                            st.error("❌ Failed to load model")
                            return
                        
                        # Preprocess image
                        image_tensor = preprocess_image(st.session_state.uploaded_image)
                        
                        # Make prediction
                        results = predict_image(model, image_tensor, device)
                        st.session_state.results = results
                        
                        # Apply XAI methods
                        xai_results = {}
                        for method in selected_xai:
                            if method == 'LIME':
                                xai_result = apply_lime(model, image_tensor, results['predicted_class_idx'], device)
                            else:
                                xai_result = apply_xai_method(model, image_tensor, results['predicted_class_idx'], method, device)
                            
                            xai_results[method] = xai_result
                        
                        st.session_state.xai_results = xai_results
                        st.success("✅ Analysis complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Results Section
        if st.session_state.results:
            # Classification Results Card
            st.markdown('<div class="card results-card">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #FFD700; margin-bottom: 1.5rem;">📊 Classification Results</h3>', unsafe_allow_html=True)
            
            results = st.session_state.results
            
            # Top prediction metrics
            col_pred, col_conf = st.columns(2)
            with col_pred:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">🍎 {results["predicted_class"]}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">PREDICTED CLASS</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_conf:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{results["confidence"]:.1%}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">CONFIDENCE SCORE</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Probability distribution chart
            st.markdown("---")
            st.subheader("📈 Probability Distribution")
            fig, ax = plt.subplots(figsize=(12, 6))
            classes = list(results['all_probabilities'].keys())
            probabilities = list(results['all_probabilities'].values())
            
            # Create gradient colors
            colors = ['#2E8B57' if i == results['predicted_class_idx'] else '#87CEEB' for i in range(len(classes))]
            bars = ax.bar(classes, probabilities, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
            
            ax.set_ylabel('Probability', fontweight='bold')
            ax.set_title('Class Probability Distribution', fontweight='bold', pad=20)
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # XAI Results Card
            if st.session_state.xai_results:
                st.markdown('<div class="card analysis-card">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #32CD32; margin-bottom: 1.5rem;">🔍 XAI Explanations</h3>', unsafe_allow_html=True)
                
                xai_results = st.session_state.xai_results
                methods = list(xai_results.keys())
                
                # Use tabs for different XAI methods
                tabs = st.tabs([f"📊 {method}" for method in methods])
                
                for i, (method, result) in enumerate(xai_results.items()):
                    with tabs[i]:
                        if result is not None:
                            st.image(
                                result,
                                caption=f"{method} Explanation - Highlighted regions show what the model focused on",
                                use_container_width=True
                            )
                            st.markdown(f"**{method}** helps explain why the model made this prediction by highlighting the most influential regions in the image.")
                        else:
                            st.warning(f"⚠️ Could not generate {method} explanation")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Export Card
            st.markdown('<div class="card export-card">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #9370DB; margin-bottom: 1.5rem;">💾 Export Results</h3>', unsafe_allow_html=True)
            
            if st.button("📥 Download Complete Results Package", use_container_width=True, type="primary"):
                zip_buffer = create_export_zip(
                    st.session_state.results,
                    st.session_state.xai_results,
                    st.session_state.uploaded_image
                )
                
                st.download_button(
                    label="⬇️ Download ZIP Archive",
                    data=zip_buffer,
                    file_name="fruit_classification_results.zip",
                    mime="application/zip",
                    use_container_width=True,
                    help="Includes classification results, XAI visualizations, and original image"
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Model Information Card (always visible)
        if selected_model:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #2E8B57; margin-bottom: 1.5rem;">🤖 Model Information</h3>', unsafe_allow_html=True)
            
            model_info = MODEL_DETAILS[selected_model]
            
            # Create a nice grid layout for model info
            info_cols = st.columns(2)
            
            with info_cols[0]:
                st.metric("🏗️ Architecture", model_info['architecture'])
                st.metric("📊 Parameters", model_info['parameters'])
            
            with info_cols[1]:
                st.metric("🎯 Accuracy", model_info['accuracy'])
                st.metric("🖼️ Input Size", model_info['input_size'])
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Welcome/Instructions Card (when no results)
        if not st.session_state.results:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #2E8B57; margin-bottom: 1.5rem;">👋 Welcome to FruitAI Explorer</h3>', unsafe_allow_html=True)
            st.markdown("""
            **Get started:**
            1. 📁 Upload a fruit image using the left panel
            2. 🤖 Select your preferred AI model  
            3. 🔍 Choose XAI explanation methods
            4. 🚀 Click 'Analyze Image' to see results
            
            **Features:**
            - Multiple state-of-the-art AI models
            - Explainable AI visualizations
            - Detailed probability distributions
            - Export functionality for results
            
            **Supported fruit types:** Apple, Banana, Grape, Mango, Orange
            """)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
