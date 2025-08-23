#!/usr/bin/env python3
"""
Unified Training Script for Fruit Classification Models

This script provides a unified interface to train various deep learning models
for fruit variety classification including:
- Vision Transformers (ViT, DeiT)
- CNN models (VGG16, ConvNeXt, EfficientNet, DenseNet)
- Custom CNN architectures
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
import argparse
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Import model factories
from Models.vit_model import create_vision_transformer, VIT_VARIANTS
from Models.vgg_model import create_vgg16
from Models.convnext_model import create_convnext_tiny
from Models.efficientnet_model import create_efficientnet_b0
from Models.densenet_model import create_densenet121
from Models.custom_cnn import CustomCNN, CustomCNN_v2

# Import data utilities
from data_utils import create_dataloaders, setup_gpu, clear_gpu_memory, print_gpu_memory

def create_model(model_type: str, num_classes: int, freeze_backbone: bool = True, device: torch.device = None) -> nn.Module:
    """
    Create model based on type
    
    Args:
        model_type: Type of model to create
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze backbone weights
        device: Device to load model on
    
    Returns:
        Created model instance
    """
    if model_type.startswith('vit_') or model_type.startswith('deit_'):
        # Vision Transformer variants
        if model_type not in VIT_VARIANTS:
            raise ValueError(f"Unknown Vision Transformer variant: {model_type}")
        
        return create_vision_transformer(
            model_name=model_type,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            device=device
        )
    
    elif model_type == 'vgg16':
        return create_vgg16(num_classes, freeze_backbone, device)
    
    elif model_type == 'convnext_tiny':
        return create_convnext_tiny(num_classes, freeze_backbone, device)
    
    elif model_type == 'efficientnet_b0':
        return create_efficientnet_b0(num_classes, freeze_backbone, device)
    
    elif model_type == 'densenet121':
        return create_densenet121(num_classes, freeze_backbone, device)
    
    elif model_type == 'custom_cnn':
        return CustomCNN(num_classes).to(device)
    
    elif model_type == 'custom_cnn_v2':
        return CustomCNN_v2(num_classes).to(device)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    task_type: str,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Train a model with unified training procedure
    
    Args:
        model: Model to train
        model_name: Name of the model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Training device
        task_type: Type of task (e.g., "Variety Classification")
        output_dir: Output directory for saving results
    
    Returns:
        Training history and results
    """
    print(f"\n--- Training {model_name} for {task_type} ---")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, 
        weight_decay=1e-4
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=1e-7)
    scaler = GradScaler()
    
    # Early stopping setup
    early_stopping_patience = 7
    min_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [],
        'learning_rates': []
    }
    
    clear_gpu_memory()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Metrics calculation
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Memory management
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / val_total
        avg_train_acc = train_correct / train_total
        avg_val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Acc: {avg_train_acc:.4f} | Val Acc: {avg_val_acc:.4f} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.6f}")
        
        scheduler.step()
        
        # Early stopping check
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            
            # Save best model
            model_filename = f"{model_name}_{task_type.lower().replace(' ', '_')}_best.pth"
            model_path = os.path.join(output_dir, model_filename)
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved best model: {model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Memory monitoring
        if epoch % 5 == 0:
            print_gpu_memory()
    
    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    clear_gpu_memory()
    return history

def main():
    parser = argparse.ArgumentParser(description='Train fruit classification models')
    parser.add_argument('--model', type=str, required=True, 
                       choices=list(VIT_VARIANTS.keys()) + [
                           'vgg16', 'convnext_tiny', 'efficientnet_b0', 
                           'densenet121', 'custom_cnn', 'custom_cnn_v2'
                       ],
                       help='Model type to train')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--freeze-backbone', action='store_true', default=True,
                       help='Freeze backbone weights (for transfer learning)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--task-type', type=str, default='Variety Classification',
                       help='Type of classification task')
    
    args = parser.parse_args()
    
    # Setup device
    device = setup_gpu()
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, train_size, val_size, test_size, class_names = create_dataloaders(
        args.data_dir, args.batch_size
    )
    
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Dataset sizes: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Create model
    model = create_model(
        model_type=args.model,
        num_classes=num_classes,
        freeze_backbone=args.freeze_backbone,
        device=device
    )
    
    # Train model
    history = train_model(
        model=model,
        model_name=args.model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        task_type=args.task_type,
        output_dir=args.output_dir
    )
    
    # Save training results
    results = {
        'model': args.model,
        'classes': class_names,
        'num_classes': num_classes,
        'dataset_sizes': {
            'train': train_size,
            'val': val_size,
            'test': test_size
        },
        'hyperparameters': {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'freeze_backbone': args.freeze_backbone
        },
        'training_history': history,
        'timestamp': datetime.now().isoformat()
    }
    
    results_filename = f"{args.model}_{args.task_type.lower().replace(' ', '_')}_results.json"
    results_path = os.path.join(args.output_dir, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Training completed! Results saved to {results_path}")

if __name__ == "__main__":
    main()
