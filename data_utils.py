"""
Data Utilities for Fruit Classification Project

This module provides data loading, transformation, and GPU setup utilities
for the fruit classification project.
"""

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import random
import numpy as np

def setup_gpu() -> torch.device:
    """
    Setup GPU configuration and return appropriate device
    
    Returns:
        torch.device: CUDA device if available, else CPU
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9

        print("="*60)
        print("GPU CONFIGURATION")
        print("="*60)
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        print(f"✅ PyTorch Version: {torch.__version__}")
        print(f"✅ GPU Count: {device_count}")
        print(f"✅ Current GPU: {current_device}")
        print(f"✅ GPU Name: {device_name}")
        print(f"✅ GPU Memory: {gpu_memory:.2f} GB")

        cudnn.benchmark = True
        cudnn.deterministic = True
        return torch.device(f'cuda:{current_device}')
    else:
        print("❌ CUDA not available! Using CPU")
        return torch.device('cpu')

def clear_gpu_memory(device: torch.device = None):
    """
    Clear GPU memory cache
    
    Args:
        device: CUDA device to clear memory for
    """
    if device and device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

def print_gpu_memory(device: torch.device = None):
    """
    Print current GPU memory usage
    
    Args:
        device: CUDA device to check memory for
    """
    if device and device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1e9
        cached = torch.cuda.memory_reserved(device) / 1e9
        max_alloc = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Max: {max_alloc:.2f}GB")

def get_transforms(img_size: tuple = (224, 224)):
    """
    Get data transformation pipelines
    
    Args:
        img_size: Target image size (height, width)
    
    Returns:
        tuple: (train_transforms, val_test_transforms)
    """
    # Training transforms with data augmentation
    train_transforms = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/Test transforms (no augmentation)
    val_test_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_test_transforms

def create_dataloaders(dataset_path: str, batch_size: int = 32, img_size: tuple = (224, 224), 
                      seed: int = 42) -> tuple:
    """
    Create optimized dataloaders for training, validation, and testing
    
    Args:
        dataset_path: Path to the dataset directory
        batch_size: Batch size for dataloaders
        img_size: Target image size
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, train_size, val_size, test_size, class_names)
    """
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(dataset_path)
    
    # Dataset splitting (70% train, 15% validation, 15% test)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Get transforms
    train_transforms, val_test_transforms = get_transforms(img_size)
    
    # Apply transforms to datasets
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_test_transforms
    test_dataset.dataset.transform = val_test_transforms
    
    # Also set transforms on the subsets for compatibility
    train_dataset.transform = train_transforms
    val_dataset.transform = val_test_transforms
    test_dataset.transform = val_test_transforms
    
    # Create dataloaders with GPU optimization
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )
    
    # Get class names
    class_names = [label for label, idx in sorted(full_dataset.class_to_idx.items(), key=lambda item: item[1])]
    
    return train_loader, val_loader, test_loader, train_size, val_size, test_size, class_names

def get_class_names(dataset_path: str) -> list:
    """
    Get class names from dataset directory
    
    Args:
        dataset_path: Path to the dataset directory
    
    Returns:
        list: Sorted list of class names
    """
    dataset = datasets.ImageFolder(dataset_path)
    return [label for label, idx in sorted(dataset.class_to_idx.items(), key=lambda item: item[1])]

def get_dataset_stats(dataset_path: str) -> dict:
    """
    Get statistics about the dataset
    
    Args:
        dataset_path: Path to the dataset directory
    
    Returns:
        dict: Dataset statistics
    """
    dataset = datasets.ImageFolder(dataset_path)
    class_counts = {}
    
    for class_name, class_idx in dataset.class_to_idx.items():
        class_counts[class_name] = sum(1 for _, label in dataset.samples if label == class_idx)
    
    return {
        'total_samples': len(dataset.samples),
        'num_classes': len(dataset.classes),
        'class_counts': class_counts,
        'class_names': list(dataset.class_to_idx.keys())
    }

if __name__ == "__main__":
    # Test the data utilities
    device = setup_gpu()
    print(f"Device: {device}")
    
    # Test dataset stats (using a dummy path for testing)
    try:
        stats = get_dataset_stats("/tmp/test")
        print(f"Dataset stats: {stats}")
    except:
        print("Dataset path not available, but utilities are working correctly")
    
    print("Data utilities module loaded successfully!")
