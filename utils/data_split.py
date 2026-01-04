"""
Data splitting utilities for consistent train/val/test splits across models.
Uses a fixed seed to ensure the same split is used by both baseline and localizer.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image
import json


class ImageFolderFlat(Dataset):
    """
    Dataset for loading images directly from a folder (no subdirectories required).
    Unlike ImageFolder, this works when all images are in a single directory.
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        # Find all image files
        self.image_files = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        
        for filename in sorted(os.listdir(root)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                self.image_files.append(os.path.join(root, filename))
        
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {root}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Return dummy label (0) for compatibility with training code
        return image, 0


def get_celeba_splits(data_root, image_size=128, split_seed=42, 
                      train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Create consistent train/val/test splits for CelebA dataset.
    
    Supports two directory structures:
    1. Official CelebA structure (via torchvision.datasets.CelebA)
    2. Plain image folder (via torchvision.datasets.ImageFolder)
    
    Args:
        data_root: Root directory containing CelebA data
        image_size: Size to resize images to
        split_seed: Random seed for reproducible splits
        train_ratio: Proportion of data for training (default 0.8)
        val_ratio: Proportion of data for validation (default 0.1)
        test_ratio: Proportion of data for testing (default 0.1)
    
    Returns:
        train_dataset, val_dataset, test_dataset, split_indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Standard transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Try different methods to load the dataset
    full_dataset = None
    
    # Method 1: Try official CelebA structure
    try:
        full_dataset = datasets.CelebA(
            root=data_root,
            split='all',
            download=False,
            transform=transform
        )
        print(f"Loaded CelebA dataset from {data_root} (official structure)")
    except (RuntimeError, FileNotFoundError):
        pass
    
    # Method 2: Try ImageFolder with celeba subdirectory
    if full_dataset is None:
        celeba_paths = [
            os.path.join(data_root, 'celeba'),
            os.path.join(data_root, 'CelebA'),
            os.path.join(data_root, 'celeba', 'img_align_celeba'),
            os.path.join(data_root, 'CelebA', 'img_align_celeba'),
            os.path.join(data_root, 'img_align_celeba'),
            data_root  # Try data_root directly
        ]
        
        for celeba_path in celeba_paths:
            if os.path.isdir(celeba_path):
                # Check if it has images directly or subdirectories
                contents = os.listdir(celeba_path)
                
                # If directory contains images directly, wrap in ImageFolderFlat
                has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in contents[:100])
                has_subdirs = any(os.path.isdir(os.path.join(celeba_path, f)) for f in contents)
                
                if has_images and not has_subdirs:
                    # Images directly in folder - use custom dataset
                    full_dataset = ImageFolderFlat(celeba_path, transform=transform)
                    print(f"Loaded images from {celeba_path} (flat folder)")
                    break
                elif has_subdirs:
                    # Has subdirectories - use ImageFolder
                    try:
                        full_dataset = datasets.ImageFolder(celeba_path, transform=transform)
                        print(f"Loaded images from {celeba_path} (ImageFolder)")
                        break
                    except:
                        continue
    
    if full_dataset is None:
        raise RuntimeError(
            f"Could not find CelebA dataset in {data_root}. "
            f"Please ensure your images are in one of these locations:\n"
            f"  - {data_root}/celeba/\n"
            f"  - {data_root}/img_align_celeba/\n"
            f"  - {data_root}/ (images directly)\n"
            f"Or use the official CelebA structure with metadata files."
        )
    
    # Get total number of samples
    total_samples = len(full_dataset)
    print(f"Total images found: {total_samples}")
    
    # Create reproducible split indices
    np.random.seed(split_seed)
    indices = np.random.permutation(total_samples)
    
    # Calculate split points
    train_end = int(train_ratio * total_samples)
    val_end = train_end + int(val_ratio * total_samples)
    
    train_indices = indices[:train_end].tolist()
    val_indices = indices[train_end:val_end].tolist()
    test_indices = indices[val_end:].tolist()
    
    # Create subset datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    split_info = {
        'seed': split_seed,
        'total_samples': total_samples,
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'test_size': len(test_indices),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio
    }
    
    print(f"Dataset splits (seed={split_seed}):")
    print(f"  Train: {len(train_indices):,} samples ({train_ratio*100:.1f}%)")
    print(f"  Val:   {len(val_indices):,} samples ({val_ratio*100:.1f}%)")
    print(f"  Test:  {len(test_indices):,} samples ({test_ratio*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset, split_info


def get_fixed_val_indices(val_dataset, num_samples=10, seed=42):
    """
    Get fixed indices for validation visualization.
    These same images will be used every epoch for consistent comparison.
    
    Args:
        val_dataset: Validation dataset
        num_samples: Number of fixed samples to use
        seed: Random seed for selection
    
    Returns:
        List of indices into the validation dataset
    """
    np.random.seed(seed)
    total_val = len(val_dataset)
    fixed_indices = np.random.choice(total_val, min(num_samples, total_val), replace=False)
    return fixed_indices.tolist()


def get_fixed_val_batch(val_dataset, fixed_indices, device='cuda'):
    """
    Get a fixed batch of validation images for consistent visualization.
    
    Args:
        val_dataset: Validation dataset
        fixed_indices: Indices of fixed validation samples
        device: Device to move tensors to
    
    Returns:
        Tensor of fixed validation images
    """
    images = []
    for idx in fixed_indices:
        img, _ = val_dataset[idx]
        images.append(img)
    
    return torch.stack(images).to(device)


class FixedMaskDataset(Dataset):
    """
    Wrapper dataset that applies fixed masks to specific indices.
    Used for consistent validation where we want the same mask positions.
    """
    def __init__(self, base_dataset, mask_size=64, image_size=128, 
                 mask_type='random_square', seed=42):
        self.base_dataset = base_dataset
        self.mask_size = mask_size
        self.image_size = image_size
        self.mask_type = mask_type
        
        # Pre-generate fixed mask positions for each sample
        np.random.seed(seed)
        self.mask_positions = []
        
        for _ in range(len(base_dataset)):
            if mask_type == 'center':
                start = (image_size - mask_size) // 2
                self.mask_positions.append((start, start))
            else:
                max_pos = image_size - mask_size
                x = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
                y = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
                self.mask_positions.append((x, y))
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        x, y = self.mask_positions[idx]
        
        # Create mask
        mask = torch.zeros(1, self.image_size, self.image_size)
        mask[:, y:y+self.mask_size, x:x+self.mask_size] = 1
        
        # Create bbox
        bbox = torch.tensor([
            x / self.image_size,
            y / self.image_size,
            (x + self.mask_size) / self.image_size,
            (y + self.mask_size) / self.image_size
        ])
        
        return image, mask, bbox, label


def get_dataloaders(args):
    """
    Create dataloaders for training with consistent splits.
    
    Args:
        args: Argument namespace with data_root, batch_size, etc.
    
    Returns:
        train_loader, val_loader, test_loader, fixed_val_images, split_info
    """
    # Get splits
    train_dataset, val_dataset, test_dataset, split_info = get_celeba_splits(
        data_root=args.data_root,
        image_size=args.image_size,
        split_seed=getattr(args, 'split_seed', 42),
        train_ratio=getattr(args, 'train_ratio', 0.8),
        val_ratio=getattr(args, 'val_ratio', 0.1),
        test_ratio=getattr(args, 'test_ratio', 0.1)
    )
    
    # Get fixed validation indices for consistent visualization
    fixed_val_indices = get_fixed_val_indices(
        val_dataset, 
        num_samples=getattr(args, 'num_fixed_val', 10),
        seed=getattr(args, 'split_seed', 42)
    )
    
    # Store fixed indices in split_info
    split_info['fixed_val_indices'] = fixed_val_indices
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Get fixed validation batch
    fixed_val_images = get_fixed_val_batch(
        val_dataset, 
        fixed_val_indices, 
        device=args.device
    )
    
    return train_loader, val_loader, test_loader, fixed_val_images, split_info


def save_split_info(split_info, save_path):
    """Save split information to JSON for reproducibility."""
    # Convert numpy types to Python types for JSON serialization
    serializable_info = {}
    for k, v in split_info.items():
        if isinstance(v, np.ndarray):
            serializable_info[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            serializable_info[k] = v.item()
        else:
            serializable_info[k] = v
    
    with open(save_path, 'w') as f:
        json.dump(serializable_info, f, indent=2)
    print(f"Split info saved to {save_path}")


def load_split_info(load_path):
    """Load split information from JSON."""
    with open(load_path, 'r') as f:
        return json.load(f)
