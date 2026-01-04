"""
Data utilities for context encoder training
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import os

class MaskGenerator:
    """Generate various types of masks for training"""
    
    def __init__(self, image_size=128, mask_size_range=(32, 80)):
        self.image_size = image_size
        self.mask_size_range = mask_size_range
    
    def center_mask(self, batch_size, mask_size=None):
        """Generate center masks"""
        if mask_size is None:
            mask_size = (self.mask_size_range[0] + self.mask_size_range[1]) // 2
        
        masks = torch.zeros(batch_size, 1, self.image_size, self.image_size)
        start = (self.image_size - mask_size) // 2
        end = start + mask_size
        masks[:, :, start:end, start:end] = 1
        
        # Also return bounding boxes
        bboxes = []
        for _ in range(batch_size):
            bbox = torch.tensor([
                start / self.image_size,
                start / self.image_size,
                end / self.image_size,
                end / self.image_size
            ])
            bboxes.append(bbox)
        
        return masks, torch.stack(bboxes)
    
    def random_bbox_mask(self, batch_size):
        """Generate random rectangular masks"""
        masks = torch.zeros(batch_size, 1, self.image_size, self.image_size)
        bboxes = []
        
        for i in range(batch_size):
            # Random mask size
            mask_h = 64
            mask_w = 64
            
            # Random position
            max_y = self.image_size - mask_h
            max_x = self.image_size - mask_w
            y = np.random.randint(0, max_y) if max_y > 0 else 0
            x = np.random.randint(0, max_x) if max_x > 0 else 0
            
            masks[i, 0, y:y+mask_h, x:x+mask_w] = 1
            
            # Normalized bbox coordinates
            bbox = torch.tensor([
                x / self.image_size,
                y / self.image_size,
                (x + mask_w) / self.image_size,
                (y + mask_h) / self.image_size
            ])
            bboxes.append(bbox)
        
        return masks, torch.stack(bboxes)
    
    def irregular_mask(self, batch_size, max_parts=8):
        """Generate irregular masks using random brushstrokes"""
        masks = torch.zeros(batch_size, 1, self.image_size, self.image_size)
        bboxes = []
        
        for i in range(batch_size):
            mask = np.zeros((self.image_size, self.image_size))
            
            # Generate random strokes
            num_parts = np.random.randint(1, max_parts)
            
            for _ in range(num_parts):
                # Random brush parameters
                num_points = np.random.randint(3, 10)
                brush_width = np.random.randint(5, 20)
                
                # Generate path
                points = []
                start_x = np.random.randint(brush_width, self.image_size - brush_width)
                start_y = np.random.randint(brush_width, self.image_size - brush_width)
                
                for _ in range(num_points):
                    points.append([start_x, start_y])
                    start_x = np.clip(start_x + np.random.randint(-20, 21), 
                                     brush_width, self.image_size - brush_width)
                    start_y = np.clip(start_y + np.random.randint(-20, 21),
                                     brush_width, self.image_size - brush_width)
                
                # Draw thick line
                for j in range(len(points) - 1):
                    x1, y1 = points[j]
                    x2, y2 = points[j + 1]
                    
                    # Bresenham's line algorithm with thickness
                    num_interp = max(abs(x2 - x1), abs(y2 - y1))
                    if num_interp > 0:
                        xs = np.linspace(x1, x2, num_interp).astype(int)
                        ys = np.linspace(y1, y2, num_interp).astype(int)
                        
                        for x, y in zip(xs, ys):
                            # Add brush thickness
                            y_start = max(0, y - brush_width // 2)
                            y_end = min(self.image_size, y + brush_width // 2)
                            x_start = max(0, x - brush_width // 2)
                            x_end = min(self.image_size, x + brush_width // 2)
                            mask[y_start:y_end, x_start:x_end] = 1
            
            masks[i, 0] = torch.tensor(mask)
            
            # Compute bounding box of irregular mask
            nonzero = np.nonzero(mask)
            if len(nonzero[0]) > 0:
                y_min, y_max = nonzero[0].min(), nonzero[0].max()
                x_min, x_max = nonzero[1].min(), nonzero[1].max()
                bbox = torch.tensor([
                    x_min / self.image_size,
                    y_min / self.image_size,
                    x_max / self.image_size,
                    y_max / self.image_size
                ])
            else:
                # Fallback to center
                bbox = torch.tensor([0.25, 0.25, 0.75, 0.75])
            bboxes.append(bbox)
        
        return masks, torch.stack(bboxes)
    
    def mixed_mask(self, batch_size, type_probs=None):
        """Generate mixed masks with different types"""
        if type_probs is None:
            type_probs = {'center': 0.3, 'random': 0.4, 'irregular': 0.3}
        
        all_masks = []
        all_bboxes = []
        
        # Determine number of each type
        num_center = int(batch_size * type_probs['center'])
        num_random = int(batch_size * type_probs['random'])
        num_irregular = batch_size - num_center - num_random
        
        # Generate each type
        if num_center > 0:
            masks_c, bboxes_c = self.center_mask(num_center)
            all_masks.append(masks_c)
            all_bboxes.append(bboxes_c)
        
        if num_random > 0:
            masks_r, bboxes_r = self.random_bbox_mask(num_random)
            all_masks.append(masks_r)
            all_bboxes.append(bboxes_r)
        
        if num_irregular > 0:
            masks_i, bboxes_i = self.irregular_mask(num_irregular)
            all_masks.append(masks_i)
            all_bboxes.append(bboxes_i)
        
        # Concatenate and shuffle
        masks = torch.cat(all_masks, dim=0)
        bboxes = torch.cat(all_bboxes, dim=0)
        
        # Shuffle
        perm = torch.randperm(batch_size)
        masks = masks[perm]
        bboxes = bboxes[perm]
        
        return masks, bboxes

class InpaintingDataset(Dataset):
    """Custom dataset wrapper that adds mask generation"""
    
    def __init__(self, base_dataset, mask_generator, mask_type='mixed'):
        self.base_dataset = base_dataset
        self.mask_generator = mask_generator
        self.mask_type = mask_type
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        # Generate mask for this image
        if self.mask_type == 'center':
            mask, bbox = self.mask_generator.center_mask(1)
        elif self.mask_type == 'random':
            mask, bbox = self.mask_generator.random_bbox_mask(1)
        elif self.mask_type == 'irregular':
            mask, bbox = self.mask_generator.irregular_mask(1)
        else:  # mixed
            mask, bbox = self.mask_generator.mixed_mask(1)
        
        return image, mask[0], bbox[0], label

def get_dataloader(config, split='train'):
    """Create dataloader from configuration"""
    
    # Setup transforms
    transform_list = []
    
    if config['dataset']['image_size']:
        transform_list.append(transforms.Resize((
            config['dataset']['image_size'],
            config['dataset']['image_size']
        )))
    
    if split == 'train' and config['dataset']['augmentation']['enabled']:
        if config['dataset']['augmentation']['random_flip']:
            transform_list.append(transforms.RandomHorizontalFlip())
        if config['dataset']['augmentation']['color_jitter']:
            transform_list.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.1))
    
    transform_list.append(transforms.ToTensor())
    
    if config['dataset']['augmentation']['normalize']:
        transform_list.append(transforms.Normalize(
            mean=config['dataset']['augmentation']['mean'],
            std=config['dataset']['augmentation']['std']
        ))
    
    transform = transforms.Compose(transform_list)
    
    # Load dataset
    dataset_name = config['dataset']['name'].lower()
    
    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(
            root=config['dataset']['root'],
            train=(split == 'train'),
            download=True,
            transform=transform
        )
    elif dataset_name == 'celeba':
        dataset = datasets.CelebA(
            root=config['dataset']['root'],
            split='train' if split == 'train' else 'valid',
            download=True,
            transform=transform
        )
    elif dataset_name == 'stl10':
        dataset = datasets.STL10(
            root=config['dataset']['root'],
            split='train' if split == 'train' else 'test',
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader

def create_train_val_dataloaders(config):
    """Create training and validation dataloaders"""
    
    # Load full training dataset
    train_loader = get_dataloader(config, 'train')
    
    # Create validation split if needed
    if config['validation']['enabled']:
        # Load validation dataset
        val_config = config.copy()
        val_config['training']['batch_size'] = min(
            32, config['training']['batch_size']
        )
        val_loader = get_dataloader(val_config, 'test')
    else:
        val_loader = None
    
    return train_loader, val_loader