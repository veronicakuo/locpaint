"""
Metrics for evaluating inpainting quality.
Includes SSIM, PSNR, LPIPS, and MSE reconstruction loss.
"""

import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# LPIPS import - will be initialized lazily
_lpips_model = None
_lpips_device = None


def get_lpips_model(device='cuda'):
    """
    Lazy initialization of LPIPS model.
    Uses AlexNet backbone as it's faster and commonly used.
    """
    global _lpips_model, _lpips_device
    
    if _lpips_model is None or _lpips_device != device:
        try:
            import lpips
            _lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device)
            _lpips_model.eval()
            _lpips_device = device
            print(f"LPIPS model loaded on {device}")
        except ImportError:
            print("Warning: lpips not installed. Install with: pip install lpips")
            print("LPIPS metric will return 0.0")
            _lpips_model = None
    
    return _lpips_model


def calculate_lpips(original, completed, device='cuda'):
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity).
    Lower is better (more similar).
    
    Args:
        original: Original images (B, C, H, W) in [-1, 1] range
        completed: Completed/inpainted images (B, C, H, W) in [-1, 1] range
        device: Device to run on
    
    Returns:
        Mean LPIPS score across batch
    """
    lpips_model = get_lpips_model(device)
    
    if lpips_model is None:
        return 0.0
    
    with torch.no_grad():
        # LPIPS expects images in [-1, 1] range
        # Our images should already be in this range
        original = original.to(device)
        completed = completed.to(device)
        
        # Calculate LPIPS
        lpips_score = lpips_model(original, completed)
        
    return lpips_score.mean().item()


def calculate_metrics(original, completed, masks=None, device='cuda'):
    """
    Calculate various metrics for inpainting evaluation.
    
    Args:
        original: Original images (B, C, H, W)
        completed: Inpainted images (B, C, H, W)
        masks: Optional masks to focus on inpainted regions (B, 1, H, W)
        device: Device for LPIPS calculation
    
    Returns:
        Dictionary of metrics: mse, psnr, ssim, lpips
    """
    metrics = {}
    
    # Ensure tensors are on CPU for numpy-based metrics
    original_cpu = original.detach().cpu()
    completed_cpu = completed.detach().cpu()
    if masks is not None:
        masks_cpu = masks.detach().cpu()
    
    # Denormalize from [-1, 1] to [0, 1] for PSNR/SSIM
    original_01 = (original_cpu + 1) / 2
    completed_01 = (completed_cpu + 1) / 2
    
    # ==================
    # MSE (Reconstruction Loss)
    # ==================
    if masks is not None:
        # MSE only on masked region
        mse_masked = ((original_cpu - completed_cpu) ** 2) * masks_cpu
        mse_val = mse_masked.sum() / (masks_cpu.sum() + 1e-8)
    else:
        mse_val = ((original_cpu - completed_cpu) ** 2).mean()
    
    metrics['mse'] = mse_val.item()
    
    # ==================
    # L1 Loss
    # ==================
    if masks is not None:
        l1_masked = torch.abs(original_cpu - completed_cpu) * masks_cpu
        l1_val = l1_masked.sum() / (masks_cpu.sum() + 1e-8)
    else:
        l1_val = torch.abs(original_cpu - completed_cpu).mean()
    
    metrics['l1'] = l1_val.item()
    
    # ==================
    # PSNR and SSIM (per image, then averaged)
    # ==================
    batch_size = original_cpu.size(0)
    psnr_vals = []
    ssim_vals = []
    
    for i in range(batch_size):
        # Convert to numpy (H, W, C) format
        orig_np = original_01[i].permute(1, 2, 0).numpy()
        comp_np = completed_01[i].permute(1, 2, 0).numpy()
        
        # Clip to valid range
        orig_np = np.clip(orig_np, 0, 1)
        comp_np = np.clip(comp_np, 0, 1)
        
        # PSNR
        try:
            psnr_val = psnr(orig_np, comp_np, data_range=1.0)
            if not np.isfinite(psnr_val):
                psnr_val = 100.0  # Perfect reconstruction
            psnr_vals.append(psnr_val)
        except:
            psnr_vals.append(0.0)
        
        # SSIM (on each channel, then average)
        try:
            ssim_val = ssim(orig_np, comp_np, data_range=1.0, 
                          channel_axis=2, win_size=7)
            ssim_vals.append(ssim_val)
        except:
            # Fallback to grayscale SSIM
            orig_gray = np.dot(orig_np, [0.299, 0.587, 0.114])
            comp_gray = np.dot(comp_np, [0.299, 0.587, 0.114])
            ssim_val = ssim(orig_gray, comp_gray, data_range=1.0)
            ssim_vals.append(ssim_val)
    
    metrics['psnr'] = float(np.mean(psnr_vals))
    metrics['ssim'] = float(np.mean(ssim_vals))
    
    # ==================
    # LPIPS (perceptual similarity)
    # ==================
    # LPIPS expects [-1, 1] range, which we already have
    metrics['lpips'] = calculate_lpips(original, completed, device=device)
    
    return metrics


def calculate_validation_metrics(model, val_loader, device='cuda', max_batches=None):
    """
    Calculate metrics over entire validation set.
    
    Args:
        model: The context encoder model (baseline or localizer)
        val_loader: Validation data loader
        device: Device to use
        max_batches: Optional limit on number of batches to evaluate
    
    Returns:
        Dictionary of averaged metrics
    """
    model.encoder.eval()
    model.decoder.eval()
    
    all_metrics = {'mse': [], 'psnr': [], 'ssim': [], 'lpips': [], 'l1': []}
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            images = images.to(device)
            batch_size = images.size(0)
            
            # Generate masks (using model's method)
            masks, bboxes = model.generate_square_mask(batch_size)
            
            # Create masked images
            masked_images = images * (1 - masks)
            
            # Forward pass
            encoded = model.encoder(masked_images)
            reconstructed = model.decoder(encoded)
            
            # Resize if necessary
            if reconstructed.size(2) != images.size(2):
                reconstructed = torch.nn.functional.interpolate(
                    reconstructed,
                    size=images.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Complete images
            completed = masked_images + reconstructed * masks
            
            # Calculate metrics
            batch_metrics = calculate_metrics(images, completed, masks, device=device)
            
            for key in all_metrics:
                all_metrics[key].append(batch_metrics[key])
    
    # Average all metrics
    avg_metrics = {key: float(np.mean(vals)) for key, vals in all_metrics.items()}
    
    model.encoder.train()
    model.decoder.train()
    
    return avg_metrics


def calculate_fixed_val_metrics(model, fixed_images, device='cuda', 
                                mask_seed=None, mask_type='random_square',
                                mask_size=64, image_size=128):
    """
    Calculate metrics on fixed validation images with consistent masks.
    
    Args:
        model: The context encoder model
        fixed_images: Fixed batch of validation images
        device: Device to use
        mask_seed: Seed for reproducible masks (None = use model's random masks)
        mask_type: Type of mask
        mask_size: Size of mask
        image_size: Size of images
    
    Returns:
        metrics: Dictionary of metrics
        completed_images: Completed images for visualization
        masks: Masks used
        bboxes: Bounding boxes
    """
    model.encoder.eval()
    model.decoder.eval()
    
    batch_size = fixed_images.size(0)
    
    with torch.no_grad():
        # Generate masks with optional fixed seed
        if mask_seed is not None:
            np.random.seed(mask_seed)
        
        masks, bboxes = model.generate_square_mask(batch_size)
        
        # Create masked images
        masked_images = fixed_images * (1 - masks)
        
        # Forward pass
        encoded = model.encoder(masked_images)
        reconstructed = model.decoder(encoded)
        
        # Resize if necessary
        if reconstructed.size(2) != fixed_images.size(2):
            reconstructed = torch.nn.functional.interpolate(
                reconstructed,
                size=fixed_images.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Complete images
        completed = masked_images + reconstructed * masks
        
        # Calculate metrics
        metrics = calculate_metrics(fixed_images, completed, masks, device=device)
    
    model.encoder.train()
    model.decoder.train()
    
    return metrics, completed, masked_images, reconstructed, masks, bboxes


def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: First set of boxes [batch, 4] or [batch, num_boxes, 4]
        boxes2: Second set of boxes [batch, 4] or [batch, num_boxes, 4]
    
    Returns:
        IoU scores
    """
    # Ensure proper dimensions
    if boxes1.dim() == 2:
        boxes1 = boxes1.unsqueeze(1)
    if boxes2.dim() == 2:
        boxes2 = boxes2.unsqueeze(1)
    
    # Compute intersection
    x1_max = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1_max = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2_min = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2_min = torch.min(boxes1[..., 3], boxes2[..., 3])
    
    intersection = torch.clamp(x2_min - x1_max, min=0) * torch.clamp(y2_min - y1_max, min=0)
    
    # Compute areas
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-7)
    
    return iou


class MetricsLogger:
    """
    Helper class to log and track metrics during training.
    """
    def __init__(self):
        self.history = {}
    
    def log(self, metrics_dict, prefix=''):
        """Log a dictionary of metrics."""
        for key, value in metrics_dict.items():
            full_key = f"{prefix}{key}" if prefix else key
            if full_key not in self.history:
                self.history[full_key] = []
            self.history[full_key].append(value)
    
    def get_history(self):
        """Get full history dictionary."""
        return self.history
    
    def get_last(self, key):
        """Get last value for a metric."""
        return self.history.get(key, [None])[-1]
    
    def get_best(self, key, mode='min'):
        """Get best value for a metric."""
        values = self.history.get(key, [])
        if not values:
            return None
        return min(values) if mode == 'min' else max(values)
