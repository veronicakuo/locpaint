"""
Comprehensive evaluation metrics for model comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import torchvision.models as models

class ComprehensiveEvaluator:
    """Comprehensive evaluation metrics for inpainting quality"""
    
    def __init__(self, device='cuda', use_lpips=False):
        self.device = device
        self.metrics_history = defaultdict(list)
        
        # Load VGG for perceptual loss
        self.vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Optional: Load LPIPS if available
        self.use_lpips = use_lpips
        if use_lpips:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='alex').to(device)
            except ImportError:
                print("LPIPS not available. Install with: pip install lpips")
                self.use_lpips = False
    
    def compute_metrics(self, original, completed, masks=None):
        """Compute all metrics"""
        metrics = {}
        
        # Ensure tensors are on the same device
        original = original.to(self.device)
        completed = completed.to(self.device)
        if masks is not None:
            masks = masks.to(self.device)
        
        # Pixel-level metrics
        metrics['l1'] = self.compute_l1_loss(original, completed, masks)
        metrics['l2'] = self.compute_l2_loss(original, completed, masks)
        metrics['psnr'] = self.compute_psnr(original, completed, masks)
        metrics['ssim'] = self.compute_ssim(original, completed, masks)
        
        # Perceptual metrics
        metrics['perceptual'] = self.compute_perceptual_loss(original, completed)
        
        if self.use_lpips:
            metrics['lpips'] = self.compute_lpips(original, completed)
        
        return metrics
    
    def compute_l1_loss(self, img1, img2, mask=None):
        """Compute L1 loss"""
        if mask is not None:
            diff = torch.abs((img1 - img2) * mask)
            return (diff.sum() / mask.sum()).item()
        return torch.abs(img1 - img2).mean().item()
    
    def compute_l2_loss(self, img1, img2, mask=None):
        """Compute L2 loss (MSE)"""
        if mask is not None:
            diff = ((img1 - img2) * mask) ** 2
            return (diff.sum() / mask.sum()).item()
        return ((img1 - img2) ** 2).mean().item()
    
    def compute_psnr(self, img1, img2, mask=None):
        """Compute PSNR"""
        mse = self.compute_l2_loss(img1, img2, mask)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(2.0) - 10 * np.log10(mse)
    
    def compute_ssim(self, img1, img2, mask=None):
        """Compute SSIM (simplified version)"""
        # This is a simplified SSIM computation
        # For exact SSIM, use skimage.metrics.structural_similarity
        
        mu1 = F.avg_pool2d(img1, 3, 1, padding=1)
        mu2 = F.avg_pool2d(img2, 3, 1, padding=1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, padding=1) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, padding=1) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1, padding=1) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if mask is not None:
            return (ssim_map * mask).sum() / mask.sum()
        return ssim_map.mean().item()
    
    def compute_perceptual_loss(self, img1, img2):
        """Compute perceptual loss using VGG features"""
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        img1 = (img1 + 1) / 2  # [-1,1] to [0,1]
        img2 = (img2 + 1) / 2
        
        img1 = (img1 - mean) / std
        img2 = (img2 - mean) / std
        
        feat1 = self.vgg(img1)
        feat2 = self.vgg(img2)
        
        return F.mse_loss(feat1, feat2).item()
    
    def compute_lpips(self, img1, img2):
        """Compute LPIPS perceptual distance"""
        if not self.use_lpips:
            return 0.0
        
        # LPIPS expects inputs in [-1, 1]
        return self.lpips_fn(img1, img2).mean().item()
    
    def evaluate_batch(self, original, completed, masks, model_name=""):
        """Evaluate a batch and store results"""
        metrics = self.compute_metrics(original, completed, masks)
        
        # Store in history
        for key, value in metrics.items():
            self.metrics_history[f"{model_name}_{key}"].append(value)
        
        return metrics