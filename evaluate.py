"""
Comprehensive Evaluation Script for Context Encoder Models
Works with: LION (Localizer), Baseline (Pathak et al.), Staged, and generic encoder-decoder models

Computes: PSNR, SSIM, LPIPS, MSE metrics
Generates: Visualization grids of masked, reconstructed, completed, ground truth

Usage:
    # For LION/Staged model (auto-imports model class if available)
    python evaluate.py --checkpoint ./localizer/checkpoints/final_model.pth --model_type staged
    
    # For Baseline model  
    python evaluate.py --checkpoint ./baseline/checkpoints/final_model.pth --model_type baseline
    
    # With explicit model file (for custom architectures)
    python evaluate.py --checkpoint ./model.pth --model_file ./staged_context_encoder.py
    
    # With custom settings
    python evaluate.py --checkpoint ./model.pth --model_type staged --num_samples 1000 --save_dir ./eval_results
    
    # Evaluate on specific dataset split
    python evaluate.py --checkpoint ./model.pth --model_type baseline --dataset cifar10 --split test

Model Loading Priority:
    1. If --model_file is provided, imports model classes from that file
    2. Tries to import StagedContextEncoder from staged_context_encoder.py
    3. Tries to import ContextEncoder from context_encoder.py
    4. Falls back to built-in Encoder/Decoder classes
"""

import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# Metrics imports
try:
    from skimage.metrics import structural_similarity as ssim_skimage
    from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Install with: pip install scikit-image")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips")

# Try to import ComprehensiveEvaluator if available
COMPREHENSIVE_EVALUATOR_CLASS = None
try:
    from evaluation_metrics import ComprehensiveEvaluator
    COMPREHENSIVE_EVALUATOR_CLASS = ComprehensiveEvaluator
    print("✓ ComprehensiveEvaluator imported successfully")
except ImportError:
    pass


# =============================================================================
# TRY TO IMPORT EXISTING MODEL DEFINITIONS
# =============================================================================

# Common locations to check for model files
model_search_paths = [
    os.path.dirname(os.path.abspath(__file__)),
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '/home/claude',
    '/mnt/user-data/uploads',
    './localizer',
    './baseline',
]

for path in model_search_paths:
    if path not in sys.path:
        sys.path.insert(0, path)

# Track which model classes are available
STAGED_ENCODER_CLASS = None
BASELINE_ENCODER_CLASS = None

try:
    from staged_context_encoder import StagedContextEncoder
    STAGED_ENCODER_CLASS = StagedContextEncoder
    print("✓ StagedContextEncoder imported successfully")
except ImportError:
    try:
        from models.staged_context_encoder import StagedContextEncoder
        STAGED_ENCODER_CLASS = StagedContextEncoder
        print("✓ StagedContextEncoder imported from models/")
    except ImportError:
        pass

try:
    from context_encoder import ContextEncoder
    BASELINE_ENCODER_CLASS = ContextEncoder
    print("✓ Baseline ContextEncoder imported successfully")
except ImportError:
    try:
        from models.context_encoder import ContextEncoder
        BASELINE_ENCODER_CLASS = ContextEncoder
        print("✓ Baseline ContextEncoder imported from models/")
    except ImportError:
        pass


# =============================================================================
# FALLBACK MODEL DEFINITIONS (standalone for evaluation)
# =============================================================================

class Encoder(nn.Module):
    """Encoder network - shared architecture for all models"""
    def __init__(self, input_channels=3, latent_dim=4000):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)
        
        self.channel_fc = nn.Conv2d(512, latent_dim, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.channel_fc(x)
        return x


class Decoder(nn.Module):
    """Decoder network - shared architecture for all models"""
    def __init__(self, latent_dim=4000):
        super(Decoder, self).__init__()
        
        self.channel_fc = nn.Conv2d(latent_dim, 512, kernel_size=1)
        
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        
    def forward(self, x):
        x = F.relu(self.channel_fc(x))
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))
        return x


class EvaluationModel:
    """
    Wrapper class for evaluation that works with any encoder-decoder model
    Supports: baseline, staged, lion, or generic checkpoints
    
    Loading priority:
    1. If StagedContextEncoder/ContextEncoder classes are available, use them
    2. Otherwise, use fallback Encoder/Decoder classes
    """
    
    def __init__(self, checkpoint_path, model_type='auto', device='cuda', 
                 model_file=None):
        """
        Initialize evaluation model
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_type: 'baseline', 'staged', 'lion', or 'auto' (auto-detect)
            device: Device to run on
            model_file: Optional path to model definition file (will try to import)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        self.full_model = None  # Will hold StagedContextEncoder or ContextEncoder if available
        
        # Try to import from model_file if provided
        if model_file:
            self._import_from_file(model_file)
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Auto-detect model type if needed
        if model_type == 'auto':
            model_type = self._detect_model_type()
        self.model_type = model_type
        print(f"Model type: {model_type}")
        
        # Extract config
        self.config = self._extract_config()
        
        # Try to load full model class first, then fall back to separate encoder/decoder
        if self._try_load_full_model():
            print("✓ Loaded using full model class")
        else:
            print("Using fallback encoder/decoder loading")
            self._load_fallback_model()
        
        # Store image/mask parameters
        self.image_size = self.config.get('image_size', 128)
        self.mask_size = self.config.get('mask_size', 64)
        self.mask_type = self.config.get('mask_type', 'center')
        
        print(f"Model loaded successfully!")
        print(f"  Image size: {self.image_size}")
        print(f"  Mask size: {self.mask_size}")
        print(f"  Mask type: {self.mask_type}")
    
    def _import_from_file(self, model_file):
        """Try to import model classes from a specific file"""
        global STAGED_ENCODER_CLASS, BASELINE_ENCODER_CLASS
        
        if not os.path.exists(model_file):
            print(f"Warning: Model file not found: {model_file}")
            return
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("model_module", model_file)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            
            if hasattr(module, 'StagedContextEncoder'):
                STAGED_ENCODER_CLASS = module.StagedContextEncoder
                print(f"✓ Imported StagedContextEncoder from {model_file}")
            if hasattr(module, 'ContextEncoder'):
                BASELINE_ENCODER_CLASS = module.ContextEncoder
                print(f"✓ Imported ContextEncoder from {model_file}")
        except Exception as e:
            print(f"Warning: Could not import from {model_file}: {e}")
    
    def _detect_model_type(self):
        """Auto-detect model type from checkpoint structure"""
        # Check for staged/lion indicators
        if 'binary_discriminator_state' in self.checkpoint:
            return 'staged'
        elif 'localizer_discriminator_state' in self.checkpoint:
            return 'lion'
        elif 'discriminator_state' in self.checkpoint:
            return 'baseline'
        elif 'encoder_state' in self.checkpoint:
            return 'generic'
        # Check model_type in config
        elif 'config' in self.checkpoint and 'model_type' in self.checkpoint['config']:
            return self.checkpoint['config']['model_type']
        else:
            print("Warning: Could not detect model type, assuming generic")
            return 'generic'
    
    def _extract_config(self):
        """Extract config from checkpoint"""
        if 'config' in self.checkpoint:
            return self.checkpoint['config']
        elif 'args' in self.checkpoint:
            # Some checkpoints store args instead of config
            args = self.checkpoint['args']
            if hasattr(args, '__dict__'):
                return vars(args)
            return args
        else:
            # Default config
            return {
                'image_size': 128,
                'mask_size': 64,
                'mask_type': 'center',
                'latent_dim': 4000
            }
    
    def _try_load_full_model(self):
        """
        Try to load using full model class (StagedContextEncoder or ContextEncoder)
        Returns True if successful, False otherwise
        """
        global STAGED_ENCODER_CLASS, BASELINE_ENCODER_CLASS
        
        model_class = None
        
        # Select appropriate model class based on type
        if self.model_type in ['staged', 'lion'] and STAGED_ENCODER_CLASS is not None:
            model_class = STAGED_ENCODER_CLASS
        elif self.model_type == 'baseline' and BASELINE_ENCODER_CLASS is not None:
            model_class = BASELINE_ENCODER_CLASS
        elif STAGED_ENCODER_CLASS is not None:
            # Default to staged if available
            model_class = STAGED_ENCODER_CLASS
        elif BASELINE_ENCODER_CLASS is not None:
            model_class = BASELINE_ENCODER_CLASS
        
        if model_class is None:
            return False
        
        try:
            # Extract constructor arguments from config
            latent_dim = self.config.get('latent_dim', 4000)
            image_size = self.config.get('image_size', 128)
            mask_size = self.config.get('mask_size', 64)
            
            # Instantiate the model
            print(f"Instantiating {model_class.__name__}...")
            self.full_model = model_class(
                latent_dim=latent_dim,
                image_size=image_size,
                mask_size=mask_size
            )
            
            # Load state dict
            if 'encoder_state' in self.checkpoint:
                self.full_model.encoder.load_state_dict(self.checkpoint['encoder_state'])
            if 'decoder_state' in self.checkpoint:
                self.full_model.decoder.load_state_dict(self.checkpoint['decoder_state'])
            
            # Also set encoder/decoder references for compatibility
            self.encoder = self.full_model.encoder.to(self.device)
            self.decoder = self.full_model.decoder.to(self.device)
            self.encoder.eval()
            self.decoder.eval()
            
            return True
            
        except Exception as e:
            print(f"Warning: Could not load full model class: {e}")
            return False
    
    def _load_fallback_model(self):
        """Load using fallback Encoder/Decoder classes"""
        latent_dim = self.config.get('latent_dim', 4000)
        self.encoder = Encoder(latent_dim=latent_dim).to(self.device)
        self.decoder = Decoder(latent_dim=latent_dim).to(self.device)
        
        # Try different key names for compatibility
        encoder_keys = ['encoder_state', 'encoder_state_dict', 'encoder']
        decoder_keys = ['decoder_state', 'decoder_state_dict', 'decoder']
        
        encoder_loaded = False
        for key in encoder_keys:
            if key in self.checkpoint:
                try:
                    self.encoder.load_state_dict(self.checkpoint[key])
                    encoder_loaded = True
                    print(f"  Loaded encoder from '{key}'")
                    break
                except Exception as e:
                    print(f"  Warning: Could not load encoder from '{key}': {e}")
        
        decoder_loaded = False
        for key in decoder_keys:
            if key in self.checkpoint:
                try:
                    self.decoder.load_state_dict(self.checkpoint[key])
                    decoder_loaded = True
                    print(f"  Loaded decoder from '{key}'")
                    break
                except Exception as e:
                    print(f"  Warning: Could not load decoder from '{key}': {e}")
        
        if not encoder_loaded or not decoder_loaded:
            raise ValueError("Could not find encoder/decoder weights in checkpoint. "
                           "Try providing --model_file pointing to your model definition.")
        
        self.encoder.eval()
        self.decoder.eval()
    
    def generate_mask(self, batch_size):
        """Generate masks for evaluation"""
        masks = torch.zeros(batch_size, 1, self.image_size, self.image_size)
        bboxes = []
        
        for i in range(batch_size):
            if self.mask_type == 'center':
                start = (self.image_size - self.mask_size) // 2
                end = start + self.mask_size
                masks[i, :, start:end, start:end] = 1
                bbox = [start / self.image_size, start / self.image_size,
                       end / self.image_size, end / self.image_size]
            elif self.mask_type == 'random_square':
                max_pos = self.image_size - self.mask_size
                x = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
                y = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
                masks[i, :, y:y+self.mask_size, x:x+self.mask_size] = 1
                bbox = [x / self.image_size, y / self.image_size,
                       (x + self.mask_size) / self.image_size,
                       (y + self.mask_size) / self.image_size]
            else:
                # Default to center
                start = (self.image_size - self.mask_size) // 2
                end = start + self.mask_size
                masks[i, :, start:end, start:end] = 1
                bbox = [start / self.image_size, start / self.image_size,
                       end / self.image_size, end / self.image_size]
            
            bboxes.append(bbox)
        
        return masks.to(self.device), torch.tensor(bboxes).to(self.device)
    
    @torch.no_grad()
    def inpaint(self, images, masks, bboxes=None):
        """
        Perform inpainting
        
        Args:
            images: Original images [B, C, H, W] in range [-1, 1]
            masks: Binary masks [B, 1, H, W] (1 = masked region)
            bboxes: Optional bounding boxes for mask locations [B, 4] (x1, y1, x2, y2 normalized)
        
        Returns:
            masked_images: Images with masked region zeroed
            reconstructed: Raw decoder output (64x64)
            completed: Final completed images (128x128 with decoder output placed in mask)
        """
        # Create masked images
        masked_images = images * (1 - masks)
        
        # Forward pass
        encoded = self.encoder(masked_images)
        reconstructed = self.decoder(encoded)  # This is 64x64
        
        # Create completed image by placing decoder output into masked region
        completed = images.clone()
        
        batch_size = images.size(0)
        decoder_size = reconstructed.size(2)  # Should be 64
        
        for i in range(batch_size):
            # Find mask bounding box for this sample
            mask_i = masks[i, 0]  # [H, W]
            
            # Get mask region coordinates
            nonzero = torch.nonzero(mask_i, as_tuple=True)
            if len(nonzero[0]) > 0:
                y_min, y_max = nonzero[0].min().item(), nonzero[0].max().item() + 1
                x_min, x_max = nonzero[1].min().item(), nonzero[1].max().item() + 1
                
                # Get the region size
                region_h = y_max - y_min
                region_w = x_max - x_min
                
                # Resize decoder output if needed to match mask region size
                if decoder_size != region_h or decoder_size != region_w:
                    recon_resized = F.interpolate(
                        reconstructed[i:i+1],
                        size=(region_h, region_w),
                        mode='bilinear',
                        align_corners=False
                    )[0]
                else:
                    recon_resized = reconstructed[i]
                
                # Place into completed image
                completed[i, :, y_min:y_max, x_min:x_max] = recon_resized
        
        # For visualization, also create a full-size version of reconstructed
        # by resizing to match image size
        reconstructed_full = F.interpolate(
            reconstructed,
            size=images.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return masked_images, reconstructed_full, completed


# =============================================================================
# METRIC COMPUTATION FUNCTIONS
# =============================================================================

def compute_mse(pred, target, mask=None):
    """
    Compute Mean Squared Error
    
    Args:
        pred: Predicted images [B, C, H, W] in range [0, 1]
        target: Target images [B, C, H, W] in range [0, 1]
        mask: Optional mask to compute MSE only in masked region [B, 1, H, W]
    
    Returns:
        MSE value (scalar)
    """
    if mask is not None:
        # Compute MSE only in masked region
        diff = (pred - target) ** 2
        diff = diff * mask
        mse = diff.sum() / (mask.sum() * pred.size(1) + 1e-8)
    else:
        mse = F.mse_loss(pred, target)
    return mse.item()


def compute_psnr(pred, target, mask=None, data_range=1.0):
    """
    Compute Peak Signal-to-Noise Ratio
    
    Args:
        pred: Predicted images [B, C, H, W] in range [0, 1]
        target: Target images [B, C, H, W] in range [0, 1]
        mask: Optional mask [B, 1, H, W]
        data_range: Data range (default 1.0 for normalized images)
    
    Returns:
        PSNR value in dB
    """
    if SKIMAGE_AVAILABLE:
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        psnr_values = []
        for i in range(pred_np.shape[0]):
            p = np.transpose(pred_np[i], (1, 2, 0))
            t = np.transpose(target_np[i], (1, 2, 0))
            
            p = np.clip(p, 0, 1)
            t = np.clip(t, 0, 1)
            
            psnr_val = psnr_skimage(t, p, data_range=data_range)
            psnr_values.append(psnr_val)
        
        return np.mean(psnr_values)
    else:
        # Fallback: compute from MSE
        mse = compute_mse(pred, target, mask)
        if mse < 1e-10:
            return 100.0
        psnr = 10 * np.log10((data_range ** 2) / mse)
        return psnr


def compute_ssim(pred, target, mask=None, data_range=1.0):
    """
    Compute Structural Similarity Index
    
    Args:
        pred: Predicted images [B, C, H, W] in range [0, 1]
        target: Target images [B, C, H, W] in range [0, 1]
        mask: Optional mask [B, 1, H, W]
        data_range: Data range (default 1.0)
    
    Returns:
        SSIM value (0 to 1)
    """
    if SKIMAGE_AVAILABLE:
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        ssim_values = []
        for i in range(pred_np.shape[0]):
            p = np.transpose(pred_np[i], (1, 2, 0))
            t = np.transpose(target_np[i], (1, 2, 0))
            
            p = np.clip(p, 0, 1)
            t = np.clip(t, 0, 1)
            
            # Determine appropriate window size
            min_dim = min(p.shape[0], p.shape[1])
            win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
            
            ssim_val = ssim_skimage(
                t, p,
                data_range=data_range,
                channel_axis=2,
                win_size=win_size
            )
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    else:
        print("Warning: SSIM requires scikit-image")
        return 0.0


def compute_lpips_metric(pred, target, lpips_model):
    """
    Compute Learned Perceptual Image Patch Similarity
    
    Args:
        pred: Predicted images [B, C, H, W] in range [0, 1]
        target: Target images [B, C, H, W] in range [0, 1]
        lpips_model: Pre-loaded LPIPS model
    
    Returns:
        LPIPS value (lower is better)
    """
    if lpips_model is None:
        return 0.0
    
    # LPIPS expects images in range [-1, 1]
    pred_scaled = pred * 2 - 1
    target_scaled = target * 2 - 1
    
    with torch.no_grad():
        lpips_val = lpips_model(pred_scaled, target_scaled)
    
    return lpips_val.mean().item()


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def denormalize(tensor):
    """Convert from [-1, 1] to [0, 1]"""
    return (tensor + 1) / 2


def create_evaluation_grid(masked, reconstructed, completed, original, masks,
                           save_path, num_samples=8):
    """
    Create evaluation visualization grid
    
    Args:
        masked: Masked images [B, C, H, W]
        reconstructed: Reconstructed patches [B, C, H, W]
        completed: Completed images [B, C, H, W]
        original: Original images [B, C, H, W]
        masks: Binary masks [B, 1, H, W]
        save_path: Path to save visualization
        num_samples: Number of samples to visualize
    """
    batch_size = min(num_samples, original.size(0))
    
    # Denormalize all images
    original = denormalize(original[:batch_size])
    masked = denormalize(masked[:batch_size])
    reconstructed = denormalize(reconstructed[:batch_size])
    completed = denormalize(completed[:batch_size])
    masks = masks[:batch_size]
    
    # Create grid
    fig, axes = plt.subplots(batch_size, 4, figsize=(12, 3 * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    column_titles = ['Original', 'Masked', 'Reconstructed', 'Completed']
    
    for i in range(batch_size):
        images_row = [original[i], masked[i], reconstructed[i], completed[i]]
        
        for j, img_tensor in enumerate(images_row):
            img = img_tensor.cpu().permute(1, 2, 0).detach().numpy()
            axes[i, j].imshow(np.clip(img, 0, 1))
            
            if i == 0:
                axes[i, j].set_title(column_titles[j], fontsize=12)
            
            axes[i, j].axis('off')
    
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.suptitle('Inpainting Evaluation Results', fontsize=14, y=1.02)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved evaluation grid to {save_path}")


def create_comparison_figure(results_list, save_path):
    """
    Create comparison figure for multiple models
    
    Args:
        results_list: List of dicts with 'name', 'images' (dict with masked/reconstructed/completed/original)
        save_path: Path to save figure
    """
    num_models = len(results_list)
    num_samples = min(4, results_list[0]['images']['original'].size(0))
    
    fig, axes = plt.subplots(num_samples, num_models + 2, figsize=(3 * (num_models + 2), 3 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for row in range(num_samples):
        # Original
        img = denormalize(results_list[0]['images']['original'][row])
        img = img.cpu().permute(1, 2, 0).numpy()
        axes[row, 0].imshow(np.clip(img, 0, 1))
        if row == 0:
            axes[row, 0].set_title('Original', fontsize=10)
        axes[row, 0].axis('off')
        
        # Masked
        img = denormalize(results_list[0]['images']['masked'][row])
        img = img.cpu().permute(1, 2, 0).numpy()
        axes[row, 1].imshow(np.clip(img, 0, 1))
        if row == 0:
            axes[row, 1].set_title('Masked', fontsize=10)
        axes[row, 1].axis('off')
        
        # Each model's completion
        for col, result in enumerate(results_list):
            img = denormalize(result['images']['completed'][row])
            img = img.cpu().permute(1, 2, 0).numpy()
            axes[row, col + 2].imshow(np.clip(img, 0, 1))
            if row == 0:
                axes[row, col + 2].set_title(result['name'], fontsize=10)
            axes[row, col + 2].axis('off')
    
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.suptitle('Model Comparison', fontsize=14, y=1.02)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_bar(metrics_dict, save_path):
    """
    Plot metrics as bar chart
    
    Args:
        metrics_dict: Dictionary with metric names and values
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    metrics_to_plot = ['PSNR', 'SSIM', 'LPIPS', 'MSE']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    for idx, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        if metric in metrics_dict:
            value = metrics_dict[metric]
            axes[idx].bar([metric], [value], color=color, edgecolor='black', linewidth=1.2)
            axes[idx].set_title(metric, fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Value')
            
            # Add value label on bar
            axes[idx].text(0, value + value * 0.02, f'{value:.4f}', 
                          ha='center', va='bottom', fontsize=11)
            
            # Set appropriate y-axis limits
            if metric == 'SSIM':
                axes[idx].set_ylim([0, 1])
            elif metric == 'LPIPS':
                axes[idx].set_ylim([0, max(0.5, value * 1.2)])
    
    plt.suptitle('Evaluation Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# DATASET LOADING
# =============================================================================

def get_eval_dataloader(args):
    """Create evaluation data loader"""
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            root=args.data_root,
            train=(args.split == 'train'),
            download=True,
            transform=transform
        )
    elif args.dataset == 'cifar100':
        dataset = datasets.CIFAR100(
            root=args.data_root,
            train=(args.split == 'train'),
            download=True,
            transform=transform
        )
    elif args.dataset == 'celeba':
        split_map = {'train': 'train', 'val': 'valid', 'test': 'test'}
        dataset = datasets.CelebA(
            root=args.data_root,
            split=split_map.get(args.split, 'test'),
            download=True,
            transform=transform
        )
    elif args.dataset == 'stl10':
        split_map = {'train': 'train', 'test': 'test', 'val': 'test'}
        dataset = datasets.STL10(
            root=args.data_root,
            split=split_map.get(args.split, 'test'),
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Limit number of samples if specified
    if args.num_samples > 0 and args.num_samples < len(dataset):
        indices = np.random.choice(len(dataset), args.num_samples, replace=False)
        dataset = Subset(dataset, indices)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return dataloader


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate(args):
    """Main evaluation function"""
    print(f"\n{'='*60}")
    print(f"Context Encoder Evaluation")
    print(f"{'='*60}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load model
    model = EvaluationModel(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
        model_file=args.model_file if args.model_file else None
    )
    
    # Override mask settings if specified via command line
    # Otherwise, use model's config values
    if args.mask_type:
        model.mask_type = args.mask_type
    if args.mask_size > 0:
        model.mask_size = args.mask_size
    if args.image_size > 0:
        model.image_size = args.image_size
    
    # IMPORTANT: Update args with model config values for dataloader
    # This ensures the dataloader uses correct image size from checkpoint
    if args.image_size <= 0:
        args.image_size = model.image_size
    if args.mask_size <= 0:
        args.mask_size = model.mask_size
    if not args.mask_type:
        args.mask_type = model.mask_type
    
    # Load LPIPS model if available (for fallback metrics)
    lpips_model = None
    if LPIPS_AVAILABLE and COMPREHENSIVE_EVALUATOR_CLASS is None:
        print("Loading LPIPS model...")
        lpips_model = lpips.LPIPS(net='alex').to(model.device)
        lpips_model.eval()
    
    # Use ComprehensiveEvaluator if available
    comprehensive_evaluator = None
    if COMPREHENSIVE_EVALUATOR_CLASS is not None:
        print("Using ComprehensiveEvaluator for metrics...")
        comprehensive_evaluator = COMPREHENSIVE_EVALUATOR_CLASS(
            device=model.device,
            use_lpips=LPIPS_AVAILABLE
        )
    
    # Get data loader (now using correct image_size from model config)
    print(f"\nLoading {args.dataset} dataset ({args.split} split)...")
    dataloader = get_eval_dataloader(args)
    print(f"Evaluating on {len(dataloader.dataset)} samples")
    
    # Initialize metric accumulators
    all_metrics = {
        'MSE': [],
        'MSE_masked': [],
        'PSNR': [],
        'PSNR_masked': [],
        'SSIM': [],
        'SSIM_masked': [],
        'LPIPS': [],
        'L1': [],
        'L1_masked': [],
        'perceptual': []
    }
    
    # Store some samples for visualization
    vis_samples = {
        'original': [],
        'masked': [],
        'reconstructed': [],
        'completed': [],
        'masks': []
    }
    num_vis_samples = min(args.num_vis_samples, len(dataloader.dataset))
    vis_collected = 0
    
    # Evaluation loop
    print("\nRunning evaluation...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(model.device)
            batch_size = images.size(0)
            
            # Generate masks
            masks, _ = model.generate_mask(batch_size)
            
            # Perform inpainting
            masked_images, reconstructed, completed = model.inpaint(images, masks)
            
            # Use ComprehensiveEvaluator if available
            if comprehensive_evaluator is not None:
                # ComprehensiveEvaluator expects [-1, 1] range
                metrics = comprehensive_evaluator.compute_metrics(images, completed, masks)
                
                all_metrics['L1'].append(metrics.get('l1', 0))
                all_metrics['MSE'].append(metrics.get('l2', 0))
                all_metrics['PSNR'].append(metrics.get('psnr', 0))
                all_metrics['SSIM'].append(metrics.get('ssim', 0))
                all_metrics['perceptual'].append(metrics.get('perceptual', 0))
                if 'lpips' in metrics:
                    all_metrics['LPIPS'].append(metrics['lpips'])
                
                # Also compute masked-region metrics
                metrics_masked = comprehensive_evaluator.compute_metrics(images, completed, masks)
                all_metrics['L1_masked'].append(metrics_masked.get('l1', 0))
                all_metrics['MSE_masked'].append(metrics_masked.get('l2', 0))
                all_metrics['PSNR_masked'].append(metrics_masked.get('psnr', 0))
                all_metrics['SSIM_masked'].append(metrics_masked.get('ssim', 0))
            else:
                # Fallback to built-in metrics
                # Convert to [0, 1] range for metrics
                images_01 = denormalize(images)
                completed_01 = denormalize(completed)
                reconstructed_01 = denormalize(reconstructed)
                
                # Compute metrics for this batch
                # Full image metrics
                all_metrics['MSE'].append(compute_mse(completed_01, images_01))
                all_metrics['PSNR'].append(compute_psnr(completed_01, images_01))
                all_metrics['SSIM'].append(compute_ssim(completed_01, images_01))
                
                # Masked region only metrics
                all_metrics['MSE_masked'].append(compute_mse(completed_01, images_01, masks))
                all_metrics['PSNR_masked'].append(compute_psnr(reconstructed_01 * masks, images_01 * masks))
                all_metrics['SSIM_masked'].append(compute_ssim(reconstructed_01, images_01))
                
                # LPIPS (perceptual)
                if lpips_model is not None:
                    all_metrics['LPIPS'].append(compute_lpips_metric(completed_01, images_01, lpips_model))
            
            # Collect visualization samples
            if vis_collected < num_vis_samples:
                samples_to_add = min(batch_size, num_vis_samples - vis_collected)
                vis_samples['original'].append(images[:samples_to_add].cpu())
                vis_samples['masked'].append(masked_images[:samples_to_add].cpu())
                vis_samples['reconstructed'].append(reconstructed[:samples_to_add].cpu())
                vis_samples['completed'].append(completed[:samples_to_add].cpu())
                vis_samples['masks'].append(masks[:samples_to_add].cpu())
                vis_collected += samples_to_add
    
    # Aggregate metrics
    final_metrics = {}
    for key, values in all_metrics.items():
        if len(values) > 0:
            final_metrics[key] = float(np.mean(values))
            final_metrics[f'{key}_std'] = float(np.std(values))
    
    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\nFull Image Metrics:")
    if 'L1' in final_metrics:
        print(f"  L1:    {final_metrics.get('L1', 0):.6f} ± {final_metrics.get('L1_std', 0):.6f}")
    print(f"  MSE:   {final_metrics.get('MSE', 0):.6f} ± {final_metrics.get('MSE_std', 0):.6f}")
    print(f"  PSNR:  {final_metrics.get('PSNR', 0):.2f} ± {final_metrics.get('PSNR_std', 0):.2f} dB")
    print(f"  SSIM:  {final_metrics.get('SSIM', 0):.4f} ± {final_metrics.get('SSIM_std', 0):.4f}")
    if 'perceptual' in final_metrics and final_metrics['perceptual'] > 0:
        print(f"  Perceptual (VGG): {final_metrics.get('perceptual', 0):.6f} ± {final_metrics.get('perceptual_std', 0):.6f}")
    if 'LPIPS' in final_metrics and final_metrics['LPIPS'] > 0:
        print(f"  LPIPS: {final_metrics.get('LPIPS', 0):.4f} ± {final_metrics.get('LPIPS_std', 0):.4f}")
    
    print(f"\nMasked Region Metrics:")
    if 'L1_masked' in final_metrics:
        print(f"  L1:    {final_metrics.get('L1_masked', 0):.6f} ± {final_metrics.get('L1_masked_std', 0):.6f}")
    print(f"  MSE:   {final_metrics.get('MSE_masked', 0):.6f} ± {final_metrics.get('MSE_masked_std', 0):.6f}")
    print(f"  PSNR:  {final_metrics.get('PSNR_masked', 0):.2f} ± {final_metrics.get('PSNR_masked_std', 0):.2f} dB")
    
    # Save metrics to JSON
    metrics_path = os.path.join(args.save_dir, 'metrics.json')
    results = {
        'checkpoint': args.checkpoint,
        'model_type': model.model_type,
        'dataset': args.dataset,
        'split': args.split,
        'num_samples': len(dataloader.dataset),
        'mask_type': model.mask_type,
        'mask_size': model.mask_size,
        'image_size': model.image_size,
        'metrics': final_metrics,
        'timestamp': datetime.now().isoformat()
    }
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Concatenate visualization samples
    vis_original = torch.cat(vis_samples['original'], dim=0)
    vis_masked = torch.cat(vis_samples['masked'], dim=0)
    vis_reconstructed = torch.cat(vis_samples['reconstructed'], dim=0)
    vis_completed = torch.cat(vis_samples['completed'], dim=0)
    vis_masks = torch.cat(vis_samples['masks'], dim=0)
    
    # Create evaluation grid
    grid_path = os.path.join(args.save_dir, 'evaluation_grid.png')
    create_evaluation_grid(
        vis_masked, vis_reconstructed, vis_completed, vis_original, vis_masks,
        grid_path, num_samples=min(8, num_vis_samples)
    )
    
    # Create metrics bar chart
    bar_path = os.path.join(args.save_dir, 'metrics_bar.png')
    plot_metrics_bar(final_metrics, bar_path)
    print(f"Metrics bar chart saved to {bar_path}")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"Results saved to: {args.save_dir}")
    print(f"{'='*60}\n")
    
    return final_metrics


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Context Encoder Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model settings
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='auto',
                       choices=['auto', 'baseline', 'staged', 'lion', 'generic'],
                       help='Model type (auto-detect if not specified)')
    parser.add_argument('--model_file', type=str, default='',
                       help='Path to model definition file (for custom architectures)')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'celeba', 'stl10'],
                       help='Dataset to evaluate on')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory for datasets')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to evaluate (0 = all)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Mask settings (override checkpoint config)
    parser.add_argument('--image_size', type=int, default=0,
                       help='Image size (0 = use checkpoint config)')
    parser.add_argument('--mask_size', type=int, default=0,
                       help='Mask size (0 = use checkpoint config)')
    parser.add_argument('--mask_type', type=str, default='',
                       choices=['', 'center', 'random_square'],
                       help='Mask type (empty = use checkpoint config)')
    
    # Output settings
    parser.add_argument('--save_dir', type=str, default='./eval_results',
                       help='Directory to save results')
    parser.add_argument('--num_vis_samples', type=int, default=16,
                       help='Number of samples for visualization')
    
    # Other settings
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = get_args()
    evaluate(args)
