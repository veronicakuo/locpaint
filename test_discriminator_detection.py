"""
Diagnostic script to test what the localizer discriminator is detecting.

Tests whether the discriminator relies on:
1. Boundary discontinuities (edges between real/fake regions)
2. Content/texture differences (blurry vs sharp, different statistics)

Test cases:
- Gaussian blur applied to random 64x64 region (no boundary discontinuity)
- Noise added to random 64x64 region
- Actual reconstructed regions from the generator
- Fully reconstructed images (no original content, no boundary)
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model(checkpoint_path, model_type='localizer', device='cuda'):
    """Load model from checkpoint."""
    
    if model_type == 'localizer':
        from localizer.context_encoder import ContextEncoderWithLocalizer
        
        # Load checkpoint to get config
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get('config', {})
        config['device'] = device
        
        # Create model and load weights
        model = ContextEncoderWithLocalizer(config)
        model.load_checkpoint(checkpoint_path)
        
    elif model_type == 'staged':
        from staged.staged_context_encoder import StagedContextEncoder
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get('config', {})
        config['device'] = device
        
        model = StagedContextEncoder(config)
        model.load_checkpoint(checkpoint_path)
        
        print(f"  Staged model loaded - Current stage: {model.current_stage}")
        print(f"  Transition epoch: {model.transition_epoch}")
    
    return model


def get_discriminator(model, model_type='localizer', which='localizer'):
    """
    Get discriminator from the model.
    
    Args:
        model: The loaded model
        model_type: 'localizer' or 'staged'
        which: For staged models, 'localizer' or 'binary'
    
    Returns:
        discriminator, is_localizer (bool)
    """
    if model_type == 'localizer':
        return model.discriminator, True
    elif model_type == 'staged':
        if which == 'localizer':
            return model.localizer_discriminator, True
        elif which == 'binary':
            return model.binary_discriminator, False
        else:
            raise ValueError(f"Unknown discriminator type: {which}")


def evaluate_binary_discriminator(discriminator, images, device):
    """Run binary discriminator and compute detection metrics."""
    discriminator.eval()
    
    with torch.no_grad():
        # Binary discriminator outputs single confidence per image
        confidence = discriminator(images)
    
    return {
        'confidence': confidence.mean().item(),
        'confidence_std': confidence.std().item(),
        'confidence_all': confidence
    }


def extract_patches_for_binary_d(images, masks, patch_size=64):
    """
    Extract patches from masked regions for binary discriminator.
    Similar to staged_context_encoder.extract_patches().
    """
    batch_size = images.size(0)
    patches = []
    
    for i in range(batch_size):
        mask = masks[i, 0]
        nonzero = torch.nonzero(mask)
        
        if len(nonzero) > 0:
            y_min, x_min = nonzero.min(dim=0)[0]
            y_max, x_max = nonzero.max(dim=0)[0]
            
            # Extract patch
            patch = images[i:i+1, :, y_min:y_max+1, x_min:x_max+1]
            
            # Resize to fixed size
            patch = F.interpolate(patch, size=(patch_size, patch_size), 
                                 mode='bilinear', align_corners=False)
            patches.append(patch)
        else:
            # Fallback: center crop
            h, w = images.shape[2], images.shape[3]
            start = (h - patch_size) // 2
            patch = images[i:i+1, :, start:start+patch_size, start:start+patch_size]
            patches.append(patch)
    
    return torch.cat(patches, dim=0)


def evaluate_binary_discriminator_with_patches(discriminator, images, masks, device, patch_size=64):
    """
    Run binary discriminator on extracted patches (as used in staged training).
    """
    discriminator.eval()
    
    # Extract patches from masked regions
    patches = extract_patches_for_binary_d(images, masks, patch_size)
    
    with torch.no_grad():
        confidence = discriminator(patches)
    
    return {
        'confidence': confidence.mean().item(),
        'confidence_std': confidence.std().item(),
        'confidence_all': confidence
    }


def apply_gaussian_blur_region(images, mask_size=64, sigma=5.0):
    """
    Apply Gaussian blur to a random square region.
    Returns blurred images, masks, and bounding boxes.
    
    This creates NO boundary discontinuity - just smoothly blurred content.
    """
    batch_size, c, h, w = images.shape
    device = images.device
    
    # Create Gaussian kernel
    kernel_size = int(6 * sigma + 1) | 1  # Ensure odd
    x = torch.arange(kernel_size, device=device) - kernel_size // 2
    gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
    gaussian_2d = gaussian_2d.expand(c, 1, kernel_size, kernel_size)
    
    blurred_images = images.clone()
    masks = torch.zeros(batch_size, 1, h, w, device=device)
    bboxes = []
    
    for i in range(batch_size):
        # Random position
        max_pos = h - mask_size
        x = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
        y = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
        
        # Apply Gaussian blur to the region
        # Pad for convolution
        pad = kernel_size // 2
        img_padded = F.pad(images[i:i+1], (pad, pad, pad, pad), mode='reflect')
        blurred_full = F.conv2d(img_padded, gaussian_2d, groups=c)
        
        # Replace only the masked region with blurred content
        blurred_images[i, :, y:y+mask_size, x:x+mask_size] = \
            blurred_full[0, :, y:y+mask_size, x:x+mask_size]
        
        masks[i, :, y:y+mask_size, x:x+mask_size] = 1
        
        bbox = torch.tensor([
            x / w, y / h,
            (x + mask_size) / w, (y + mask_size) / h
        ])
        bboxes.append(bbox)
    
    bboxes = torch.stack(bboxes).to(device)
    return blurred_images, masks, bboxes


def apply_noise_region(images, mask_size=64, noise_std=0.3):
    """
    Add Gaussian noise to a random square region.
    """
    batch_size, c, h, w = images.shape
    device = images.device
    
    noisy_images = images.clone()
    masks = torch.zeros(batch_size, 1, h, w, device=device)
    bboxes = []
    
    for i in range(batch_size):
        max_pos = h - mask_size
        x = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
        y = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
        
        # Add noise to region
        noise = torch.randn(c, mask_size, mask_size, device=device) * noise_std
        noisy_images[i, :, y:y+mask_size, x:x+mask_size] += noise
        noisy_images = noisy_images.clamp(-1, 1)
        
        masks[i, :, y:y+mask_size, x:x+mask_size] = 1
        
        bbox = torch.tensor([
            x / w, y / h,
            (x + mask_size) / w, (y + mask_size) / h
        ])
        bboxes.append(bbox)
    
    bboxes = torch.stack(bboxes).to(device)
    return noisy_images, masks, bboxes


def apply_mean_color_region(images, mask_size=64):
    """
    Replace a random square region with its mean color.
    No boundary discontinuity in terms of edges, but uniform color.
    """
    batch_size, c, h, w = images.shape
    device = images.device
    
    modified_images = images.clone()
    masks = torch.zeros(batch_size, 1, h, w, device=device)
    bboxes = []
    
    for i in range(batch_size):
        max_pos = h - mask_size
        x = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
        y = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
        
        # Get mean color of the region
        region = images[i, :, y:y+mask_size, x:x+mask_size]
        mean_color = region.mean(dim=(1, 2), keepdim=True)
        
        # Replace with mean color
        modified_images[i, :, y:y+mask_size, x:x+mask_size] = mean_color.expand(c, mask_size, mask_size)
        
        masks[i, :, y:y+mask_size, x:x+mask_size] = 1
        
        bbox = torch.tensor([
            x / w, y / h,
            (x + mask_size) / w, (y + mask_size) / h
        ])
        bboxes.append(bbox)
    
    bboxes = torch.stack(bboxes).to(device)
    return modified_images, masks, bboxes


def get_reconstructed_images(model, images, model_type='localizer'):
    """Get completed images using the generator (with boundary discontinuity)."""
    batch_size = images.size(0)
    
    # Generate masks
    masks, bboxes = model.generate_square_mask(batch_size)
    
    # Mask and reconstruct
    masked = images * (1 - masks)
    
    model.encoder.eval()
    model.decoder.eval()
    
    with torch.no_grad():
        encoded = model.encoder(masked)
        reconstructed = model.decoder(encoded)
        
        if reconstructed.size(2) != images.size(2):
            reconstructed = F.interpolate(
                reconstructed, size=images.shape[2:],
                mode='bilinear', align_corners=False
            )
    
    # Completed = masked original + reconstructed region (HAS BOUNDARY)
    completed = masked + reconstructed * masks
    
    # Fully reconstructed = just the reconstruction (NO BOUNDARY)
    # We paste reconstructed content but with smooth blending at edges
    fully_reconstructed = reconstructed
    
    return completed, fully_reconstructed, masks, bboxes


def compute_iou(pred_bbox, true_bbox):
    """Compute IoU between predicted and true bounding boxes."""
    # pred_bbox: [batch, num_boxes, 4] or [batch, 4]
    # true_bbox: [batch, 4]
    
    if pred_bbox.dim() == 3:
        pred_bbox = pred_bbox.squeeze(1)
    
    # Intersection
    x1 = torch.max(pred_bbox[:, 0], true_bbox[:, 0])
    y1 = torch.max(pred_bbox[:, 1], true_bbox[:, 1])
    x2 = torch.min(pred_bbox[:, 2], true_bbox[:, 2])
    y2 = torch.min(pred_bbox[:, 3], true_bbox[:, 3])
    
    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    intersection = inter_w * inter_h
    
    # Union
    pred_area = (pred_bbox[:, 2] - pred_bbox[:, 0]) * (pred_bbox[:, 3] - pred_bbox[:, 1])
    true_area = (true_bbox[:, 2] - true_bbox[:, 0]) * (true_bbox[:, 3] - true_bbox[:, 1])
    union = pred_area + true_area - intersection + 1e-7
    
    return intersection / union


def evaluate_detection(discriminator, images, true_bboxes, device):
    """Run discriminator and compute detection metrics."""
    discriminator.eval()
    
    with torch.no_grad():
        pred_bboxes, confidence = discriminator(images)
    
    iou = compute_iou(pred_bboxes, true_bboxes)
    
    return {
        'confidence': confidence.mean().item(),
        'iou': iou.mean().item(),
        'iou_std': iou.std().item(),
        'pred_bboxes': pred_bboxes,
        'confidence_all': confidence
    }


def visualize_results(images, pred_bboxes, true_bboxes, confidence, 
                      title, save_path, num_samples=8):
    """Visualize images with predicted and true bounding boxes."""
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(images))):
        ax = axes[i]
        
        # Denormalize image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        
        ax.imshow(img)
        
        h, w = images.shape[2], images.shape[3]
        
        # True bbox (green)
        tb = true_bboxes[i].cpu().numpy()
        rect_true = plt.Rectangle(
            (tb[0] * w, tb[1] * h),
            (tb[2] - tb[0]) * w, (tb[3] - tb[1]) * h,
            fill=False, color='green', linewidth=2, label='True'
        )
        ax.add_patch(rect_true)
        
        # Predicted bbox (red)
        pb = pred_bboxes[i].squeeze().cpu().numpy()
        rect_pred = plt.Rectangle(
            (pb[0] * w, pb[1] * h),
            (pb[2] - pb[0]) * w, (pb[3] - pb[1]) * h,
            fill=False, color='red', linewidth=2, linestyle='--', label='Pred'
        )
        ax.add_patch(rect_pred)
        
        # Confidence and IoU
        conf = confidence[i].item() if confidence.dim() > 1 else confidence[i].item()
        iou = compute_iou(pred_bboxes[i:i+1], true_bboxes[i:i+1]).item()
        ax.set_title(f'Conf: {conf:.2f}, IoU: {iou:.2f}', fontsize=10)
        ax.axis('off')
    
    # Add legend
    axes[0].legend(loc='upper left', fontsize=8)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def run_diagnostic(args):
    """Run full diagnostic suite."""
    
    device = torch.device(args.device)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.model_type, device)
    discriminator, is_localizer = get_discriminator(model, args.model_type, 'localizer')
    
    # For staged models, also get binary discriminator
    binary_discriminator = None
    if args.model_type == 'staged':
        binary_discriminator, _ = get_discriminator(model, args.model_type, 'binary')
        print(f"  Testing BOTH discriminators (binary + localizer)")
    
    # Load CIFAR-10 train set
    print("Loading CIFAR-10 train set...")
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=transform
    )
    
    # Use subset for testing
    np.random.seed(args.seed)
    indices = np.random.choice(len(dataset), args.num_samples, replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)
    
    # Get a batch of images
    images, _ = next(iter(loader))
    images = images.to(device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    binary_results = {} if binary_discriminator else None
    
    # =========================================================================
    # Test 1: Real images (no modification) - should have LOW confidence
    # =========================================================================
    print("\n" + "="*60)
    print("Test 1: Real images (unmodified)")
    print("="*60)
    
    discriminator.eval()
    with torch.no_grad():
        pred_bboxes, confidence = discriminator(images)
    
    results['real'] = {
        'confidence': confidence.mean().item(),
        'confidence_std': confidence.std().item()
    }
    print(f"  [Localizer] Confidence: {results['real']['confidence']:.4f} ± {results['real']['confidence_std']:.4f}")
    print(f"  (Should be LOW ~0 for real images)")
    
    if binary_discriminator:
        # For real images, create dummy center masks for patch extraction
        dummy_masks = torch.zeros(images.size(0), 1, args.image_size, args.image_size, device=device)
        center = (args.image_size - args.mask_size) // 2
        dummy_masks[:, :, center:center+args.mask_size, center:center+args.mask_size] = 1
        
        binary_metrics = evaluate_binary_discriminator_with_patches(
            binary_discriminator, images, dummy_masks, device, patch_size=args.mask_size
        )
        binary_results['real'] = binary_metrics
        print(f"  [Binary D]  Confidence: {binary_metrics['confidence']:.4f} ± {binary_metrics['confidence_std']:.4f}")
    
    # =========================================================================
    # Test 2: Gaussian blur regions (NO boundary discontinuity)
    # =========================================================================
    print("\n" + "="*60)
    print("Test 2: Gaussian blur regions (NO boundary discontinuity)")
    print("="*60)
    
    for sigma in [3.0, 5.0, 10.0, 20.0]:
        print(f"\n  Sigma = {sigma}:")
        
        blurred, masks, true_bboxes = apply_gaussian_blur_region(
            images, mask_size=args.mask_size, sigma=sigma
        )
        
        metrics = evaluate_detection(discriminator, blurred, true_bboxes, device)
        results[f'blur_sigma_{sigma}'] = metrics
        
        print(f"    [Localizer] Confidence: {metrics['confidence']:.4f}, IoU: {metrics['iou']:.4f} ± {metrics['iou_std']:.4f}")
        
        if binary_discriminator:
            binary_metrics = evaluate_binary_discriminator_with_patches(
                binary_discriminator, blurred, masks, device, patch_size=args.mask_size
            )
            binary_results[f'blur_sigma_{sigma}'] = binary_metrics
            print(f"    [Binary D]  Confidence: {binary_metrics['confidence']:.4f}")
        
        # Visualize
        visualize_results(
            blurred, metrics['pred_bboxes'], true_bboxes, metrics['confidence_all'],
            f'Gaussian Blur (σ={sigma}) - No Boundary',
            os.path.join(args.output_dir, f'blur_sigma_{sigma}.png')
        )
    
    # =========================================================================
    # Test 3: Noise regions
    # =========================================================================
    print("\n" + "="*60)
    print("Test 3: Noise regions")
    print("="*60)
    
    for noise_std in [0.1, 0.3, 0.5]:
        print(f"\n  Noise std = {noise_std}:")
        
        noisy, masks, true_bboxes = apply_noise_region(
            images, mask_size=args.mask_size, noise_std=noise_std
        )
        
        metrics = evaluate_detection(discriminator, noisy, true_bboxes, device)
        results[f'noise_std_{noise_std}'] = metrics
        
        print(f"    [Localizer] Confidence: {metrics['confidence']:.4f}, IoU: {metrics['iou']:.4f} ± {metrics['iou_std']:.4f}")
        
        if binary_discriminator:
            binary_metrics = evaluate_binary_discriminator_with_patches(
                binary_discriminator, noisy, masks, device, patch_size=args.mask_size
            )
            binary_results[f'noise_std_{noise_std}'] = binary_metrics
            print(f"    [Binary D]  Confidence: {binary_metrics['confidence']:.4f}")
        
        visualize_results(
            noisy, metrics['pred_bboxes'], true_bboxes, metrics['confidence_all'],
            f'Gaussian Noise (std={noise_std})',
            os.path.join(args.output_dir, f'noise_std_{noise_std}.png')
        )
    
    # =========================================================================
    # Test 4: Mean color regions
    # =========================================================================
    print("\n" + "="*60)
    print("Test 4: Mean color regions (uniform color patch)")
    print("="*60)
    
    mean_color_imgs, masks, true_bboxes = apply_mean_color_region(
        images, mask_size=args.mask_size
    )
    
    metrics = evaluate_detection(discriminator, mean_color_imgs, true_bboxes, device)
    results['mean_color'] = metrics
    
    print(f"  [Localizer] Confidence: {metrics['confidence']:.4f}, IoU: {metrics['iou']:.4f} ± {metrics['iou_std']:.4f}")
    
    if binary_discriminator:
        binary_metrics = evaluate_binary_discriminator_with_patches(
            binary_discriminator, mean_color_imgs, masks, device, patch_size=args.mask_size
        )
        binary_results['mean_color'] = binary_metrics
        print(f"  [Binary D]  Confidence: {binary_metrics['confidence']:.4f}")
    
    visualize_results(
        mean_color_imgs, metrics['pred_bboxes'], true_bboxes, metrics['confidence_all'],
        'Mean Color Regions',
        os.path.join(args.output_dir, 'mean_color.png')
    )
    
    # =========================================================================
    # Test 5: Actual reconstructions (WITH boundary discontinuity)
    # =========================================================================
    print("\n" + "="*60)
    print("Test 5: Actual reconstructions (WITH boundary)")
    print("="*60)
    
    completed, fully_recon, masks, true_bboxes = get_reconstructed_images(
        model, images, args.model_type
    )
    
    # 5a: Completed images (original + reconstructed region)
    metrics_completed = evaluate_detection(discriminator, completed, true_bboxes, device)
    results['completed'] = metrics_completed
    
    print(f"\n  Completed (masked original + reconstructed patch):")
    print(f"    [Localizer] Confidence: {metrics_completed['confidence']:.4f}, IoU: {metrics_completed['iou']:.4f} ± {metrics_completed['iou_std']:.4f}")
    
    if binary_discriminator:
        binary_metrics = evaluate_binary_discriminator_with_patches(
            binary_discriminator, completed, masks, device, patch_size=args.mask_size
        )
        binary_results['completed'] = binary_metrics
        print(f"    [Binary D]  Confidence: {binary_metrics['confidence']:.4f}")
    
    visualize_results(
        completed, metrics_completed['pred_bboxes'], true_bboxes, metrics_completed['confidence_all'],
        'Completed Images (WITH Boundary)',
        os.path.join(args.output_dir, 'completed.png')
    )
    
    # 5b: Fully reconstructed (entire image through generator)
    metrics_fully = evaluate_detection(discriminator, fully_recon, true_bboxes, device)
    results['fully_reconstructed'] = metrics_fully
    
    print(f"\n  Fully reconstructed (entire image through G, NO boundary):")
    print(f"    [Localizer] Confidence: {metrics_fully['confidence']:.4f}, IoU: {metrics_fully['iou']:.4f} ± {metrics_fully['iou_std']:.4f}")
    
    if binary_discriminator:
        binary_metrics = evaluate_binary_discriminator_with_patches(
            binary_discriminator, fully_recon, masks, device, patch_size=args.mask_size
        )
        binary_results['fully_reconstructed'] = binary_metrics
        print(f"    [Binary D]  Confidence: {binary_metrics['confidence']:.4f}")
    
    visualize_results(
        fully_recon, metrics_fully['pred_bboxes'], true_bboxes, metrics_fully['confidence_all'],
        'Fully Reconstructed (NO Boundary)',
        os.path.join(args.output_dir, 'fully_reconstructed.png')
    )
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY - LOCALIZER DISCRIMINATOR")
    print("="*60)
    
    print("\nConfidence scores (higher = more confident there's a fake region):")
    print(f"  Real images:        {results['real']['confidence']:.4f}")
    print(f"  Blur (σ=5):         {results['blur_sigma_5.0']['confidence']:.4f}")
    print(f"  Blur (σ=20):        {results['blur_sigma_20.0']['confidence']:.4f}")
    print(f"  Noise (std=0.3):    {results['noise_std_0.3']['confidence']:.4f}")
    print(f"  Mean color:         {results['mean_color']['confidence']:.4f}")
    print(f"  Completed (w/ boundary): {results['completed']['confidence']:.4f}")
    print(f"  Fully recon (no boundary): {results['fully_reconstructed']['confidence']:.4f}")
    
    print("\nIoU scores (higher = better localization):")
    print(f"  Blur (σ=5):         {results['blur_sigma_5.0']['iou']:.4f}")
    print(f"  Blur (σ=20):        {results['blur_sigma_20.0']['iou']:.4f}")
    print(f"  Noise (std=0.3):    {results['noise_std_0.3']['iou']:.4f}")
    print(f"  Mean color:         {results['mean_color']['iou']:.4f}")
    print(f"  Completed:          {results['completed']['iou']:.4f}")
    print(f"  Fully recon:        {results['fully_reconstructed']['iou']:.4f}")
    
    if binary_results:
        print("\n" + "="*60)
        print("SUMMARY - BINARY DISCRIMINATOR (Staged Model)")
        print("="*60)
        
        print("\nConfidence scores (higher = thinks image is REAL):")
        print(f"  Real images:        {binary_results['real']['confidence']:.4f}")
        print(f"  Blur (σ=5):         {binary_results['blur_sigma_5.0']['confidence']:.4f}")
        print(f"  Blur (σ=20):        {binary_results['blur_sigma_20.0']['confidence']:.4f}")
        print(f"  Noise (std=0.3):    {binary_results['noise_std_0.3']['confidence']:.4f}")
        print(f"  Mean color:         {binary_results['mean_color']['confidence']:.4f}")
        print(f"  Completed:          {binary_results['completed']['confidence']:.4f}")
        print(f"  Fully recon:        {binary_results['fully_reconstructed']['confidence']:.4f}")
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    blur_iou = results['blur_sigma_5.0']['iou']
    completed_iou = results['completed']['iou']
    
    if blur_iou > 0.5:
        print("\n⚠️  HIGH blur IoU suggests discriminator is NOT relying on boundaries!")
        print("    It's detecting content/texture differences (blur vs sharp).")
        print("    This means it may be 'cheating' by detecting low-frequency artifacts.")
    else:
        print("\n✓  LOW blur IoU suggests discriminator IS relying on boundaries.")
        print("    It can't localize regions that are just blurred.")
    
    if results['fully_reconstructed']['iou'] > 0.5:
        print("\n⚠️  Can localize fully reconstructed images (no boundary)!")
        print("    Discriminator detects generator artifacts, not just boundaries.")
    
    if binary_results:
        real_conf = binary_results['real']['confidence']
        fake_conf = binary_results['completed']['confidence']
        if real_conf > 0.7 and fake_conf < 0.3:
            print("\n✓  Binary discriminator correctly distinguishes real vs completed")
        elif abs(real_conf - fake_conf) < 0.2:
            print("\n⚠️  Binary discriminator struggling to distinguish real vs fake")
    
    # Save results
    import json
    results_serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            results_serializable[k] = {
                kk: vv for kk, vv in v.items() 
                if not isinstance(vv, torch.Tensor)
            }
        else:
            results_serializable[k] = v
    
    if binary_results:
        results_serializable['binary_discriminator'] = {}
        for k, v in binary_results.items():
            if isinstance(v, dict):
                results_serializable['binary_discriminator'][k] = {
                    kk: vv for kk, vv in v.items() 
                    if not isinstance(vv, torch.Tensor)
                }
    
    results_path = os.path.join(args.output_dir, 'diagnostic_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Diagnostic: What is the discriminator detecting?')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='localizer',
                       choices=['localizer', 'staged'],
                       help='Type of model')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--mask_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=32,
                       help='Number of images to test')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./diagnostic_results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    run_diagnostic(args)


if __name__ == "__main__":
    main()
