"""
Model Comparison Visualization Script

Generates side-by-side comparison of inpainting results across different models:
- L2 reconstruction only (no adversarial)
- Baseline (Pathak et al.) with binary discriminator
- Localizer discriminator 
- Staged/Phased localizer

Uses 10 images from CIFAR-10 validation set, one per class, with fixed seed
for reproducibility across runs.
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

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def load_model(checkpoint_path, model_type, device='cuda'):
    """Load model from checkpoint."""
    
    if checkpoint_path is None:
        return None
    
    if not os.path.exists(checkpoint_path):
        print(f"  Warning: Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        if model_type == 'baseline':
            from baseline.baseline_context_encoder import BaselineContextEncoder
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            config = checkpoint.get('config', {})
            config['device'] = device
            
            model = BaselineContextEncoder(config)
            model.load_checkpoint(checkpoint_path)
            
        elif model_type == 'localizer':
            from localizer.context_encoder import ContextEncoderWithLocalizer
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            config = checkpoint.get('config', {})
            config['device'] = device
            
            model = ContextEncoderWithLocalizer(config)
            model.load_checkpoint(checkpoint_path)
            
        elif model_type == 'staged':
            from staged.staged_context_encoder import StagedContextEncoder
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            config = checkpoint.get('config', {})
            config['device'] = device
            
            model = StagedContextEncoder(config)
            model.load_checkpoint(checkpoint_path)
            
        elif model_type == 'l2_only':
            # L2-only model uses same architecture as baseline but trained without adversarial
            from baseline.baseline_context_encoder import BaselineContextEncoder
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            config = checkpoint.get('config', {})
            config['device'] = device
            
            model = BaselineContextEncoder(config)
            model.load_checkpoint(checkpoint_path)
        
        print(f"  Loaded {model_type} model from {checkpoint_path}")
        return model
        
    except Exception as e:
        print(f"  Error loading {model_type} model: {e}")
        return None


def get_one_per_class_indices(dataset, seed=27):
    """
    Get indices for one image per class in CIFAR-10.
    Uses fixed seed for reproducibility.
    
    Returns:
        indices: List of 10 indices (one per class)
        labels: Corresponding class labels
    """
    np.random.seed(seed)
    
    # Group indices by class
    class_indices = {i: [] for i in range(10)}
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[label].append(idx)
    
    # Select one random image per class
    selected_indices = []
    selected_labels = []
    
    for class_id in range(10):
        indices = class_indices[class_id]
        selected_idx = np.random.choice(indices)
        selected_indices.append(selected_idx)
        selected_labels.append(class_id)
    
    return selected_indices, selected_labels


def generate_fixed_masks(batch_size, image_size, mask_size, seed=123):
    """Generate fixed masks with reproducible positions."""
    np.random.seed(seed)
    
    masks = torch.zeros(batch_size, 1, image_size, image_size)
    bboxes = []
    
    for i in range(batch_size):
        max_pos = image_size - mask_size
        x = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
        y = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
        
        masks[i, :, y:y+mask_size, x:x+mask_size] = 1
        
        bbox = [x / image_size, y / image_size,
                (x + mask_size) / image_size, (y + mask_size) / image_size]
        bboxes.append(bbox)
    
    return masks, bboxes


def complete_images(model, images, masks, device):
    """
    Generate completed images using a model.
    
    Returns:
        completed: Images with reconstructed regions pasted in
    """
    if model is None:
        return None
    
    model.encoder.eval()
    model.decoder.eval()
    
    images = images.to(device)
    masks = masks.to(device)
    
    with torch.no_grad():
        # Mask images
        masked = images * (1 - masks)
        
        # Encode and decode
        encoded = model.encoder(masked)
        reconstructed = model.decoder(encoded)
        
        # Resize if needed
        if reconstructed.size(2) != images.size(2):
            reconstructed = F.interpolate(
                reconstructed,
                size=images.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Create completed images
        completed = masked + reconstructed * masks
    
    return completed


def denormalize(tensor):
    """Convert from [-1, 1] to [0, 1] range."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def create_comparison_figure(images, masked, completions_dict, class_labels, 
                             masks, save_path, title="Model Comparison"):
    """
    Create comparison visualization.
    
    Args:
        images: Original images [N, C, H, W]
        masked: Masked images [N, C, H, W]
        completions_dict: Dict of {model_name: completed_images}
        class_labels: List of class indices
        masks: Mask tensors [N, 1, H, W]
        save_path: Where to save the figure
        title: Figure title
    """
    n_images = len(images)
    
    # Column order
    model_names = ['L2 Only', 'Pathak et al.', 'LocPaint', 'Staged LocPaint']
    model_keys = ['l2_only', 'baseline', 'localizer', 'staged']
    
    # Count available models
    available_models = [k for k in model_keys if k in completions_dict and completions_dict[k] is not None]
    n_cols = 2 + len(available_models)  # Original, Masked, + models
    
    fig, axes = plt.subplots(n_images, n_cols, figsize=(2.5 * n_cols, 2.5 * n_images))
    
    # Make sure axes is 2D
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    # Column headers
    col_titles = ['Original', 'Masked']
    for key in model_keys:
        if key in completions_dict and completions_dict[key] is not None:
            idx = model_keys.index(key)
            col_titles.append(model_names[idx])
    
    for row in range(n_images):
        class_name = CIFAR10_CLASSES[class_labels[row]]
        
        col = 0
        
        # Original image
        img = denormalize(images[row]).cpu().permute(1, 2, 0).numpy()
        axes[row, col].imshow(img)
        axes[row, col].set_ylabel(class_name, fontsize=10, fontweight='bold')
        if row == 0:
            axes[row, col].set_title(col_titles[col], fontsize=11, fontweight='bold')
        axes[row, col].axis('off')
        col += 1
        
        # Masked image
        img_masked = denormalize(masked[row]).cpu().permute(1, 2, 0).numpy()
        axes[row, col].imshow(img_masked)
        if row == 0:
            axes[row, col].set_title(col_titles[col], fontsize=11, fontweight='bold')
        axes[row, col].axis('off')
        col += 1
        
        # Model completions
        title_idx = 2
        for key in model_keys:
            if key not in completions_dict or completions_dict[key] is None:
                continue
            
            completed = completions_dict[key]
            img_completed = denormalize(completed[row]).cpu().permute(1, 2, 0).numpy()
            axes[row, col].imshow(img_completed)
            if row == 0:
                axes[row, col].set_title(col_titles[title_idx], fontsize=11, fontweight='bold')
            axes[row, col].axis('off')
            col += 1
            title_idx += 1
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Saved comparison figure to {save_path}")


def create_detailed_comparison(images, masked, completions_dict, class_labels,
                               masks, save_path):
    """
    Create detailed comparison with zoomed inpainted regions.
    """
    n_images = len(images)
    model_keys = ['l2_only', 'baseline', 'localizer', 'staged']
    model_names = ['L2 Only', 'Baseline', 'Localizer', 'Staged']
    
    available_models = [k for k in model_keys if k in completions_dict and completions_dict[k] is not None]
    
    if len(available_models) == 0:
        print("No models available for detailed comparison")
        return
    
    # For each image, show: original, masked, and zoomed comparisons
    fig, axes = plt.subplots(n_images, 2 + len(available_models), 
                             figsize=(2.5 * (2 + len(available_models)), 2.5 * n_images))
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for row in range(n_images):
        class_name = CIFAR10_CLASSES[class_labels[row]]
        mask = masks[row, 0].cpu().numpy()
        
        # Find mask bounding box
        nonzero = np.nonzero(mask)
        if len(nonzero[0]) > 0:
            y_min, y_max = nonzero[0].min(), nonzero[0].max()
            x_min, x_max = nonzero[1].min(), nonzero[1].max()
            # Add small padding
            pad = 5
            y_min = max(0, y_min - pad)
            y_max = min(mask.shape[0], y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(mask.shape[1], x_max + pad)
        else:
            y_min, y_max = 0, mask.shape[0]
            x_min, x_max = 0, mask.shape[1]
        
        col = 0
        
        # Original (zoomed to mask region)
        img = denormalize(images[row]).cpu().permute(1, 2, 0).numpy()
        axes[row, col].imshow(img[y_min:y_max, x_min:x_max])
        axes[row, col].set_ylabel(class_name, fontsize=10, fontweight='bold')
        if row == 0:
            axes[row, col].set_title('Ground Truth\n(zoomed)', fontsize=10, fontweight='bold')
        axes[row, col].axis('off')
        col += 1
        
        # Masked (zoomed)
        img_masked = denormalize(masked[row]).cpu().permute(1, 2, 0).numpy()
        axes[row, col].imshow(img_masked[y_min:y_max, x_min:x_max])
        if row == 0:
            axes[row, col].set_title('Masked\n(zoomed)', fontsize=10, fontweight='bold')
        axes[row, col].axis('off')
        col += 1
        
        # Model completions (zoomed)
        for key in model_keys:
            if key not in completions_dict or completions_dict[key] is None:
                continue
            
            idx = model_keys.index(key)
            completed = completions_dict[key]
            img_completed = denormalize(completed[row]).cpu().permute(1, 2, 0).numpy()
            axes[row, col].imshow(img_completed[y_min:y_max, x_min:x_max])
            if row == 0:
                axes[row, col].set_title(f'{model_names[idx]}\n(zoomed)', fontsize=10, fontweight='bold')
            axes[row, col].axis('off')
            col += 1
    
    plt.suptitle('Zoomed Comparison of Inpainted Regions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path_zoomed = save_path.replace('.png', '_zoomed.png')
    os.makedirs(os.path.dirname(save_path_zoomed), exist_ok=True)
    plt.savefig(save_path_zoomed, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Saved zoomed comparison to {save_path_zoomed}")


def main():
    parser = argparse.ArgumentParser(description='Model Comparison Visualization')
    
    # Model checkpoints
    parser.add_argument('--l2_checkpoint', type=str, default=None,
                       help='Path to L2-only model checkpoint')
    parser.add_argument('--baseline_checkpoint', type=str, default=None,
                       help='Path to baseline (Pathak) model checkpoint')
    parser.add_argument('--localizer_checkpoint', type=str, default=None,
                       help='Path to localizer model checkpoint')
    parser.add_argument('--staged_checkpoint', type=str, default=None,
                       help='Path to staged model checkpoint')
    
    # Data settings
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--mask_size', type=int, default=64)
    
    # Seeds for reproducibility
    parser.add_argument('--image_seed', type=int, default=42,
                       help='Seed for selecting images (one per class)')
    parser.add_argument('--mask_seed', type=int, default=123,
                       help='Seed for mask positions')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./comparison_results')
    parser.add_argument('--output_name', type=str, default='model_comparison.png')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_train', action='store_true',
                       help='Use training set instead of test set')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load CIFAR-10
    print("\nLoading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = datasets.CIFAR10(
        root=args.data_root,
        train=args.use_train,
        download=True,
        transform=transform
    )
    
    # Get one image per class
    print(f"\nSelecting one image per class (seed={args.image_seed})...")
    indices, labels = get_one_per_class_indices(dataset, seed=args.image_seed)
    
    print("Selected images:")
    for idx, label in zip(indices, labels):
        print(f"  Class {label} ({CIFAR10_CLASSES[label]}): index {idx}")
    
    # Load images
    images = []
    for idx in indices:
        img, _ = dataset[idx]
        images.append(img)
    images = torch.stack(images)
    
    # Generate fixed masks
    print(f"\nGenerating masks (seed={args.mask_seed})...")
    masks, bboxes = generate_fixed_masks(
        len(images), args.image_size, args.mask_size, seed=args.mask_seed
    )
    
    # Create masked images
    masked = images * (1 - masks)
    
    # Load models
    print("\nLoading models...")
    models = {}
    
    if args.l2_checkpoint:
        print("  Loading L2-only model...")
        models['l2_only'] = load_model(args.l2_checkpoint, 'l2_only', device)
    
    if args.baseline_checkpoint:
        print("  Loading baseline model...")
        models['baseline'] = load_model(args.baseline_checkpoint, 'baseline', device)
    
    if args.localizer_checkpoint:
        print("  Loading localizer model...")
        models['localizer'] = load_model(args.localizer_checkpoint, 'localizer', device)
    
    if args.staged_checkpoint:
        print("  Loading staged model...")
        models['staged'] = load_model(args.staged_checkpoint, 'staged', device)
    
    if len(models) == 0:
        print("\nError: No models loaded! Please provide at least one checkpoint.")
        print("Example usage:")
        print("  python compare_models.py \\")
        print("    --baseline_checkpoint ./baseline/checkpoints/best_model.pth \\")
        print("    --localizer_checkpoint ./localizer/checkpoints/best_model.pth \\")
        print("    --staged_checkpoint ./staged/checkpoints/best_model.pth")
        return
    
    # Generate completions
    print("\nGenerating completions...")
    completions = {}
    
    for name, model in models.items():
        if model is not None:
            print(f"  {name}...")
            completions[name] = complete_images(model, images, masks, device)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Main comparison figure
    save_path = os.path.join(args.output_dir, args.output_name)
    create_comparison_figure(
        images, masked, completions, labels, masks, save_path,
        title=f"Inpainting Model Comparison (CIFAR-10, {args.mask_size}x{args.mask_size} mask)"
    )
    
    # Zoomed comparison
    create_detailed_comparison(
        images, masked, completions, labels, masks, save_path
    )
    
    # Save metadata
    metadata = {
        'image_seed': args.image_seed,
        'mask_seed': args.mask_seed,
        'image_size': args.image_size,
        'mask_size': args.mask_size,
        'selected_indices': indices,
        'class_labels': labels,
        'class_names': [CIFAR10_CLASSES[l] for l in labels],
        'models_loaded': list(completions.keys()),
        'checkpoints': {
            'l2_only': args.l2_checkpoint,
            'baseline': args.baseline_checkpoint,
            'localizer': args.localizer_checkpoint,
            'staged': args.staged_checkpoint
        }
    }
    
    import json
    metadata_path = os.path.join(args.output_dir, 'comparison_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDone! Results saved to {args.output_dir}/")
    print(f"  - {args.output_name}")
    print(f"  - {args.output_name.replace('.png', '_zoomed.png')}")
    print(f"  - comparison_metadata.json")


if __name__ == "__main__":
    main()
