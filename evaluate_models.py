"""
Model Evaluation Script

Evaluates all models (L2-only, Baseline, Localizer, Staged) on the same
train/val/test split using consistent seeds.

Metrics: MSE (L2), PSNR, SSIM, LPIPS
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_split import get_celeba_splits
from utils.metrics import calculate_metrics


def load_model(checkpoint_path, model_type, device='cuda'):
    """Load model from checkpoint."""
    
    if checkpoint_path is None:
        return None
    
    if not os.path.exists(checkpoint_path):
        print(f"  Warning: Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        if model_type == 'baseline' or model_type == 'l2_only':
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
        
        print(f"  ✓ Loaded {model_type} from {checkpoint_path}")
        return model
        
    except Exception as e:
        print(f"  ✗ Error loading {model_type}: {e}")
        return None


def generate_masks(batch_size, image_size, mask_size, mask_type='random_square', seed=None):
    """Generate masks for evaluation."""
    if seed is not None:
        np.random.seed(seed)
    
    masks = torch.zeros(batch_size, 1, image_size, image_size)
    
    for i in range(batch_size):
        if mask_type == 'center':
            start = (image_size - mask_size) // 2
            masks[i, :, start:start+mask_size, start:start+mask_size] = 1
        else:  # random_square
            max_pos = image_size - mask_size
            x = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
            y = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
            masks[i, :, y:y+mask_size, x:x+mask_size] = 1
    
    return masks


def evaluate_model(model, dataloader, device, image_size=128, mask_size=64, 
                   mask_type='random_square', mask_seed=None, max_batches=None):
    """
    Evaluate a model on a dataset.
    
    Returns:
        Dictionary with mse, psnr, ssim, lpips (mean and std)
    """
    if model is None:
        return None
    
    model.encoder.eval()
    model.decoder.eval()
    
    all_metrics = {
        'mse': [],
        'psnr': [],
        'ssim': [],
        'lpips': []
    }
    
    # Set seed for reproducible masks across models
    if mask_seed is not None:
        np.random.seed(mask_seed)
    
    num_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, total=num_batches, desc="Evaluating")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            images = images.to(device)
            batch_size = images.size(0)
            
            # Generate masks (same seed means same masks for all models)
            masks = generate_masks(batch_size, image_size, mask_size, mask_type)
            masks = masks.to(device)
            
            # Mask images
            masked = images * (1 - masks)
            
            # Generate reconstruction
            encoded = model.encoder(masked)
            reconstructed = model.decoder(encoded)
            
            # Resize if needed
            if reconstructed.size(2) != image_size:
                reconstructed = F.interpolate(
                    reconstructed,
                    size=(image_size, image_size),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Create completed images
            completed = masked + reconstructed * masks
            
            # Calculate metrics
            batch_metrics = calculate_metrics(images, completed, masks, device=device)
            
            for key in all_metrics:
                all_metrics[key].append(batch_metrics[key])
    
    # Aggregate metrics
    results = {}
    for key in all_metrics:
        values = all_metrics[key]
        results[f'{key}_mean'] = np.mean(values)
        results[f'{key}_std'] = np.std(values)
    
    return results


def evaluate_all_models(args):
    """Evaluate all models and compare."""
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset with consistent split
    print(f"\nLoading dataset with split_seed={args.split_seed}...")
    
    if args.dataset == 'celeba':
        train_dataset, val_dataset, test_dataset, split_info = get_celeba_splits(
            data_root=args.data_root,
            image_size=args.image_size,
            split_seed=args.split_seed,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
    else:
        # For CIFAR-10, use standard train/test split
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        train_dataset = datasets.CIFAR10(root=args.data_root, train=True, 
                                         download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=args.data_root, train=False,
                                        download=True, transform=transform)
        val_dataset = test_dataset  # Use test as val for CIFAR
        split_info = {'dataset': 'cifar10'}
    
    # Select evaluation set
    if args.eval_set == 'test':
        eval_dataset = test_dataset
        print(f"Evaluating on TEST set ({len(eval_dataset)} images)")
    elif args.eval_set == 'val':
        eval_dataset = val_dataset
        print(f"Evaluating on VALIDATION set ({len(eval_dataset)} images)")
    else:
        eval_dataset = train_dataset
        print(f"Evaluating on TRAIN set ({len(eval_dataset)} images)")
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Load models
    print("\nLoading models...")
    models = {}
    
    if args.l2_checkpoint:
        models['L2 Only'] = load_model(args.l2_checkpoint, 'l2_only', device)
    
    if args.baseline_checkpoint:
        models['Pathak et al.'] = load_model(args.baseline_checkpoint, 'baseline', device)
    
    if args.localizer_checkpoint:
        models['LocPaint'] = load_model(args.localizer_checkpoint, 'localizer', device)
    
    if args.staged_checkpoint:
        models['Staged LocPaint'] = load_model(args.staged_checkpoint, 'staged', device)
    
    if len(models) == 0:
        print("\nError: No models loaded!")
        return
    
    # Evaluate each model
    print(f"\nEvaluating models (mask_seed={args.mask_seed} for reproducibility)...")
    print(f"Mask type: {args.mask_type}, Mask size: {args.mask_size}x{args.mask_size}")
    if args.max_batches:
        print(f"Max batches: {args.max_batches}")
    
    results = {}
    
    for name, model in models.items():
        if model is None:
            continue
        
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print('='*50)
        
        # Reset seed before each model for same masks
        metrics = evaluate_model(
            model, eval_loader, device,
            image_size=args.image_size,
            mask_size=args.mask_size,
            mask_type=args.mask_type,
            mask_seed=args.mask_seed,
            max_batches=args.max_batches
        )
        
        results[name] = metrics
        
        print(f"\n  MSE:   {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
        print(f"  PSNR:  {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}")
        print(f"  SSIM:  {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
        print(f"  LPIPS: {metrics['lpips_mean']:.4f} ± {metrics['lpips_std']:.4f}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    # Header
    print(f"\n{'Model':<25} {'MSE':>12} {'PSNR':>12} {'SSIM':>12} {'LPIPS':>12}")
    print("-"*80)
    
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['mse_mean']:>12.6f} {metrics['psnr_mean']:>12.2f} "
              f"{metrics['ssim_mean']:>12.4f} {metrics['lpips_mean']:>12.4f}")
    
    print("-"*80)
    
    # Find best for each metric
    print("\nBest models:")
    
    metric_goals = {
        'mse_mean': ('MSE', 'min'),
        'psnr_mean': ('PSNR', 'max'),
        'ssim_mean': ('SSIM', 'max'),
        'lpips_mean': ('LPIPS', 'min')
    }
    
    for metric_key, (metric_name, goal) in metric_goals.items():
        if goal == 'min':
            best_model = min(results.items(), key=lambda x: x[1][metric_key])
        else:
            best_model = max(results.items(), key=lambda x: x[1][metric_key])
        
        print(f"  {metric_name}: {best_model[0]} ({best_model[1][metric_key]:.4f})")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_data = {
        'config': {
            'dataset': args.dataset,
            'eval_set': args.eval_set,
            'split_seed': args.split_seed,
            'mask_seed': args.mask_seed,
            'mask_type': args.mask_type,
            'mask_size': args.mask_size,
            'image_size': args.image_size,
            'max_batches': args.max_batches,
            'num_images_evaluated': len(eval_dataset) if args.max_batches is None 
                                    else min(args.max_batches * args.batch_size, len(eval_dataset))
        },
        'checkpoints': {
            'l2_only': args.l2_checkpoint,
            'baseline': args.baseline_checkpoint,
            'localizer': args.localizer_checkpoint,
            'staged': args.staged_checkpoint
        },
        'results': results
    }
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Create LaTeX table with bold best and underlined second best
    latex_path = os.path.join(args.output_dir, 'evaluation_table.tex')
    
    # Find best and second best for each metric
    metric_keys = ['mse_mean', 'psnr_mean', 'ssim_mean', 'lpips_mean']
    metric_formats = {
        'mse_mean': '.4f',
        'psnr_mean': '.2f', 
        'ssim_mean': '.4f',
        'lpips_mean': '.4f'
    }
    metric_goals = {
        'mse_mean': 'min',    # Lower is better
        'psnr_mean': 'max',   # Higher is better
        'ssim_mean': 'max',   # Higher is better
        'lpips_mean': 'min'   # Lower is better
    }
    
    # Get rankings for each metric
    rankings = {}
    for metric_key in metric_keys:
        values = [(name, metrics[metric_key]) for name, metrics in results.items()]
        
        if metric_goals[metric_key] == 'min':
            sorted_values = sorted(values, key=lambda x: x[1])  # Ascending
        else:
            sorted_values = sorted(values, key=lambda x: x[1], reverse=True)  # Descending
        
        rankings[metric_key] = {
            'best': sorted_values[0][0] if len(sorted_values) > 0 else None,
            'second': sorted_values[1][0] if len(sorted_values) > 1 else None
        }
    
    def format_value(name, metric_key, value):
        """Format value with bold for best, underline for second best."""
        fmt = metric_formats[metric_key]
        formatted = f"{value:{fmt}}"
        
        if rankings[metric_key]['best'] == name:
            return f"\\textbf{{{formatted}}}"
        elif rankings[metric_key]['second'] == name:
            return f"\\underline{{{formatted}}}"
        else:
            return formatted
    
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Inpainting Model Comparison. \\textbf{Bold} = best, \\underline{underline} = second best.}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Model & MSE $\\downarrow$ & PSNR $\\uparrow$ & SSIM $\\uparrow$ & LPIPS $\\downarrow$ \\\\\n")
        f.write("\\hline\n")
        
        for name, metrics in results.items():
            # Escape underscores for LaTeX
            latex_name = name.replace('_', '\\_')
            
            mse_str = format_value(name, 'mse_mean', metrics['mse_mean'])
            psnr_str = format_value(name, 'psnr_mean', metrics['psnr_mean'])
            ssim_str = format_value(name, 'ssim_mean', metrics['ssim_mean'])
            lpips_str = format_value(name, 'lpips_mean', metrics['lpips_mean'])
            
            f.write(f"{latex_name} & {mse_str} & {psnr_str} & {ssim_str} & {lpips_str} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to {latex_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Inpainting Models')
    
    # Model checkpoints
    parser.add_argument('--l2_checkpoint', type=str, default=None,
                       help='Path to L2-only model checkpoint')
    parser.add_argument('--baseline_checkpoint', type=str, default=None,
                       help='Path to baseline (Pathak) model checkpoint')
    parser.add_argument('--localizer_checkpoint', type=str, default=None,
                       help='Path to localizer model checkpoint')
    parser.add_argument('--staged_checkpoint', type=str, default=None,
                       help='Path to staged model checkpoint')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='celeba',
                       choices=['celeba', 'cifar10'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--eval_set', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Which split to evaluate on')
    
    # Split settings (must match training!)
    parser.add_argument('--split_seed', type=int, default=42,
                       help='Random seed for data split (use same as training)')
    
    # Mask settings
    parser.add_argument('--mask_seed', type=int, default=456,
                       help='Seed for mask generation (ensures same masks for all models)')
    parser.add_argument('--mask_type', type=str, default='random_square',
                       choices=['center', 'random_square'])
    parser.add_argument('--mask_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=128)
    
    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_batches', type=int, default=None,
                       help='Max batches to evaluate (None = full dataset)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./evaluation_results')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    evaluate_all_models(args)


if __name__ == "__main__":
    main()
