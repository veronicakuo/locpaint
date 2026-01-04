"""
Unified evaluation script for comparing Baseline vs Localizer models
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
from scipy import stats
import argparse
import os
import json
from tqdm import tqdm

from baseline_context_encoder import BaselineContextEncoder
from your_localizer_model import ContextEncoderWithLocalizer  # Your localizer implementation
from evaluation_metrics import ComprehensiveEvaluator

def load_models(baseline_checkpoint, localizer_checkpoint, device='cuda'):
    """Load both models from checkpoints"""
    
    # Load baseline
    baseline_ckpt = torch.load(baseline_checkpoint, map_location=device)
    baseline_config = baseline_ckpt['config']
    baseline_model = BaselineContextEncoder(baseline_config)
    baseline_model.load_checkpoint(baseline_checkpoint)
    
    # Load localizer
    localizer_ckpt = torch.load(localizer_checkpoint, map_location=device)
    localizer_config = localizer_ckpt['config']
    localizer_model = ContextEncoderWithLocalizer(
        device=device,
        **localizer_config
    )
    localizer_model.load_checkpoint(localizer_checkpoint)
    
    return baseline_model, localizer_model

def evaluate_single_model(model, test_loader, evaluator, model_name, device='cuda'):
    """Evaluate a single model on test set"""
    
    model.encoder.eval()
    model.decoder.eval()
    
    all_metrics = []
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            images = images.to(device)
            
            # Test with different mask types
            for mask_type in ['center', 'random', 'irregular']:
                # Generate masks
                if hasattr(model, 'generate_mask'):
                    masks = model.generate_mask(images.size(0), mask_type)
                else:
                    masks = generate_masks(images.size(0), mask_type, device)
                
                # Create masked images
                masked_images = images * (1 - masks)
                
                # Inpaint
                if hasattr(model, 'infer'):
                    reconstructed = model.infer(masked_images)
                else:
                    encoded = model.encoder(masked_images)
                    reconstructed = model.decoder(encoded)
                
                # Resize if needed
                if reconstructed.size(2) != images.size(2):
                    reconstructed = F.interpolate(
                        reconstructed,
                        size=images.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Complete images
                completed = masked_images + reconstructed * masks
                
                # Calculate metrics
                metrics = evaluator.evaluate_batch(images, completed, masks, model_name)
                metrics['mask_type'] = mask_type
                all_metrics.append(metrics)
    
    return all_metrics

def compare_models(baseline_metrics, localizer_metrics):
    """Statistical comparison between models"""
    
    comparison_results = []
    
    # Group by mask type
    mask_types = set([m['mask_type'] for m in baseline_metrics])
    
    for mask_type in mask_types:
        base_type = [m for m in baseline_metrics if m['mask_type'] == mask_type]
        loc_type = [m for m in localizer_metrics if m['mask_type'] == mask_type]
        
        for metric_name in ['l1', 'l2', 'psnr', 'ssim']:
            base_vals = [m[metric_name] for m in base_type]
            loc_vals = [m[metric_name] for m in loc_type]
            
            # Paired t-test
            t_stat, p_value = stats.ttest_ind(base_vals, loc_vals)
            
            # Calculate means and improvement
            base_mean = np.mean(base_vals)
            loc_mean = np.mean(loc_vals)
            
            if metric_name in ['l1', 'l2']:  # Lower is better
                improvement = (base_mean - loc_mean) / base_mean * 100
                better = 'Localizer' if loc_mean < base_mean else 'Baseline'
            else:  # Higher is better
                improvement = (loc_mean - base_mean) / base_mean * 100
                better = 'Localizer' if loc_mean > base_mean else 'Baseline'
            
            comparison_results.append({
                'Mask Type': mask_type,
                'Metric': metric_name,
                'Baseline Mean': base_mean,
                'Localizer Mean': loc_mean,
                'Better Model': better,
                'Improvement (%)': improvement,
                'p-value': p_value,
                'Significant': p_value < 0.05
            })
    
    return pd.DataFrame(comparison_results)

def create_comparison_figure(baseline_model, localizer_model, test_images, save_path):
    """Create visual comparison figure"""
    
    device = baseline_model.device
    num_samples = min(5, test_images.size(0))
    
    fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3*num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            image = test_images[i:i+1].to(device)
            
            # Generate mask
            mask = baseline_model.generate_mask(1, 'random')
            masked = image * (1 - mask)
            
            # Baseline reconstruction
            recon_base = baseline_model.infer(masked)
            completed_base = masked + recon_base * mask
            
            # Localizer reconstruction
            recon_loc = localizer_model.infer(masked)
            completed_loc = masked + recon_loc * mask
            
            # Denormalize for display
            image_show = (image[0] + 1) / 2
            masked_show = (masked[0] + 1) / 2
            completed_base_show = (completed_base[0] + 1) / 2
            completed_loc_show = (completed_loc[0] + 1) / 2
            
            # Display
            axes[i, 0].imshow(image_show.cpu().permute(1, 2, 0))
            axes[i, 0].set_title("Original")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(masked_show.cpu().permute(1, 2, 0))
            axes[i, 1].set_title("Masked")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(completed_base_show.cpu().permute(1, 2, 0))
            axes[i, 2].set_title("Baseline")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(completed_loc_show.cpu().permute(1, 2, 0))
            axes[i, 3].set_title("Localizer")
            axes[i, 3].axis('off')
            
            # Difference maps
            diff_base = torch.abs(image_show - completed_base_show).mean(0)
            diff_loc = torch.abs(image_show - completed_loc_show).mean(0)
            
            axes[i, 4].imshow(diff_base.cpu().numpy(), cmap='hot')
            axes[i, 4].set_title("Baseline Error")
            axes[i, 4].axis('off')
            
            axes[i, 5].imshow(diff_loc.cpu().numpy(), cmap='hot')
            axes[i, 5].set_title("Localizer Error")
            axes[i, 5].axis('off')
    
    plt.suptitle("Model Comparison: Baseline vs Localizer", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--localizer_checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='./evaluation_results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    baseline_model, localizer_model = load_models(
        args.baseline_checkpoint,
        args.localizer_checkpoint,
        args.device
    )
    
    # Create test data loader
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(device=args.device)
    
    # Evaluate both models
    print("\nEvaluating Baseline Model...")
    baseline_metrics = evaluate_single_model(
        baseline_model, test_loader, evaluator, "Baseline", args.device
    )
    
    print("\nEvaluating Localizer Model...")
    localizer_metrics = evaluate_single_model(
        localizer_model, test_loader, evaluator, "Localizer", args.device
    )
    
    # Statistical comparison
    print("\nPerforming statistical comparison...")
    comparison_df = compare_models(baseline_metrics, localizer_metrics)
    
    # Save results
    comparison_df.to_csv(os.path.join(args.output_dir, 'comparison_results.csv'))
    print("\nComparison Results:")
    print(comparison_df.to_string())
    
    # Create visualization
    print("\nCreating comparison visualizations...")
    test_images = next(iter(test_loader))[0][:10]
    create_comparison_figure(
        baseline_model,
        localizer_model,
        test_images,
        os.path.join(args.output_dir, 'visual_comparison.png')
    )
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()