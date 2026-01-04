"""
Training script for Staged Context Encoder with Validation Metrics.
Stage 1: Binary discriminator (Pathak et al.)
Stage 2: Localizer discriminator (after transition_epoch)

Parallel structure to baseline and localizer training scripts.
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from staged_context_encoder import StagedContextEncoder
from utils.visualization import (
    visualize_results, 
    plot_training_curves, 
    visualize_bbox_predictions,
    visualize_fixed_validation
)
from utils.metrics import (
    calculate_metrics, 
    calculate_validation_metrics,
    calculate_fixed_val_metrics
)
from utils.data_split import (
    get_celeba_splits, 
    get_fixed_val_indices, 
    get_fixed_val_batch,
    save_split_info
)


def get_args():
    parser = argparse.ArgumentParser(description='Train Staged Context Encoder')
    
    # Model parameters
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--mask_size', type=int, default=64)
    parser.add_argument('--mask_type', type=str, default='random_square',
                       choices=['center', 'random_square'],
                       help='Type of mask: center (original paper) or random_square')
    parser.add_argument('--latent_dim', type=int, default=4000)
    
    # STAGED TRAINING PARAMETERS (key difference from baseline/localizer)
    parser.add_argument('--transition_epoch', type=int, default=20,
                       help='Epoch to switch from binary to localizer discriminator')
    parser.add_argument('--transition_warmup', type=int, default=5,
                       help='Number of epochs to gradually increase localizer weight')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                       help='Warmup epochs when loading from baseline (gradual localizer introduction)')
    
    # Localizer-specific parameters (used in Stage 2)
    parser.add_argument('--num_boxes', type=int, default=1)
    parser.add_argument('--d_capacity_factor', type=float, default=1.0,
                       help='Discriminator capacity factor (reduce to weaken)')
    parser.add_argument('--d_dropout_rate', type=float, default=0.0,
                       help='Dropout rate in discriminator')
    parser.add_argument('--d_num_layers', type=int, default=5,
                       help='Number of layers in localizer discriminator')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr_g', type=float, default=0.0002)
    parser.add_argument('--lr_d', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    
    # Loss weights for Stage 1 (baseline)
    parser.add_argument('--lambda_rec', type=float, default=0.999,
                       help='Reconstruction loss weight')
    parser.add_argument('--lambda_adv', type=float, default=0.001,
                       help='Adversarial loss weight (Stage 1)')
    
    # Loss weights for Stage 2 (localizer)
    parser.add_argument('--lambda_conf', type=float, default=0.0005,
                       help='Confidence loss weight (Stage 2)')
    parser.add_argument('--lambda_loc', type=float, default=0.0001,
                       help='Localization loss weight (Stage 2)')
    parser.add_argument('--lambda_iou', type=float, default=0.0005,
                       help='IoU loss weight (Stage 2)')
    
    # Training dynamics for Stage 2
    parser.add_argument('--d_update_ratio', type=int, default=1,
                       help='Update discriminator every N iterations (Stage 2)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Label smoothing factor')
    
    # Dataset and splits (parallel to baseline/localizer)
    parser.add_argument('--dataset', type=str, default='celeba',
                       choices=['cifar10', 'celeba', 'stl10'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--split_seed', type=int, default=42,
                       help='Random seed for train/val/test split (use same across models)')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--num_fixed_val', type=int, default=10,
                       help='Number of fixed validation images for visualization')
    
    # Validation settings
    parser.add_argument('--val_freq', type=int, default=1,
                       help='Validation frequency (every N epochs)')
    parser.add_argument('--val_max_batches', type=int, default=50,
                       help='Max batches for validation (None = full val set)')
    parser.add_argument('--fixed_mask_seed', type=int, default=123,
                       help='Seed for fixed validation masks')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./staged/checkpoints')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--viz_freq', type=int, default=100)
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from staged checkpoint')
    parser.add_argument('--load_baseline', type=str, default=None,
                       help='Initialize from baseline checkpoint (encoder/decoder only)')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def get_dataloaders(args):
    """Create dataloaders with consistent splits (parallel to baseline/localizer)."""
    if args.dataset == 'celeba':
        # Use the split utility for CelebA
        train_dataset, val_dataset, test_dataset, split_info = get_celeba_splits(
            data_root=args.data_root,
            image_size=args.image_size,
            split_seed=args.split_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        
        # Get fixed validation indices
        fixed_val_indices = get_fixed_val_indices(
            val_dataset, 
            num_samples=args.num_fixed_val,
            seed=args.split_seed
        )
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
        
        # Get fixed validation batch
        fixed_val_images = get_fixed_val_batch(
            val_dataset, 
            fixed_val_indices, 
            device=args.device
        )
        
        return train_loader, val_loader, fixed_val_images, split_info
    
    else:
        # For other datasets, use standard approach
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        if args.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(root=args.data_root, train=True, 
                                            download=True, transform=transform)
            val_dataset = datasets.CIFAR10(root=args.data_root, train=False,
                                          download=True, transform=transform)
        elif args.dataset == 'stl10':
            train_dataset = datasets.STL10(root=args.data_root, split='train',
                                          download=True, transform=transform)
            val_dataset = datasets.STL10(root=args.data_root, split='test',
                                        download=True, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers)
        
        # Get fixed validation images
        fixed_val_indices = get_fixed_val_indices(val_dataset, args.num_fixed_val, args.split_seed)
        fixed_val_images = get_fixed_val_batch(val_dataset, fixed_val_indices, args.device)
        
        split_info = {'dataset': args.dataset, 'fixed_val_indices': fixed_val_indices}
        
        return train_loader, val_loader, fixed_val_images, split_info


def train_epoch(model, dataloader, epoch, args):
    """Train for one epoch."""
    epoch_metrics = {
        'loss_g': 0, 'loss_d': 0, 'loss_d_localizer': 0, 'loss_rec': 0,
        'loss_adv': 0, 'loss_conf_g': 0, 'loss_loc_g': 0, 'loss_iou_g': 0,
        'iou': 0, 
        'd_real_acc': 0, 'd_fake_acc': 0,
        'loc_real_acc': 0, 'loc_fake_acc': 0,
        'd_updates': 0, 'd_loc_updates': 0
    }
    
    # Check for stage transition
    stage = model.check_transition(epoch)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Stage {stage}]")
    
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(model.device)
        
        # Training step
        metrics = model.train_step(images)
        
        # Accumulate metrics
        for key in epoch_metrics:
            if key in metrics:
                epoch_metrics[key] += metrics[key]
        
        # Track D updates
        if metrics.get('d_updated', True):
            epoch_metrics['d_updates'] += 1
        if metrics.get('d_loc_updated', False):
            epoch_metrics['d_loc_updates'] += 1
        
        # Update progress bar
        status = {
            'G': f"{metrics['loss_g']:.4f}",
            'D': f"{metrics['loss_d']:.4f}",
            'Rec': f"{metrics['loss_rec']:.4f}"
        }
        
        if stage == 2:
            status['IoU'] = f"{metrics['iou']:.3f}"
            if metrics.get('d_loc_updated', False):
                status['D_loc'] = '✓'
        
        pbar.set_postfix(status)
        
        # Visualize samples
        if batch_idx % args.viz_freq == 0:
            save_path = os.path.join(
                args.checkpoint_dir,
                'samples',
                f'epoch_{epoch:03d}_batch_{batch_idx:05d}.png'
            )
            
            visualize_results(
                metrics['masked_images'][:8],
                metrics['reconstructed'][:8],
                metrics['completed_images'][:8],
                images[:8],
                metrics['masks'][:8],
                save_path
            )
            
            # Visualize bounding boxes in Stage 2
            if stage == 2 and metrics['bbox_pred'] is not None:
                bbox_save_path = os.path.join(
                    args.checkpoint_dir,
                    'samples',
                    f'bbox_epoch_{epoch:03d}_batch_{batch_idx:05d}.png'
                )
                visualize_bbox_predictions(
                    metrics['completed_images'][:8],
                    metrics['bbox_pred'][:8],
                    metrics['bbox_true'][:8],
                    metrics['confidence'][:8],
                    bbox_save_path
                )
    
    # Average metrics properly
    num_batches = len(dataloader)
    num_d_updates = epoch_metrics['d_updates']
    num_d_loc_updates = epoch_metrics['d_loc_updates']
    
    for key in epoch_metrics:
        if key in ['d_updates', 'd_loc_updates']:
            continue
        elif key in ['d_real_acc', 'd_fake_acc', 'loss_d']:
            # Binary D metrics: average over D updates
            if num_d_updates > 0:
                epoch_metrics[key] /= num_d_updates
        elif key in ['loc_real_acc', 'loc_fake_acc', 'loss_d_localizer']:
            # Localizer D metrics: average over localizer D updates
            if num_d_loc_updates > 0:
                epoch_metrics[key] /= num_d_loc_updates
        else:
            # G metrics: average over all batches
            epoch_metrics[key] /= num_batches
    
    epoch_metrics['stage'] = stage
    
    return epoch_metrics


def validate(model, val_loader, fixed_val_images, epoch, args):
    """Run validation and return metrics."""
    print(f"\n  Running validation...")
    
    # Calculate metrics on validation set
    val_metrics = calculate_validation_metrics(
        model, 
        val_loader, 
        max_batches=args.val_max_batches,
        device=model.device
    )
    
    print(f"  Val metrics - MSE: {val_metrics['mse']:.4f}, "
          f"PSNR: {val_metrics['psnr']:.2f}, "
          f"SSIM: {val_metrics['ssim']:.4f}, "
          f"LPIPS: {val_metrics['lpips']:.4f}")
    
    # Calculate metrics on fixed validation images with fixed masks
    # Returns: metrics, completed, masked, reconstructed, masks, bboxes
    fixed_metrics, completed, masked, reconstructed, masks, _ = calculate_fixed_val_metrics(
        model,
        fixed_val_images,
        mask_seed=args.fixed_mask_seed,
        device=model.device
    )
    
    # Save fixed validation visualization
    fixed_val_path = os.path.join(
        args.checkpoint_dir,
        'fixed_val',
        f'fixed_val_epoch_{epoch:03d}.png'
    )
    visualize_fixed_validation(
        masked, reconstructed, completed, fixed_val_images,
        masks, epoch, fixed_val_path, 
        metrics={'PSNR': fixed_metrics['psnr'], 
                'SSIM': fixed_metrics['ssim'],
                'LPIPS': fixed_metrics['lpips']}
    )
    
    return val_metrics, fixed_metrics


def main():
    args = get_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint_dir, 'fixed_val'), exist_ok=True)
    
    # Create config dictionary from args
    config = {
        # Model parameters
        'image_size': args.image_size,
        'mask_size': args.mask_size,
        'mask_type': args.mask_type,
        'latent_dim': args.latent_dim,
        
        # Staged training
        'transition_epoch': args.transition_epoch,
        'transition_warmup': args.transition_warmup,
        'warmup_epochs': args.warmup_epochs,
        
        # Localizer parameters
        'num_boxes': args.num_boxes,
        'd_capacity_factor': args.d_capacity_factor,
        'd_dropout_rate': args.d_dropout_rate,
        'd_num_layers': args.d_num_layers,
        
        # Training parameters
        'lr_g': args.lr_g,
        'lr_d': args.lr_d,
        'beta1': args.beta1,
        'beta2': args.beta2,
        
        # Loss weights
        'lambda_rec': args.lambda_rec,
        'lambda_adv': args.lambda_adv,
        'lambda_conf': args.lambda_conf,
        'lambda_loc': args.lambda_loc,
        'lambda_iou': args.lambda_iou,
        
        # Training dynamics
        'd_update_ratio': args.d_update_ratio,
        'label_smoothing': args.label_smoothing,
        
        # Other
        'device': args.device,
        'batch_size': args.batch_size,
        'dataset': args.dataset,
        'seed': args.seed,
        'split_seed': args.split_seed
    }
    
    # Initialize model
    model = StagedContextEncoder(config)
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        metrics = model.load_checkpoint(args.resume)
        start_epoch = model.current_epoch + 1
    elif args.load_baseline:
        # Load baseline model and configure warmup for Stage 2
        model.load_baseline_checkpoint(
            args.load_baseline,
            start_epoch=args.transition_epoch,
            warmup_epochs=args.warmup_epochs
        )
        start_epoch = model.current_epoch
        print(f"Starting from baseline checkpoint at epoch {start_epoch} (Stage 2)")
    
    # Get data loaders with consistent splits
    train_loader, val_loader, fixed_val_images, split_info = get_dataloaders(args)
    
    # Save split info for reproducibility
    split_info_path = os.path.join(args.checkpoint_dir, 'split_info.json')
    save_split_info(split_info, split_info_path)
    
    # Training history with validation metrics (parallel to baseline/localizer)
    history = {
        # Training metrics
        'loss_g': [], 'loss_d': [], 'loss_d_localizer': [], 'loss_rec': [],
        'loss_adv': [], 'loss_conf_g': [], 'loss_loc_g': [], 'loss_iou_g': [],
        'iou': [], 
        'd_real_acc': [], 'd_fake_acc': [],
        'loc_real_acc': [], 'loc_fake_acc': [],
        'stage': [],
        # Validation metrics
        'val_mse': [], 'val_psnr': [], 'val_ssim': [], 'val_lpips': [], 'val_l1': [],
        # Fixed validation metrics
        'fixed_val_mse': [], 'fixed_val_psnr': [], 'fixed_val_ssim': [], 'fixed_val_lpips': []
    }
    
    # Restore history if resuming
    history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
    if args.resume and os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        print(f"Restored training history ({len(history.get('loss_g', []))} epochs)")
    
    # Print configuration
    print(f"\n{'='*60}")
    print(f"STAGED CONTEXT ENCODER TRAINING")
    print(f"{'='*60}")
    print(f"Starting from epoch {start_epoch}")
    print(f"Dataset: {args.dataset}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Device: {model.device}")
    print(f"Mask: {args.mask_type} ({args.mask_size}x{args.mask_size})")
    print(f"Split seed: {args.split_seed} (same seed = same split across models)")
    print(f"Fixed validation images: {args.num_fixed_val}")
    print(f"\n--- Staged Training Schedule ---")
    print(f"Stage 1 (Binary D): Epochs 0-{args.transition_epoch-1}")
    print(f"Transition: Epochs {args.transition_epoch}-{args.transition_epoch + args.transition_warmup - 1}")
    print(f"Stage 2 (Localizer D): Epochs {args.transition_epoch}+")
    print(f"\n--- Stage 1 Parameters (Binary Discriminator) ---")
    print(f"  Lambda Rec: {args.lambda_rec}")
    print(f"  Lambda Adv: {args.lambda_adv}")
    print(f"\n--- Stage 2 Parameters (Localizer Discriminator) ---")
    print(f"  Capacity factor: {args.d_capacity_factor}")
    print(f"  Dropout rate: {args.d_dropout_rate}")
    print(f"  Num layers: {args.d_num_layers}")
    print(f"  Lambda Conf: {args.lambda_conf}")
    print(f"  Lambda Loc: {args.lambda_loc} (NEGATIVE in G loss)")
    print(f"  Lambda IoU: {args.lambda_iou}")
    print(f"  D Update Ratio: 1/{args.d_update_ratio}")
    if args.warmup_epochs > 0:
        print(f"\n--- Warmup Configuration ---")
        print(f"  Warmup epochs: {args.warmup_epochs}")
        print(f"  (Localizer losses gradually increase from 0 to full weight)")
    print(f"{'='*60}\n")
    
    # Best metrics tracking (restore from history if resuming)
    best_val_psnr = max(history.get('val_psnr', [0.0])) if history.get('val_psnr') else 0.0
    best_val_ssim = max(history.get('val_ssim', [0.0])) if history.get('val_ssim') else 0.0
    best_val_lpips = min(history.get('val_lpips', [float('inf')])) if history.get('val_lpips') else float('inf')
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        model.current_epoch = epoch
        
        # Train epoch
        epoch_metrics = train_epoch(model, train_loader, epoch, args)
        
        # Update training history
        for key in ['loss_g', 'loss_d', 'loss_d_localizer', 'loss_rec', 'loss_adv',
                    'loss_conf_g', 'loss_loc_g', 'loss_iou_g', 'iou',
                    'd_real_acc', 'd_fake_acc', 'loc_real_acc', 'loc_fake_acc', 'stage']:
            if key in epoch_metrics:
                history[key].append(epoch_metrics[key])
        
        # Validation
        if (epoch + 1) % args.val_freq == 0:
            val_metrics, fixed_metrics = validate(
                model, val_loader, fixed_val_images, epoch, args
            )
            
            # Update validation history
            history['val_mse'].append(val_metrics['mse'])
            history['val_psnr'].append(val_metrics['psnr'])
            history['val_ssim'].append(val_metrics['ssim'])
            history['val_lpips'].append(val_metrics['lpips'])
            history['val_l1'].append(val_metrics['l1'])
            
            # Update fixed validation history
            history['fixed_val_mse'].append(fixed_metrics['mse'])
            history['fixed_val_psnr'].append(fixed_metrics['psnr'])
            history['fixed_val_ssim'].append(fixed_metrics['ssim'])
            history['fixed_val_lpips'].append(fixed_metrics['lpips'])
        
        # Print epoch summary
        stage = epoch_metrics['stage']
        print(f"\nEpoch {epoch+1} Summary [Stage {stage}]:")
        print(f"  Train - G: {epoch_metrics['loss_g']:.4f}, "
              f"D: {epoch_metrics['loss_d']:.4f}, "
              f"Rec: {epoch_metrics['loss_rec']:.4f}")
        print(f"  Train - D Accuracy: {(epoch_metrics['d_real_acc'] + epoch_metrics['d_fake_acc'])/2:.2f}")
        
        if stage == 2:
            print(f"  Train - Localizer D Loss: {epoch_metrics.get('loss_d_localizer', 0):.4f}")
            print(f"  Train - Localizer Accuracy: {(epoch_metrics.get('loc_real_acc', 0) + epoch_metrics.get('loc_fake_acc', 0))/2:.2f}")
            print(f"  Train - IoU: {epoch_metrics['iou']:.3f}")
            print(f"  Train - D Updates: {epoch_metrics['d_updates']}/{len(train_loader)}, "
                  f"D_loc Updates: {epoch_metrics['d_loc_updates']}/{len(train_loader)}")
        
        # Show warmup status
        if model.is_in_warmup():
            warmup_prog = model.get_warmup_progress()
            print(f"  [WARMUP - Progress: {warmup_prog:.1%}, Localizer losses scaled to {warmup_prog:.1%}]")
        
        # Indicate if transitioning
        if epoch == args.transition_epoch:
            print(f"  [TRANSITION STARTED - Localizer now active]")
        elif epoch > args.transition_epoch and epoch < args.transition_epoch + args.transition_warmup:
            blend = model.get_blend_factor()
            print(f"  [TRANSITION - Blend factor: {blend:.2f}]")
        
        if (epoch + 1) % args.val_freq == 0:
            print(f"  Val   - MSE: {val_metrics['mse']:.4f}, "
                  f"PSNR: {val_metrics['psnr']:.2f}, "
                  f"SSIM: {val_metrics['ssim']:.4f}, "
                  f"LPIPS: {val_metrics['lpips']:.4f}")
            
            # Track best metrics
            if val_metrics['psnr'] > best_val_psnr:
                best_val_psnr = val_metrics['psnr']
                best_psnr_path = os.path.join(args.checkpoint_dir, 'best_psnr_model.pth')
                model.save_checkpoint(best_psnr_path, {**epoch_metrics, **val_metrics})
                print(f"  New best PSNR: {best_val_psnr:.2f}")
            
            if val_metrics['ssim'] > best_val_ssim:
                best_val_ssim = val_metrics['ssim']
                best_ssim_path = os.path.join(args.checkpoint_dir, 'best_ssim_model.pth')
                model.save_checkpoint(best_ssim_path, {**epoch_metrics, **val_metrics})
                print(f"  New best SSIM: {best_val_ssim:.4f}")
            
            if val_metrics['lpips'] < best_val_lpips:
                best_val_lpips = val_metrics['lpips']
                best_lpips_path = os.path.join(args.checkpoint_dir, 'best_lpips_model.pth')
                model.save_checkpoint(best_lpips_path, {**epoch_metrics, **val_metrics})
                print(f"  New best LPIPS: {best_val_lpips:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'checkpoint_epoch_{epoch+1:03d}.pth'
            )
            all_metrics = {**epoch_metrics}
            if (epoch + 1) % args.val_freq == 0:
                all_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
            model.save_checkpoint(checkpoint_path, all_metrics)
            
            # Save training curves
            plot_path = os.path.join(args.checkpoint_dir, 'training_curves.png')
            plot_training_curves(history, plot_path, include_val=True)
            
            # Save training history
            history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
        
        # Save best model based on reconstruction loss
        if epoch_metrics['loss_rec'] < model.best_loss:
            model.best_loss = epoch_metrics['loss_rec']
            best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            model.save_checkpoint(best_path, epoch_metrics)
            print(f"  New best train loss: {model.best_loss:.4f}")
        
        # Track best IoU in Stage 2
        if stage == 2 and epoch_metrics['iou'] > model.best_iou:
            model.best_iou = epoch_metrics['iou']
            best_iou_path = os.path.join(args.checkpoint_dir, 'best_iou_model.pth')
            model.save_checkpoint(best_iou_path, epoch_metrics)
            print(f"  New best IoU: {model.best_iou:.3f}")
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    model.save_checkpoint(final_path, epoch_metrics)
    
    # Save final training history
    history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final training curves
    plot_path = os.path.join(args.checkpoint_dir, 'training_curves.png')
    plot_training_curves(history, plot_path, include_val=True)
    
    # Save config for reference
    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Final train reconstruction loss: {model.best_loss:.4f}")
    print(f"Best IoU achieved: {model.best_iou:.3f}")
    print(f"Best validation PSNR: {best_val_psnr:.2f}")
    print(f"Best validation SSIM: {best_val_ssim:.4f}")
    print(f"Best validation LPIPS: {best_val_lpips:.4f}")


if __name__ == "__main__":
    main()
