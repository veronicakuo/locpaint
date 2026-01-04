"""
Visualization utilities for context encoder training and evaluation.
Includes support for fixed validation images and comprehensive training curves.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import torchvision.utils as vutils
from PIL import Image
import os


def denormalize(tensor):
    """Convert from [-1, 1] to [0, 1]"""
    return (tensor + 1) / 2


def visualize_results(masked_images, reconstructed, completed, original, masks, save_path):
    """
    Visualize inpainting results in a grid.
    
    Args:
        masked_images: Masked input images
        reconstructed: Reconstructed patches
        completed: Completed images
        original: Original images
        masks: Binary masks
        save_path: Path to save visualization
    """
    batch_size = min(8, original.size(0))
    
    # Denormalize all images
    original = denormalize(original[:batch_size])
    masked_images = denormalize(masked_images[:batch_size])
    reconstructed = denormalize(reconstructed[:batch_size])
    completed = denormalize(completed[:batch_size])
    masks = masks[:batch_size]
    
    # Create grid
    fig, axes = plt.subplots(batch_size, 4, figsize=(12, 3*batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Original
        img = original[i].cpu().permute(1, 2, 0).detach().numpy()
        axes[i, 0].imshow(np.clip(img, 0, 1))
        if i == 0:
            axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Masked
        img = masked_images[i].cpu().permute(1, 2, 0).detach().numpy()
        axes[i, 1].imshow(np.clip(img, 0, 1))
        if i == 0:
            axes[i, 1].set_title('Masked')
        axes[i, 1].axis('off')
        
        # Reconstructed
        img = reconstructed[i].cpu().permute(1, 2, 0).detach().numpy()
        axes[i, 2].imshow(np.clip(img, 0, 1))
        if i == 0:
            axes[i, 2].set_title('Reconstructed')
        axes[i, 2].axis('off')
        
        # Completed
        img = completed[i].cpu().permute(1, 2, 0).detach().numpy()
        axes[i, 3].imshow(np.clip(img, 0, 1))
        if i == 0:
            axes[i, 3].set_title('Completed')
        axes[i, 3].axis('off')
    
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def visualize_fixed_validation(masked_images, reconstructed, completed, original, 
                                masks, epoch, save_path, metrics=None):
    """
    Visualize fixed validation images with metrics annotation.
    
    Args:
        masked_images: Masked input images
        reconstructed: Reconstructed patches
        completed: Completed images
        original: Original images
        masks: Binary masks
        epoch: Current epoch number
        save_path: Path to save visualization
        metrics: Optional dictionary of metrics to display
    """
    batch_size = original.size(0)
    
    # Denormalize all images
    original = denormalize(original)
    masked_images = denormalize(masked_images)
    reconstructed = denormalize(reconstructed)
    completed = denormalize(completed)
    
    # Create grid
    fig, axes = plt.subplots(batch_size, 4, figsize=(10, 2.5*batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Original
        img = original[i].cpu().permute(1, 2, 0).detach().numpy()
        axes[i, 0].imshow(np.clip(img, 0, 1))
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=10)
        axes[i, 0].axis('off')
        
        # Masked
        img = masked_images[i].cpu().permute(1, 2, 0).detach().numpy()
        axes[i, 1].imshow(np.clip(img, 0, 1))
        if i == 0:
            axes[i, 1].set_title('Masked', fontsize=10)
        axes[i, 1].axis('off')
        
        # Reconstructed
        img = reconstructed[i].cpu().permute(1, 2, 0).detach().numpy()
        axes[i, 2].imshow(np.clip(img, 0, 1))
        if i == 0:
            axes[i, 2].set_title('Reconstructed', fontsize=10)
        axes[i, 2].axis('off')
        
        # Completed
        img = completed[i].cpu().permute(1, 2, 0).detach().numpy()
        axes[i, 3].imshow(np.clip(img, 0, 1))
        if i == 0:
            axes[i, 3].set_title('Completed', fontsize=10)
        axes[i, 3].axis('off')
    
    # Add title with epoch and metrics
    title = f'Epoch {epoch}'
    if metrics:
        metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        title += f'  —  {metrics_str}'
    
    # Use tight_layout with rect to leave space for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.suptitle(title, fontsize=11, y=0.995)
    
    # Reduce spacing between subplots
    plt.subplots_adjust(wspace=0.02, hspace=0.05)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_training_curves(history, save_path, include_val=True):
    """
    Plot training curves (losses and discriminator accuracy).
    Validation metrics are saved separately via plot_validation_curves().
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save plot
        include_val: Whether to also generate validation plot
    """
    # ==================
    # TRAINING METRICS PLOT
    # ==================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generator loss
    if 'loss_g' in history and len(history['loss_g']) > 0:
        axes[0, 0].plot(history['loss_g'], label='Generator Loss', color='darkblue', linewidth=1.5)
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Discriminator loss
    if 'loss_d' in history and len(history['loss_d']) > 0:
        axes[0, 1].plot(history['loss_d'], label='Discriminator Loss', color='mediumvioletred', linewidth=1.5)
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Reconstruction loss
    if 'loss_rec' in history and len(history['loss_rec']) > 0:
        axes[1, 0].plot(history['loss_rec'], label='Reconstruction Loss', color='darkorange', linewidth=1.5)
        axes[1, 0].set_title('Reconstruction Loss (MSE)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Discriminator accuracy
    if 'd_real_acc' in history and 'd_fake_acc' in history:
        if len(history['d_real_acc']) > 0:
            axes[1, 1].plot(history['d_real_acc'], label='Real Accuracy', color='darkgreen', linewidth=1.5)
            axes[1, 1].plot(history['d_fake_acc'], label='Fake Accuracy', color='darkred', linewidth=1.5)
            axes[1, 1].set_title('Discriminator Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_ylim([0, 1])
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==================
    # VALIDATION METRICS PLOT (separate file)
    # ==================
    if include_val:
        val_save_path = save_path.replace('.png', '_validation.png')
        plot_validation_curves(history, val_save_path)


def plot_validation_curves(history, save_path):
    """
    Plot validation metrics separately (MSE, PSNR, SSIM, LPIPS).
    
    Args:
        history: Dictionary with validation metrics
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Validation MSE
    ax_mse = axes[0, 0]
    if 'val_mse' in history and len(history['val_mse']) > 0:
        ax_mse.plot(history['val_mse'], label='Val MSE', color='darkorange', linewidth=1.5)
    if 'fixed_val_mse' in history and len(history['fixed_val_mse']) > 0:
        ax_mse.plot(history['fixed_val_mse'], label='Fixed Val MSE', color='coral', 
                   linewidth=1.5, linestyle='--')
    ax_mse.set_title('Validation MSE (↓ lower is better)')
    ax_mse.set_xlabel('Epoch')
    ax_mse.set_ylabel('MSE')
    ax_mse.legend()
    ax_mse.grid(True, alpha=0.3)
    
    # Validation PSNR
    ax_psnr = axes[0, 1]
    if 'val_psnr' in history and len(history['val_psnr']) > 0:
        ax_psnr.plot(history['val_psnr'], label='Val PSNR', color='darkslateblue', linewidth=1.5)
    if 'fixed_val_psnr' in history and len(history['fixed_val_psnr']) > 0:
        ax_psnr.plot(history['fixed_val_psnr'], label='Fixed Val PSNR', color='slateblue', 
                    linewidth=1.5, linestyle='--')
    ax_psnr.set_title('Validation PSNR (↑ higher is better)')
    ax_psnr.set_xlabel('Epoch')
    ax_psnr.set_ylabel('PSNR (dB)')
    ax_psnr.legend()
    ax_psnr.grid(True, alpha=0.3)
    
    # Validation SSIM
    ax_ssim = axes[1, 0]
    if 'val_ssim' in history and len(history['val_ssim']) > 0:
        ax_ssim.plot(history['val_ssim'], label='Val SSIM', color='rebeccapurple', linewidth=1.5)
    if 'fixed_val_ssim' in history and len(history['fixed_val_ssim']) > 0:
        ax_ssim.plot(history['fixed_val_ssim'], label='Fixed Val SSIM', color='mediumpurple', 
                    linewidth=1.5, linestyle='--')
    ax_ssim.set_title('Validation SSIM (↑ higher is better)')
    ax_ssim.set_xlabel('Epoch')
    ax_ssim.set_ylabel('SSIM')
    ax_ssim.legend()
    ax_ssim.grid(True, alpha=0.3)
    
    # Validation LPIPS
    ax_lpips = axes[1, 1]
    if 'val_lpips' in history and len(history['val_lpips']) > 0:
        ax_lpips.plot(history['val_lpips'], label='Val LPIPS', color='darkred', linewidth=1.5)
    if 'fixed_val_lpips' in history and len(history['fixed_val_lpips']) > 0:
        ax_lpips.plot(history['fixed_val_lpips'], label='Fixed Val LPIPS', color='indianred', 
                     linewidth=1.5, linestyle='--')
    ax_lpips.set_title('Validation LPIPS (↓ lower is better)')
    ax_lpips.set_xlabel('Epoch')
    ax_lpips.set_ylabel('LPIPS')
    ax_lpips.legend()
    ax_lpips.grid(True, alpha=0.3)
    
    plt.suptitle('Validation Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(histories, labels, save_path):
    """
    Plot comparison of metrics between multiple models.
    
    Args:
        histories: List of history dictionaries
        labels: List of model names
        save_path: Path to save plot
    """
    metrics_to_plot = ['val_mse', 'val_psnr', 'val_ssim', 'val_lpips']
    titles = ['MSE (↓)', 'PSNR (↑)', 'SSIM (↑)', 'LPIPS (↓)']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx]
        for i, (history, label) in enumerate(zip(histories, labels)):
            if metric in history:
                ax.plot(history[metric], label=label, color=colors[i % len(colors)])
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_grid(images_dict, save_path, titles=None):
    """
    Create a comparison grid for multiple models.
    
    Args:
        images_dict: Dictionary with model names as keys and image tensors as values
        save_path: Path to save visualization
        titles: Optional titles for each column
    """
    model_names = list(images_dict.keys())
    num_models = len(model_names)
    batch_size = images_dict[model_names[0]].size(0)
    
    fig, axes = plt.subplots(batch_size, num_models, figsize=(4*num_models, 4*batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    if num_models == 1:
        axes = axes.reshape(-1, 1)
    
    for col, model_name in enumerate(model_names):
        images = denormalize(images_dict[model_name])
        
        for row in range(batch_size):
            img = images[row].cpu().permute(1, 2, 0).numpy()
            axes[row, col].imshow(np.clip(img, 0, 1))
            
            if row == 0:
                title = titles[col] if titles else model_name
                axes[row, col].set_title(title)
            
            axes[row, col].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_bbox_predictions(images, bboxes_pred, bboxes_true, confidences, save_path):
    """
    Visualize bounding box predictions for localizer.
    
    Args:
        images: Input images
        bboxes_pred: Predicted bounding boxes
        bboxes_true: Ground truth bounding boxes
        confidences: Confidence scores
        save_path: Path to save visualization
    """
    batch_size = min(8, images.size(0))
    images = denormalize(images[:batch_size])
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(min(batch_size, 8)):
        img = images[i].cpu().permute(1, 2, 0).detach().numpy()
        axes[i].imshow(np.clip(img, 0, 1))
        
        h, w = img.shape[:2]
        
        # Draw true bbox in green
        if bboxes_true is not None and i < len(bboxes_true):
            true_box = bboxes_true[i].cpu().detach().numpy()
            if true_box.ndim > 1:
                true_box = true_box[0]
            x1, y1, x2, y2 = true_box * w
            rect_true = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='#35ba23', facecolor='none'
            )
            axes[i].add_patch(rect_true)
        
        # Draw predicted bbox in red
        if bboxes_pred is not None and i < len(bboxes_pred):
            pred_box = bboxes_pred[i].cpu().detach().numpy()
            if pred_box.ndim > 1:
                pred_box = pred_box[0]
            x1, y1, x2, y2 = pred_box * w
            rect_pred = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='#bf1029', facecolor='none'
            )
            axes[i].add_patch(rect_pred)
        
        # Add confidence score
        if confidences is not None and i < len(confidences):
            conf = confidences[i].item() if hasattr(confidences[i], 'item') else confidences[i]
            axes[i].set_title(f'Conf: {conf:.2f}')
        
        axes[i].axis('off')
    
    # Turn off any unused subplots
    for i in range(batch_size, 8):
        axes[i].axis('off')

    plt.subplots_adjust(wspace=0.02)
    plt.suptitle('Bounding Box Predictions', fontsize=12)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def create_epoch_summary(epoch, train_metrics, val_metrics, save_path):
    """
    Create a summary visualization for an epoch.
    
    Args:
        epoch: Epoch number
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
        save_path: Path to save summary
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Create text summary
    text = f"Epoch {epoch} Summary\n"
    text += "=" * 40 + "\n\n"
    
    text += "Training Metrics:\n"
    for key, val in train_metrics.items():
        text += f"  {key}: {val:.4f}\n"
    
    text += "\nValidation Metrics:\n"
    for key, val in val_metrics.items():
        text += f"  {key}: {val:.4f}\n"
    
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
