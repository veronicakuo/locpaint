"""
Baseline Context Encoder Implementation
Faithful to Pathak et al. (2016) - uses square masks only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datetime import datetime
import os

class BaselineEncoder(nn.Module):
    """Original encoder architecture from Pathak et al."""
    def __init__(self, input_channels=3, latent_dim=4000):
        super(BaselineEncoder, self).__init__()
        
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

class BaselineDecoder(nn.Module):
    """Original decoder architecture from Pathak et al."""
    def __init__(self, latent_dim=4000):
        super(BaselineDecoder, self).__init__()
        
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

class BaselineDiscriminator(nn.Module):
    """Standard binary discriminator from original paper"""
    def __init__(self, input_channels=3):
        super(BaselineDiscriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
        
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        return x.view(x.size(0), -1)

class BaselineContextEncoder:
    """
    Baseline Context Encoder - faithful to original paper
    Uses ONLY square masks (center or random position)
    """
    
    def __init__(self, config):
        """
        Initialize from config dictionary
        Config should contain: image_size, mask_size, mask_type, latent_dim,
                              lr_g, lr_d, beta1, beta2, lambda_rec, lambda_adv, device
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))
        
        # Model parameters
        self.image_size = config.get('image_size', 128)
        self.mask_size = config.get('mask_size', 64)
        self.latent_dim = config.get('latent_dim', 4000)
        
        # IMPORTANT: Mask type for baseline
        # 'center' - always center (like original paper)
        # 'random_square' - random position but always square
        self.mask_type = config.get('mask_type', 'center')  
        
        # Training parameters
        self.lr_g = config.get('lr_g', 0.0002)
        self.lr_d = config.get('lr_d', 0.0002)
        self.beta1 = config.get('beta1', 0.5)
        self.beta2 = config.get('beta2', 0.999)
        
        # Loss weights (following original paper)
        self.lambda_rec = config.get('lambda_rec', 0.999)
        self.lambda_adv = config.get('lambda_adv', 0.001)
        
        # Initialize networks
        self.encoder = BaselineEncoder(latent_dim=self.latent_dim).to(self.device)
        self.decoder = BaselineDecoder(latent_dim=self.latent_dim).to(self.device)
        self.discriminator = BaselineDiscriminator().to(self.device)
        
        # Optimizers
        self.optimizer_g = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr_g,
            betas=(self.beta1, self.beta2)
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_d,
            betas=(self.beta1, self.beta2)
        )
        
        # Loss functions
        self.criterion_rec = nn.MSELoss()
        self.criterion_adv = nn.BCELoss()
        
        # Training stats
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
    
    def generate_square_mask(self, batch_size):
        """
        Generate SQUARE masks only - this is what the baseline should use
        Returns both masks and bounding boxes for fair comparison with localizer
        """
        masks = torch.zeros(batch_size, 1, self.image_size, self.image_size)
        bboxes = []
        
        for i in range(batch_size):
            if self.mask_type == 'center':
                # Center mask (original paper default)
                start = (self.image_size - self.mask_size) // 2
                end = start + self.mask_size
                
                masks[i, :, start:end, start:end] = 1
                
                bbox = torch.tensor([
                    start / self.image_size,
                    start / self.image_size,
                    end / self.image_size,
                    end / self.image_size
                ])
                
            elif self.mask_type == 'random_square':
                # Random position but ALWAYS SQUARE
                max_pos = self.image_size - self.mask_size
                x = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
                y = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
                
                masks[i, :, y:y+self.mask_size, x:x+self.mask_size] = 1
                
                bbox = torch.tensor([
                    x / self.image_size,
                    y / self.image_size,
                    (x + self.mask_size) / self.image_size,
                    (y + self.mask_size) / self.image_size
                ])
            else:
                raise ValueError(f"Unknown mask type: {self.mask_type}. Use 'center' or 'random_square'")
            
            bboxes.append(bbox)
        
        return masks.to(self.device), torch.stack(bboxes).to(self.device)
    
    def extract_patches(self, images, masks):
        """Extract the masked regions as 64x64 patches for discriminator"""
        batch_size = images.size(0)
        patches = []
        
        for i in range(batch_size):
            # Find bounding box of mask
            mask = masks[i, 0]
            nonzero = torch.nonzero(mask)
            
            if len(nonzero) > 0:
                y_min = nonzero[:, 0].min().item()
                y_max = nonzero[:, 0].max().item() + 1
                x_min = nonzero[:, 1].min().item()
                x_max = nonzero[:, 1].max().item() + 1
                
                # Extract patch
                patch = images[i:i+1, :, y_min:y_max, x_min:x_max]
                
                # Resize to 64x64 for discriminator
                patch = F.interpolate(patch, size=(64, 64), mode='bilinear', align_corners=False)
                patches.append(patch)
            else:
                # Fallback - should never happen with proper masks
                patches.append(torch.zeros(1, 3, 64, 64).to(self.device))
        
        return torch.cat(patches, dim=0)
    
    def train_step(self, images):
        """Training step with SQUARE masks only"""
        batch_size = images.size(0)
        
        # Labels for adversarial loss
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # Generate SQUARE masks (not irregular!)
        masks, bboxes = self.generate_square_mask(batch_size)
        
        # Create masked images
        masked_images = images * (1 - masks)
        
        # Extract patches for discriminator
        real_patches = self.extract_patches(images, masks)
        
        # ==================
        # Train Generator
        # ==================
        self.optimizer_g.zero_grad()
        
        # Encode and decode
        encoded = self.encoder(masked_images)
        reconstructed = self.decoder(encoded)
        
        # Resize if necessary
        if reconstructed.size(2) != self.image_size:
            reconstructed = F.interpolate(
                reconstructed, 
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Extract fake patches
        fake_patches = self.extract_patches(reconstructed, masks)
        
        # Reconstruction loss (on full image)
        real_content = images * masks
        fake_content = reconstructed * masks
        loss_rec = self.criterion_rec(fake_content, real_content)
        
        # Adversarial loss
        pred_fake = self.discriminator(fake_patches)
        loss_adv = self.criterion_adv(pred_fake, real_labels)
        
        # Combined generator loss
        loss_g = self.lambda_rec * loss_rec + self.lambda_adv * loss_adv
        
        loss_g.backward()
        self.optimizer_g.step()
        
        # =====================
        # Train Discriminator
        # =====================
        self.optimizer_d.zero_grad()
        
        # Real patches
        pred_real = self.discriminator(real_patches)
        loss_d_real = self.criterion_adv(pred_real, real_labels)
        
        # Fake patches (detached)
        pred_fake_d = self.discriminator(fake_patches.detach())
        loss_d_fake = self.criterion_adv(pred_fake_d, fake_labels)
        
        # Combined discriminator loss
        loss_d = (loss_d_real + loss_d_fake) / 2
        
        loss_d.backward()
        self.optimizer_d.step()
        
        # Complete the images
        completed_images = masked_images + reconstructed * masks
        
        self.global_step += 1
        
        return {
            'loss_g': loss_g.item(),
            'loss_d': loss_d.item(),
            'loss_rec': loss_rec.item(),
            'loss_adv': loss_adv.item(),
            'loss_d_real': loss_d_real.item(),
            'loss_d_fake': loss_d_fake.item(),
            'd_real_acc': (pred_real > 0.5).float().mean().item(),
            'd_fake_acc': (pred_fake_d < 0.5).float().mean().item(),
            'completed_images': completed_images,
            'masked_images': masked_images,
            'reconstructed': reconstructed,
            'masks': masks,
            'bboxes': bboxes,  # Return bboxes for fair comparison
            'mask_type': self.mask_type
        }
    
    def save_checkpoint(self, save_path, metrics=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'encoder_state': self.encoder.state_dict(),
            'decoder_state': self.decoder.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'optimizer_g_state': self.optimizer_g.state_dict(),
            'optimizer_d_state': self.optimizer_d.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss,
            'metrics': metrics if metrics else {},
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder_state'])
        self.decoder.load_state_dict(checkpoint['decoder_state'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint.get('metrics', {})
    
    def infer(self, masked_images):
        """Inference mode"""
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            encoded = self.encoder(masked_images)
            reconstructed = self.decoder(encoded)
            
            if reconstructed.size(2) != masked_images.size(2):
                reconstructed = F.interpolate(
                    reconstructed,
                    size=masked_images.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
        
        return reconstructed