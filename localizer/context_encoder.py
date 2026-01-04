"""
Context Encoder with Localizer Discriminator Implementationl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datetime import datetime
import os


class LocalizerEncoder(nn.Module):
    """Encoder network - identical to baseline for fair comparison"""
    def __init__(self, input_channels=3, latent_dim=4000):
        super(LocalizerEncoder, self).__init__()
        
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


class LocalizerDecoder(nn.Module):
    """Decoder network - identical to baseline for fair comparison"""
    def __init__(self, latent_dim=4000):
        super(LocalizerDecoder, self).__init__()
        
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


class LocalizerDiscriminator(nn.Module):
    """
    Localizer Discriminator with adjustable capacity.
    Outputs both confidence (is there a fake region?) and bbox (where is it?).
    """
    def __init__(self, input_channels=3, num_boxes=1, image_size=128, 
                 capacity_factor=1.0, dropout_rate=0.0, num_layers=5):
        super(LocalizerDiscriminator, self).__init__()
        
        self.num_boxes = num_boxes
        self.image_size = image_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        
        # Channel capacities (adjustable via capacity_factor)
        base_channels = int(64 * capacity_factor)
        
        layers = []
        in_channels = input_channels
        out_channels = base_channels
        
        # Build convolutional layers
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=4, stride=2, padding=1))
            
            if i > 0:  # Skip batch norm on first layer
                layers.append(nn.BatchNorm2d(out_channels))
            
            layers.append(nn.LeakyReLU(0.2))
            
            if dropout_rate > 0 and i > 1:  # Add dropout to later layers
                layers.append(nn.Dropout2d(dropout_rate))
            
            in_channels = out_channels
            out_channels = min(out_channels * 2, int(512 * capacity_factor))
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate feature size
        feat_size = image_size // (2 ** num_layers)
        self.feat_dim = in_channels * feat_size * feat_size
        
        # Detection head (confidence: is there a fake region?)
        self.detection_head = nn.Sequential(
            nn.Linear(self.feat_dim, int(256 * capacity_factor)),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(int(256 * capacity_factor), num_boxes)
        )
        
        # Localization head (bbox: where is the fake region?)
        self.localization_head = nn.Sequential(
            nn.Linear(self.feat_dim, int(256 * capacity_factor)),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(int(256 * capacity_factor), num_boxes * 4)
        )
    
    def forward(self, x):
        # Extract features
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        
        # Predict confidence and bbox
        confidence = torch.sigmoid(self.detection_head(features))
        bbox_raw = self.localization_head(features)
        bbox = torch.sigmoid(bbox_raw.view(-1, self.num_boxes, 4))
        
        return bbox, confidence


class ContextEncoderWithLocalizer:
    """
    Context Encoder with Localizer Discriminator.
    
    Key difference from baseline: discriminator predicts WHERE the inpainted
    region is (bounding box) rather than just binary real/fake classification.
    """
    
    def __init__(self, config):
        """
        Initialize from config dictionary.
        
        Config parameters:
            Model: image_size, mask_size, mask_type, latent_dim, num_boxes
            Discriminator: d_capacity_factor, d_dropout_rate, d_num_layers
            Training: lr_g, lr_d, beta1, beta2
            Generator loss weights: lambda_rec, lambda_conf, lambda_loc, lambda_iou
            Dynamics: d_update_ratio, label_smoothing, warmup_epochs, gradient_clip
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))
        
        # Model parameters
        self.image_size = config.get('image_size', 128)
        self.mask_size = config.get('mask_size', 64)
        self.mask_type = config.get('mask_type', 'random_square')
        self.latent_dim = config.get('latent_dim', 4000)
        self.num_boxes = config.get('num_boxes', 1)
        
        # Discriminator parameters
        self.d_capacity_factor = config.get('d_capacity_factor', 1.0)
        self.d_dropout_rate = config.get('d_dropout_rate', 0.0)
        self.d_num_layers = config.get('d_num_layers', 5)
        
        # Training parameters
        self.lr_g = config.get('lr_g', 0.0002)
        self.lr_d = config.get('lr_d', 0.0002)
        self.beta1 = config.get('beta1', 0.5)
        self.beta2 = config.get('beta2', 0.999)
        
        # Generator loss weights
        # L_G = lambda_rec * L_rec + lambda_conf * L_conf + lambda_loc * L_loc + lambda_iou * L_iou
        self.lambda_rec = config.get('lambda_rec', 0.999)
        self.lambda_conf = config.get('lambda_conf', 0.0005)
        self.lambda_loc = config.get('lambda_loc', 0.00025)
        self.lambda_iou = config.get('lambda_iou', 0.00025)
        
        # Training dynamics
        self.d_update_ratio = config.get('d_update_ratio', 1)
        self.label_smoothing = config.get('label_smoothing', 0.1)
        self.warmup_epochs = config.get('warmup_epochs', 0)
        
        # Initialize networks
        self.encoder = LocalizerEncoder(latent_dim=self.latent_dim).to(self.device)
        self.decoder = LocalizerDecoder(latent_dim=self.latent_dim).to(self.device)
        self.discriminator = LocalizerDiscriminator(
            num_boxes=self.num_boxes,
            image_size=self.image_size,
            capacity_factor=self.d_capacity_factor,
            dropout_rate=self.d_dropout_rate,
            num_layers=self.d_num_layers
        ).to(self.device)
        
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
        self.criterion_conf = nn.BCELoss()
        self.criterion_loc = nn.SmoothL1Loss()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.best_iou = 0.0
        self.d_update_counter = 0
    
    def generate_square_mask(self, batch_size):
        """Generate square masks with random or center positions."""
        masks = torch.zeros(batch_size, 1, self.image_size, self.image_size)
        bboxes = []
        
        for i in range(batch_size):
            if self.mask_type == 'center':
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
                raise ValueError(f"Unknown mask type: {self.mask_type}")
            
            bboxes.append(bbox)
        
        return masks.to(self.device), torch.stack(bboxes).to(self.device)
    
    def compute_iou(self, boxes1, boxes2):
        """Compute IoU between predicted and ground truth boxes."""
        if boxes1.dim() == 2:
            boxes1 = boxes1.unsqueeze(1)
        if boxes2.dim() == 2:
            boxes2 = boxes2.unsqueeze(1)
        
        # Intersection
        x1_max = torch.max(boxes1[..., 0], boxes2[..., 0])
        y1_max = torch.max(boxes1[..., 1], boxes2[..., 1])
        x2_min = torch.min(boxes1[..., 2], boxes2[..., 2])
        y2_min = torch.min(boxes1[..., 3], boxes2[..., 3])
        
        intersection = torch.clamp(x2_min - x1_max, min=0) * \
                      torch.clamp(y2_min - y1_max, min=0)
        
        # Union
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-7)
    
    def get_smooth_labels(self, batch_size, target_value):
        """
        Get labels with optional smoothing.
        target_value: 0.0 for "real" (no fake region), 1.0 for "fake" (has fake region)
        """
        if target_value > 0.5:
            # Target is ~1 (fake): use 1.0 - smoothing
            labels = torch.ones(batch_size, 1) * (1.0 - self.label_smoothing)
        else:
            # Target is ~0 (real): use 0.0 + smoothing
            labels = torch.ones(batch_size, 1) * self.label_smoothing
        
        return labels.to(self.device)
    
    def train_step(self, images):
        """
        Single training step.
        
        Returns dict with losses, metrics, and generated images.
        """
        batch_size = images.size(0)
        
        # Generate masks and target bboxes
        masks, target_bboxes = self.generate_square_mask(batch_size)
        if self.num_boxes == 1:
            target_bboxes = target_bboxes.unsqueeze(1)
        
        # Create masked images
        masked_images = images * (1 - masks)
        
        # =====================
        # GENERATOR UPDATE
        # =====================
        self.optimizer_g.zero_grad()
        
        # Forward pass through generator
        encoded = self.encoder(masked_images)
        reconstructed = self.decoder(encoded)
        
        if reconstructed.size(2) != self.image_size:
            reconstructed = F.interpolate(
                reconstructed,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Create completed images
        completed_images = masked_images + reconstructed * masks
        
        # Get discriminator predictions on completed images
        bbox_pred, conf_pred = self.discriminator(completed_images)
        
        # --- Generator Losses ---
        
        # L_rec: Reconstruction loss
        loss_rec = self.criterion_rec(reconstructed * masks, images * masks)
        
        # L_conf: G wants D to output LOW confidence (think it's real)
        target_conf_g = self.get_smooth_labels(batch_size, target_value=0.0)
        loss_conf_g = self.criterion_conf(conf_pred, target_conf_g)
        
        # L_loc and L_iou: G is penalized when D can localize accurately
        # 
        # L_loc: Negative SmoothL1 - G wants to MAXIMIZE the bbox error
        #   Low SmoothL1 = D accurate = bad for G
        #   So we use NEGATIVE SmoothL1: G minimizing -SmoothL1 = G maximizing SmoothL1
        loss_loc_g = -self.criterion_loc(bbox_pred, target_bboxes)
        
        # L_iou: Positive IoU - G is penalized when D has high overlap
        #   High IoU = D accurate = bad for G
        #   G minimizing IoU = G wants D to have low overlap with true bbox
        iou_pred = self.compute_iou(bbox_pred, target_bboxes)
        loss_iou_g = iou_pred.mean()
        
        # Handle warmup (only reconstruction loss)
        if self.current_epoch < self.warmup_epochs:
            lambda_conf = 0.0
            lambda_loc = 0.0
            lambda_iou = 0.0
        else:
            lambda_conf = self.lambda_conf
            lambda_loc = self.lambda_loc
            lambda_iou = self.lambda_iou
        
        # Total generator loss
        # L_G = lambda_rec * L_rec + lambda_conf * L_conf + lambda_loc * L_loc + lambda_iou * L_iou
        loss_g = (self.lambda_rec * loss_rec + 
                  lambda_conf * loss_conf_g + 
                  lambda_loc * loss_loc_g +
                  lambda_iou * loss_iou_g)
        
        loss_g.backward()
        
        if self.config.get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                self.config['gradient_clip']
            )
        
        self.optimizer_g.step()
        
        # =====================
        # DISCRIMINATOR UPDATE
        # =====================
        self.d_update_counter += 1
        update_d = (self.d_update_counter % self.d_update_ratio == 0)
        update_d = update_d and (self.current_epoch >= self.warmup_epochs)
        
        if update_d:
            self.optimizer_d.zero_grad()
            
            # --- Real Images ---
            # D should output LOW confidence for real images (no fake region)
            bbox_real, conf_real = self.discriminator(images)
            target_conf_real = self.get_smooth_labels(batch_size, target_value=0.0)
            loss_d_real_conf = self.criterion_conf(conf_real, target_conf_real)
            
            # NO bbox loss for real images - there's no ground truth bbox
            
            # --- Fake (Completed) Images ---
            # D should output HIGH confidence for fake images
            completed_detached = completed_images.detach()
            bbox_fake, conf_fake = self.discriminator(reconstructed.detach()) # TODO: REVERT
            target_conf_fake = self.get_smooth_labels(batch_size, target_value=1.0)
            loss_d_fake_conf = self.criterion_conf(conf_fake, target_conf_fake)
            
            # D should correctly localize the inpainted region
            loss_d_fake_loc = self.criterion_loc(bbox_fake, target_bboxes)
            
            # IoU loss: D wants high IoU, so loss = 1 - IoU
            iou_fake = self.compute_iou(bbox_fake, target_bboxes)
            loss_d_fake_iou = 1.0 - iou_fake.mean()
            
            # --- Total Discriminator Loss ---
            # L_D = L_real_conf + L_fake_conf + 1.0 * L_fake_loc + 0.5 * L_fake_iou
            loss_d = (loss_d_real_conf + 
                      loss_d_fake_conf + 
                      1.0 * loss_d_fake_loc + 
                      0.5 * loss_d_fake_iou)
            
            loss_d.backward()
            
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(),
                    self.config['gradient_clip']
                )
            
            self.optimizer_d.step()
            
            # Metrics
            d_real_acc = (conf_real < 0.5).float().mean().item()
            d_fake_acc = (conf_fake > 0.5).float().mean().item()
            iou_metric = iou_fake.mean().item()
        else:
            # No D update this step
            loss_d = torch.tensor(0.0)
            loss_d_real_conf = torch.tensor(0.0)
            loss_d_fake_conf = torch.tensor(0.0)
            loss_d_fake_loc = torch.tensor(0.0)
            loss_d_fake_iou = torch.tensor(0.0)
            d_real_acc = 0.0
            d_fake_acc = 0.0
            iou_metric = iou_pred.mean().item()
            bbox_fake = bbox_pred
            conf_fake = conf_pred
        
        self.global_step += 1
        
        return {
            # Generator losses
            'loss_g': loss_g.item(),
            'loss_rec': loss_rec.item(),
            'loss_conf_g': loss_conf_g.item(),
            'loss_loc_g': loss_loc_g.item(),
            'loss_iou_g': loss_iou_g.item(),
            
            # Discriminator losses
            'loss_d': loss_d.item() if update_d else 0.0,
            'loss_d_real_conf': loss_d_real_conf.item() if update_d else 0.0,
            'loss_d_fake_conf': loss_d_fake_conf.item() if update_d else 0.0,
            'loss_d_fake_loc': loss_d_fake_loc.item() if update_d else 0.0,
            'loss_d_fake_iou': loss_d_fake_iou.item() if update_d else 0.0,
            
            # Metrics
            'iou': iou_metric,
            'd_real_acc': d_real_acc,
            'd_fake_acc': d_fake_acc,
            'd_updated': update_d,
            
            # Images for visualization
            'completed_images': reconstructed, # TODO: REVERT
            'masked_images': masked_images,
            'reconstructed': completed_images, # TODO: REVERT
            'masks': masks,
            'bbox_pred': bbox_fake,
            'bbox_true': target_bboxes,
            'confidence': conf_fake,
            'mask_type': self.mask_type
        }
    
    def save_checkpoint(self, save_path, metrics=None):
        """Save model checkpoint."""
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
            'best_iou': self.best_iou,
            'metrics': metrics if metrics else {},
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder_state'])
        self.decoder.load_state_dict(checkpoint['decoder_state'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.best_iou = checkpoint.get('best_iou', 0.0)
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint.get('metrics', {})
    
    def infer(self, masked_images):
        """Run inference (no gradients)."""
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
