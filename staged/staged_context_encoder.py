"""
Staged Context Encoder Implementation
Starts with baseline binary discriminator (Pathak et al.), 
then transitions to localizer discriminator after warmup epochs.

This addresses the problem of the localizer discriminator becoming too powerful
too early by first letting the generator learn basic reconstruction with the
simpler binary discriminator.

Loss Functions (Stage 2):
-------------------------
Discriminator (fixed weights):
    L_D = L_real_conf + L_fake_conf + 1.0 * L_fake_loc + 0.5 * L_fake_iou
    
    - L_real_conf: BCE(conf_real, ~0) - real images should have LOW confidence
    - L_fake_conf: BCE(conf_fake, ~1) - fake images should have HIGH confidence
    - L_fake_loc: SmoothL1(bbox_fake, bbox_true) - localize the inpainted region
    - L_fake_iou: 1 - IoU(bbox_fake, bbox_true) - maximize overlap with true bbox
    
    NOTE: No bbox penalty for real images (no ground truth exists)

Generator:
    L_G = lambda_rec * L_rec + lambda_adv * L_binary_adv 
        + lambda_conf * L_conf + lambda_loc * L_loc + lambda_iou * L_iou
    
    - L_rec: MSE reconstruction loss
    - L_binary_adv: BCE loss to fool binary discriminator
    - L_conf: BCE(conf_pred, ~0) - G wants D to output low confidence (think it's real)
    - L_loc: -SmoothL1(bbox_pred, bbox_true) - G wants D to have HIGH bbox error (NEGATIVE)
    - L_iou: IoU(bbox_pred, bbox_true) - G penalized when D has high IoU with true bbox
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datetime import datetime
import os


# =============================================================================
# SHARED ENCODER/DECODER (identical to both baseline and localizer)
# =============================================================================

class Encoder(nn.Module):
    """Encoder network - identical architecture for fair comparison"""
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
    """Decoder network - identical architecture for fair comparison"""
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


# =============================================================================
# BASELINE BINARY DISCRIMINATOR (from Pathak et al.)
# =============================================================================

class BinaryDiscriminator(nn.Module):
    """Standard binary discriminator from original paper"""
    def __init__(self, input_channels=3):
        super(BinaryDiscriminator, self).__init__()
        
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


# =============================================================================
# LOCALIZER DISCRIMINATOR (your novel approach)
# =============================================================================

class LocalizerDiscriminator(nn.Module):
    """
    Localizer Discriminator - predicts bounding boxes instead of binary classification
    With adjustable capacity for training stability
    """
    def __init__(self, input_channels=3, num_boxes=1, image_size=128, 
                 capacity_factor=1.0, dropout_rate=0.0, num_layers=5):
        super(LocalizerDiscriminator, self).__init__()
        
        self.num_boxes = num_boxes
        self.image_size = image_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        
        # Adjustable channel capacities
        base_channels = int(64 * capacity_factor)
        
        layers = []
        in_channels = input_channels
        out_channels = base_channels
        
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=4, stride=2, padding=1))
            
            if i > 0:
                layers.append(nn.BatchNorm2d(out_channels))
            
            layers.append(nn.LeakyReLU(0.2))
            
            if dropout_rate > 0 and i > 1:
                layers.append(nn.Dropout2d(dropout_rate))
            
            in_channels = out_channels
            out_channels = min(out_channels * 2, int(512 * capacity_factor))
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate feature size
        feat_size = image_size // (2 ** num_layers)
        self.feat_dim = in_channels * feat_size * feat_size
        
        # Detection head (confidence)
        self.detection_head = nn.Sequential(
            nn.Linear(self.feat_dim, int(128 * capacity_factor)),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(int(128 * capacity_factor), num_boxes)
        )
        
        # Localization head (bbox coordinates)
        self.localization_head = nn.Sequential(
            nn.Linear(self.feat_dim, int(128 * capacity_factor)),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(int(128 * capacity_factor), num_boxes * 4)
        )
    
    def forward(self, x):
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        
        confidence = torch.sigmoid(self.detection_head(features))
        bbox_raw = self.localization_head(features)
        bbox = torch.sigmoid(bbox_raw.view(-1, self.num_boxes, 4))
        
        return bbox, confidence


# =============================================================================
# STAGED CONTEXT ENCODER - Main Class
# =============================================================================

class StagedContextEncoder:
    """
    Context Encoder with Staged Training:
    - Stage 1 (epochs 0 to transition_epoch-1): Use baseline binary discriminator
    - Stage 2 (epochs transition_epoch onwards): Use localizer discriminator
    
    The encoder/decoder weights are preserved across the transition.
    
    Loss Functions (Stage 2):
    -------------------------
    Discriminator (fixed weights, no lambda scaling):
        L_D = L_real_conf + L_fake_conf + 1.0 * L_fake_loc + 0.5 * L_fake_iou
    
    Generator:
        L_G = lambda_rec * L_rec + lambda_conf * L_conf + lambda_loc * L_loc + lambda_iou * L_iou
        Default: 0.999 * L_rec + 0.0005 * L_conf + 0.00025 * L_loc + 0.00025 * L_iou
    """
    
    def __init__(self, config):
        """
        Initialize from config dictionary.
        
        Key config parameters:
        - transition_epoch: When to switch from binary to localizer discriminator
        - All other parameters similar to baseline/localizer configs
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))
        
        # Model parameters
        self.image_size = config.get('image_size', 128)
        self.mask_size = config.get('mask_size', 64)
        self.mask_type = config.get('mask_type', 'random_square')
        self.latent_dim = config.get('latent_dim', 4000)
        
        # Staged training parameters
        self.transition_epoch = config.get('transition_epoch', 20)
        self.transition_warmup = config.get('transition_warmup', 5)  # Gradual blend
        
        # Warmup after loading baseline (reconstruction-only before introducing localizer)
        self.warmup_epochs = config.get('warmup_epochs', 0)
        self.warmup_start_epoch = config.get('warmup_start_epoch', 0)  # Set when loading baseline
        
        # Localizer-specific parameters
        self.num_boxes = config.get('num_boxes', 1)
        self.d_capacity_factor = config.get('d_capacity_factor', 1.0)
        self.d_dropout_rate = config.get('d_dropout_rate', 0.0)
        self.d_num_layers = config.get('d_num_layers', 5)
        
        # Training parameters
        self.lr_g = config.get('lr_g', 0.0002)
        self.lr_d = config.get('lr_d', 0.0002)
        self.beta1 = config.get('beta1', 0.5)
        self.beta2 = config.get('beta2', 0.999)
        
        # Loss weights for Stage 1 (baseline)
        self.lambda_rec = config.get('lambda_rec', 0.999)
        self.lambda_adv = config.get('lambda_adv', 0.001)
        
        # Loss weights for Stage 2 Generator (localizer)
        # These are the GENERATOR loss weights (small values to not overpower reconstruction)
        self.lambda_conf = config.get('lambda_conf', 0.0005)
        self.lambda_loc = config.get('lambda_loc', 0.00025)   # bbox regression weight for G
        self.lambda_iou = config.get('lambda_iou', 0.00025)   # IoU loss weight for G
        
        # Discriminator loss weights are FIXED (not configurable via lambda)
        # L_D = L_real_conf + L_fake_conf + 1.0 * L_fake_loc + 0.5 * L_fake_iou
        self.d_weight_bbox = 1.0   # Fixed weight for discriminator bbox loss
        self.d_weight_iou = 0.5    # Fixed weight for discriminator IoU loss
        
        # Training dynamics
        self.d_update_ratio = config.get('d_update_ratio', 1)
        self.label_smoothing = config.get('label_smoothing', 0.0)
        
        # Initialize encoder/decoder (shared across stages)
        self.encoder = Encoder(latent_dim=self.latent_dim).to(self.device)
        self.decoder = Decoder(latent_dim=self.latent_dim).to(self.device)
        
        # Initialize both discriminators
        self.binary_discriminator = BinaryDiscriminator().to(self.device)
        self.localizer_discriminator = LocalizerDiscriminator(
            num_boxes=self.num_boxes,
            image_size=self.image_size,
            capacity_factor=self.d_capacity_factor,
            dropout_rate=self.d_dropout_rate,
            num_layers=self.d_num_layers
        ).to(self.device)
        
        # Current stage tracking
        self.current_stage = 1  # Start with binary discriminator
        
        # Optimizers for generator
        self.optimizer_g = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr_g,
            betas=(self.beta1, self.beta2)
        )
        
        # Optimizers for both discriminators
        self.optimizer_d_binary = optim.Adam(
            self.binary_discriminator.parameters(),
            lr=self.lr_d,
            betas=(self.beta1, self.beta2)
        )
        
        self.optimizer_d_localizer = optim.Adam(
            self.localizer_discriminator.parameters(),
            lr=self.lr_d,
            betas=(self.beta1, self.beta2)
        )
        
        # Loss functions
        self.criterion_rec = nn.MSELoss()
        self.criterion_adv = nn.BCELoss()
        self.criterion_conf = nn.BCELoss()
        self.criterion_loc = nn.SmoothL1Loss()
        
        # Training stats
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.best_iou = 0.0
        self.d_update_counter = 0
    
    def generate_square_mask(self, batch_size):
        """Generate square masks (same as both baseline and localizer)"""
        masks = torch.zeros(batch_size, 1, self.image_size, self.image_size)
        bboxes = []
        
        for i in range(batch_size):
            if self.mask_type == 'center':
                start = (self.image_size - self.mask_size) // 2
                end = start + self.mask_size
                masks[i, :, start:end, start:end] = 1
                bbox = torch.tensor([
                    start / self.image_size, start / self.image_size,
                    end / self.image_size, end / self.image_size
                ])
            elif self.mask_type == 'random_square':
                max_pos = self.image_size - self.mask_size
                x = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
                y = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
                masks[i, :, y:y+self.mask_size, x:x+self.mask_size] = 1
                bbox = torch.tensor([
                    x / self.image_size, y / self.image_size,
                    (x + self.mask_size) / self.image_size,
                    (y + self.mask_size) / self.image_size
                ])
            else:
                raise ValueError(f"Unknown mask type: {self.mask_type}")
            
            bboxes.append(bbox)
        
        masks = masks.to(self.device)
        bboxes = torch.stack(bboxes).to(self.device)
        return masks, bboxes
    
    def get_smooth_labels(self, batch_size, is_real):
        """Get labels with optional smoothing"""
        if self.label_smoothing > 0:
            if is_real:
                return torch.ones(batch_size, 1).to(self.device) * (1.0 - self.label_smoothing)
            else:
                return torch.ones(batch_size, 1).to(self.device) * self.label_smoothing
        else:
            if is_real:
                return torch.ones(batch_size, 1).to(self.device)
            else:
                return torch.zeros(batch_size, 1).to(self.device)
    
    def compute_iou(self, pred_bbox, target_bbox):
        """Compute IoU between predicted and target bounding boxes"""
        pred = pred_bbox.squeeze(1)
        target = target_bbox
        
        x1_inter = torch.max(pred[:, 0], target[:, 0])
        y1_inter = torch.max(pred[:, 1], target[:, 1])
        x2_inter = torch.min(pred[:, 2], target[:, 2])
        y2_inter = torch.min(pred[:, 3], target[:, 3])
        
        inter_width = torch.clamp(x2_inter - x1_inter, min=0)
        inter_height = torch.clamp(y2_inter - y1_inter, min=0)
        intersection = inter_width * inter_height
        
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = pred_area + target_area - intersection + 1e-6
        
        iou = intersection / union
        return iou
    
    def get_current_stage(self):
        """Determine current training stage based on epoch"""
        if self.current_epoch < self.transition_epoch:
            return 1
        else:
            return 2
    
    def get_blend_factor(self):
        """Get blend factor for gradual transition between stages"""
        if self.current_epoch < self.transition_epoch:
            return 0.0
        elif self.current_epoch < self.transition_epoch + self.transition_warmup:
            progress = (self.current_epoch - self.transition_epoch) / self.transition_warmup
            return progress
        else:
            return 1.0
    
    def get_warmup_progress(self):
        """
        Get warmup progress (0.0 to 1.0) for localizer losses after loading baseline.
        Returns 1.0 if not in warmup period.
        """
        if self.warmup_epochs <= 0:
            return 1.0
        
        epochs_since_warmup_start = self.current_epoch - self.warmup_start_epoch
        
        if epochs_since_warmup_start < 0:
            return 0.0
        elif epochs_since_warmup_start >= self.warmup_epochs:
            return 1.0
        else:
            return epochs_since_warmup_start / self.warmup_epochs
    
    def is_in_warmup(self):
        """Check if currently in warmup period"""
        return self.get_warmup_progress() < 1.0
    
    def train_step_stage1(self, images, masks, target_bboxes):
        """
        Stage 1: Baseline training with binary discriminator only.
        Same as original Pathak et al. approach.
        """
        batch_size = images.size(0)
        
        # Prepare images
        masked_images = images * (1 - masks)
        
        # ==================
        # Train Generator
        # ==================
        self.encoder.train()
        self.decoder.train()
        self.optimizer_g.zero_grad()
        
        # Forward pass
        encoded = self.encoder(masked_images)
        reconstructed = self.decoder(encoded)
        
        # Resize if needed
        if reconstructed.size(2) != images.size(2):
            reconstructed = F.interpolate(
                reconstructed, 
                size=images.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Complete the image
        completed_images = images * (1 - masks) + reconstructed * masks
        
        # Extract patches for binary discriminator
        fake_patches = self.extract_patches(reconstructed, masks)
        
        # Generator losses
        real_labels = self.get_smooth_labels(batch_size, is_real=True)
        
        # Reconstruction loss
        real_content = images * masks
        fake_content = reconstructed * masks
        loss_rec = self.criterion_rec(fake_content, real_content)
        
        # Adversarial loss
        pred_fake = self.binary_discriminator(fake_patches)
        loss_adv = self.criterion_adv(pred_fake, real_labels)
        
        # Combined loss
        loss_g = self.lambda_rec * loss_rec + self.lambda_adv * loss_adv
        
        loss_g.backward()
        self.optimizer_g.step()
        
        # ==================
        # Train Discriminator
        # ==================
        self.optimizer_d_binary.zero_grad()
        
        # Real samples
        real_patches = self.extract_patches(images, masks)
        pred_real = self.binary_discriminator(real_patches)
        loss_d_real = self.criterion_adv(pred_real, real_labels)
        
        # Fake samples
        fake_labels = self.get_smooth_labels(batch_size, is_real=False)
        pred_fake_d = self.binary_discriminator(fake_patches.detach())
        loss_d_fake = self.criterion_adv(pred_fake_d, fake_labels)
        
        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward()
        self.optimizer_d_binary.step()
        
        return {
            'loss_g': loss_g.item(),
            'loss_d': loss_d.item(),
            'loss_rec': loss_rec.item(),
            'loss_adv': loss_adv.item(),
            'loss_conf_g': 0.0,
            'loss_loc_g': 0.0,
            'loss_iou_g': 0.0,
            'd_real_acc': (pred_real > 0.5).float().mean().item(),
            'd_fake_acc': (pred_fake_d < 0.5).float().mean().item(),
            'completed_images': completed_images,
            'masked_images': masked_images,
            'reconstructed': reconstructed,
            'masks': masks,
            'bbox_pred': None,
            'bbox_true': target_bboxes,
            'confidence': None,
            'iou': 0.0,
            'stage': 1,
            'd_updated': True
        }
    
    def extract_patches(self, images, masks):
        """Extract the masked region as patches for binary discriminator"""
        batch_size = images.size(0)
        patches = []
        
        for i in range(batch_size):
            mask = masks[i, 0]
            nonzero = torch.nonzero(mask)
            if len(nonzero) > 0:
                y_min, x_min = nonzero.min(dim=0)[0]
                y_max, x_max = nonzero.max(dim=0)[0]
                patch = images[i:i+1, :, y_min:y_max+1, x_min:x_max+1]
                patch = F.interpolate(patch, size=(64, 64), mode='bilinear', align_corners=False)
            else:
                patch = F.interpolate(images[i:i+1], size=(64, 64), mode='bilinear', align_corners=False)
            patches.append(patch)
        
        return torch.cat(patches, dim=0)
    
    def train_step_stage2(self, images, masks, target_bboxes):
        """
        Stage 2: Training with BOTH binary and localizer discriminators.
        Binary D provides texture quality signal, Localizer D provides spatial feedback.
        
        Loss Functions:
        ---------------
        Discriminator (fixed weights):
            L_D = L_real_conf + L_fake_conf + 1.0 * L_fake_loc + 0.5 * L_fake_iou
            
        Generator:
            L_G = lambda_rec * L_rec + lambda_adv * L_binary_adv 
                + lambda_conf * L_conf + lambda_loc * L_loc + lambda_iou * L_iou
        """
        batch_size = images.size(0)
        
        # Prepare images
        masked_images = images * (1 - masks)
        
        # Get warmup progress for localizer losses
        warmup_progress = self.get_warmup_progress()
        
        # Scale localizer losses by warmup progress
        current_lambda_conf = self.lambda_conf * warmup_progress
        current_lambda_loc = self.lambda_loc * warmup_progress
        current_lambda_iou = self.lambda_iou * warmup_progress
        
        # ==================
        # Train Generator
        # ==================
        self.encoder.train()
        self.decoder.train()
        self.optimizer_g.zero_grad()
        
        # Forward pass
        encoded = self.encoder(masked_images)
        reconstructed = self.decoder(encoded)
        
        if reconstructed.size(2) != images.size(2):
            reconstructed = F.interpolate(
                reconstructed,
                size=images.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        completed_images = images * (1 - masks) + reconstructed * masks
        
        # Extract patches for binary discriminator
        fake_patches = self.extract_patches(reconstructed, masks)
        real_labels = self.get_smooth_labels(batch_size, is_real=True)
        fake_labels = self.get_smooth_labels(batch_size, is_real=False)
        
        # Convert target bboxes for localizer (expects different format)
        target_bboxes_loc = target_bboxes.unsqueeze(1) if target_bboxes.dim() == 2 else target_bboxes
        target_bboxes_loc = target_bboxes_loc.squeeze(1)
        
        # --- Reconstruction Loss (always full weight) ---
        real_content = images * masks
        fake_content = reconstructed * masks
        loss_rec = self.criterion_rec(fake_content, real_content)
        
        # --- Binary Discriminator Loss (texture quality - always on) ---
        pred_fake_binary = self.binary_discriminator(fake_patches)
        loss_adv = self.criterion_adv(pred_fake_binary, real_labels)
        
        # --- Localizer Discriminator Loss (spatial feedback - gradual) ---
        bbox_pred, conf_pred = self.localizer_discriminator(completed_images)
        
        # Confidence loss (fool localizer - want LOW confidence)
        target_conf_g = self.get_smooth_labels(batch_size, is_real=False)
        loss_conf_g = self.criterion_conf(conf_pred, target_conf_g)
        
        # L_loc: NEGATIVE SmoothL1 - G wants to MAXIMIZE the bbox error
        # Low SmoothL1 = D accurate = bad for G
        # So we use NEGATIVE SmoothL1: G minimizing -SmoothL1 = G maximizing SmoothL1
        loss_loc_g = -self.criterion_loc(bbox_pred.squeeze(1), target_bboxes_loc)
        
        # L_iou: Positive IoU - G is penalized when D has high overlap
        # High IoU = D accurate = bad for G
        # G minimizing IoU = G wants D to have low overlap with true bbox
        iou = self.compute_iou(bbox_pred, target_bboxes_loc)
        loss_iou_g = iou.mean()
        
        # Combined generator loss:
        # L_G = lambda_rec * L_rec + lambda_adv * L_binary_adv 
        #     + lambda_conf * L_conf + lambda_loc * L_loc + lambda_iou * L_iou
        loss_g = (self.lambda_rec * loss_rec + 
                 self.lambda_adv * loss_adv +
                 current_lambda_conf * loss_conf_g + 
                 current_lambda_loc * loss_loc_g +
                 current_lambda_iou * loss_iou_g)
        
        loss_g.backward()
        self.optimizer_g.step()
        
        # ==================
        # Train Binary Discriminator (always)
        # ==================
        self.optimizer_d_binary.zero_grad()
        
        pred_real_binary = self.binary_discriminator(self.extract_patches(images, masks))
        loss_d_binary_real = self.criterion_adv(pred_real_binary, real_labels)
        
        pred_fake_binary_d = self.binary_discriminator(fake_patches.detach())
        loss_d_binary_fake = self.criterion_adv(pred_fake_binary_d, fake_labels)
        
        loss_d_binary = (loss_d_binary_real + loss_d_binary_fake) / 2
        loss_d_binary.backward()
        self.optimizer_d_binary.step()
        
        # ==================
        # Train Localizer Discriminator (with update ratio, after warmup)
        # ==================
        # L_D = L_real_conf + L_fake_conf + 1.0 * L_fake_loc + 0.5 * L_fake_iou
        # Note: NO bbox loss on real images (no null box enforcement)
        
        self.d_update_counter += 1
        update_d_loc = (self.d_update_counter % self.d_update_ratio == 0)
        
        # Skip localizer updates during very early warmup
        skip_d_in_warmup = warmup_progress < 0.2
        
        if update_d_loc and not skip_d_in_warmup:
            self.optimizer_d_localizer.zero_grad()
            
            # Real images - should have LOW confidence (no fake region)
            # NO bbox loss on real images - if human can't see fake, they draw random box
            bbox_real, conf_real = self.localizer_discriminator(images)
            target_conf_real = self.get_smooth_labels(batch_size, is_real=False)  # target = 0
            loss_d_loc_real_conf = self.criterion_conf(conf_real, target_conf_real)
            
            # Fake/completed images - should have HIGH confidence (has fake region)
            completed_detached = completed_images.detach()
            bbox_fake, conf_fake = self.localizer_discriminator(completed_detached)
            target_conf_fake = self.get_smooth_labels(batch_size, is_real=True)  # target = 1
            loss_d_loc_fake_conf = self.criterion_conf(conf_fake, target_conf_fake)
            
            # Localization loss on fake images ONLY (with FIXED weight 1.0)
            loss_d_loc_bbox = self.criterion_loc(bbox_fake.squeeze(1), target_bboxes_loc)
            
            # IoU loss on fake images ONLY (with FIXED weight 0.5)
            iou_fake = self.compute_iou(bbox_fake, target_bboxes_loc)
            loss_d_loc_iou = 1.0 - iou_fake.mean()
            
            # Combined discriminator loss with FIXED weights:
            # L_D = L_real_conf + L_fake_conf + 1.0 * L_fake_loc + 0.5 * L_fake_iou
            loss_d_localizer = (loss_d_loc_real_conf + 
                               loss_d_loc_fake_conf + 
                               self.d_weight_bbox * loss_d_loc_bbox + 
                               self.d_weight_iou * loss_d_loc_iou)
            
            loss_d_localizer.backward()
            self.optimizer_d_localizer.step()
        else:
            loss_d_localizer = torch.tensor(0.0)
            conf_real = torch.zeros(batch_size, 1)
            conf_fake = torch.ones(batch_size, 1)
            bbox_fake = bbox_pred
            iou_fake = iou
        
        # Track if localizer D was actually updated
        d_loc_actually_updated = update_d_loc and not skip_d_in_warmup
        
        return {
            'loss_g': loss_g.item(),
            'loss_d': loss_d_binary.item(),  # Binary discriminator loss
            'loss_d_localizer': loss_d_localizer.item() if d_loc_actually_updated else 0.0,
            'loss_rec': loss_rec.item(),
            'loss_adv': loss_adv.item(),  # Binary adversarial loss
            'loss_conf_g': loss_conf_g.item(),
            'loss_loc_g': loss_loc_g.item(),
            'loss_iou_g': loss_iou_g.item(),
            'd_real_acc': (pred_real_binary > 0.5).float().mean().item(),  # Binary D accuracy
            'd_fake_acc': (pred_fake_binary_d < 0.5).float().mean().item(),
            'loc_real_acc': (conf_real < 0.5).float().mean().item(),  # Localizer accuracy
            'loc_fake_acc': (conf_fake > 0.5).float().mean().item(),
            'completed_images': completed_images,
            'masked_images': masked_images,
            'reconstructed': reconstructed,
            'masks': masks,
            'bbox_pred': bbox_fake if d_loc_actually_updated else bbox_pred,
            'bbox_true': target_bboxes_loc,
            'confidence': conf_fake if d_loc_actually_updated else conf_pred,
            'iou': iou_fake.mean().item() if d_loc_actually_updated else iou.mean().item(),
            'stage': 2,
            'd_updated': True,  # Binary D always updates
            'd_loc_updated': d_loc_actually_updated,
            'warmup_progress': warmup_progress,
            'in_warmup': self.is_in_warmup()
        }
    
    def train_step(self, images):
        """Main training step - routes to appropriate stage"""
        batch_size = images.size(0)
        
        # Generate masks
        masks, target_bboxes = self.generate_square_mask(batch_size)
        
        # Determine current stage
        stage = self.get_current_stage()
        self.current_stage = stage
        
        # Get blend factor for potential mixed training
        blend = self.get_blend_factor()
        
        if stage == 1:
            metrics = self.train_step_stage1(images, masks, target_bboxes)
        else:
            metrics = self.train_step_stage2(images, masks, target_bboxes)
        
        metrics['blend_factor'] = blend
        self.global_step += 1
        
        return metrics
    
    def check_transition(self, epoch):
        """
        Check if we're transitioning stages and handle it.
        Call this at the start of each epoch.
        """
        old_stage = self.current_stage
        self.current_epoch = epoch
        new_stage = self.get_current_stage()
        
        if old_stage == 1 and new_stage == 2:
            print(f"\n{'='*60}")
            print(f"TRANSITIONING from Stage 1 (Binary) to Stage 2 (Localizer)")
            print(f"Epoch {epoch}: Localizer discriminator now active")
            print(f"{'='*60}\n")
            
            # Optional: Reset discriminator learning rate for fresh start
            for param_group in self.optimizer_d_localizer.param_groups:
                param_group['lr'] = self.lr_d
        
        return new_stage
    
    def save_checkpoint(self, save_path, metrics=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'current_stage': self.current_stage,
            'encoder_state': self.encoder.state_dict(),
            'decoder_state': self.decoder.state_dict(),
            'binary_discriminator_state': self.binary_discriminator.state_dict(),
            'localizer_discriminator_state': self.localizer_discriminator.state_dict(),
            'optimizer_g_state': self.optimizer_g.state_dict(),
            'optimizer_d_binary_state': self.optimizer_d_binary.state_dict(),
            'optimizer_d_localizer_state': self.optimizer_d_localizer.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss,
            'best_iou': self.best_iou,
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
        self.binary_discriminator.load_state_dict(checkpoint['binary_discriminator_state'])
        self.localizer_discriminator.load_state_dict(checkpoint['localizer_discriminator_state'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state'])
        self.optimizer_d_binary.load_state_dict(checkpoint['optimizer_d_binary_state'])
        self.optimizer_d_localizer.load_state_dict(checkpoint['optimizer_d_localizer_state'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.current_stage = checkpoint.get('current_stage', 1)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.best_iou = checkpoint.get('best_iou', 0.0)
        
        print(f"Loaded checkpoint from epoch {self.current_epoch} (Stage {self.current_stage})")
        return checkpoint.get('metrics', {})
    
    def load_baseline_checkpoint(self, checkpoint_path, start_epoch=None, warmup_epochs=None):
        """
        Load a baseline checkpoint to initialize encoder/decoder.
        Useful for starting Stage 2 from a pre-trained baseline model.
        
        Args:
            checkpoint_path: Path to baseline checkpoint
            start_epoch: Epoch to start from (default: transition_epoch)
            warmup_epochs: Number of warmup epochs for localizer (default: config value)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder_state'])
        self.decoder.load_state_dict(checkpoint['decoder_state'])
        
        # Optionally load binary discriminator if available
        if 'discriminator_state' in checkpoint:
            self.binary_discriminator.load_state_dict(checkpoint['discriminator_state'])
        
        # Set starting epoch
        if start_epoch is not None:
            self.current_epoch = start_epoch
        else:
            self.current_epoch = self.transition_epoch
        
        # Configure warmup for localizer
        if warmup_epochs is not None:
            self.warmup_epochs = warmup_epochs
        
        # Set warmup start to current epoch
        self.warmup_start_epoch = self.current_epoch
        
        print(f"Loaded baseline checkpoint (encoder/decoder weights)")
        print(f"  Starting at epoch {self.current_epoch}")
        if self.warmup_epochs > 0:
            print(f"  Warmup: {self.warmup_epochs} epochs (until epoch {self.current_epoch + self.warmup_epochs})")
        
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
