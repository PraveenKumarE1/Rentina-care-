"""Advanced classification models for DR grading using pre-trained backbones.

Supports EfficientNet, ConvNeXt, Vision Transformer (ViT), and ensemble models
with proper heads for 5-class DR grading and binary diabetes risk.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score


# ============================================================================
# Model Architectures
# ============================================================================

class GeM(nn.Module):
    """Generalized Mean Pooling (GeM) - better than GAP for fine-grained classification."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6, trainable: bool = True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p) if trainable else p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)


class MultiSampleDropout(nn.Module):
    """Multi-sample dropout for better uncertainty estimation."""
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.5, num_samples: int = 5):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_features, out_features)
        self.num_samples = num_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return self.fc(x)
        outputs = []
        for _ in range(self.num_samples):
            outputs.append(self.fc(self.dropout(x)))
        return torch.stack(outputs).mean(0)


class ClassifierHead(nn.Module):
    """Advanced classifier head with options for pooling, dropout, and calibration."""
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        use_gem: bool = True,
        use_multisample_dropout: bool = False,
        num_msd_samples: int = 5,
    ):
        super().__init__()
        self.use_gem = use_gem
        if use_gem:
            self.pool = GeM()
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.pre_fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        if use_multisample_dropout:
            self.classifier = MultiSampleDropout(hidden_dim, num_classes, dropout, num_msd_samples)
        else:
            self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = self.pool(x).flatten(1)
        elif x.dim() == 3:  # ViT-style (B, N, C)
            x = x.mean(dim=1)
        x = self.pre_fc(x)
        return self.classifier(x)


class DRClassifier(nn.Module):
    """DR classifier with pre-trained backbone."""
    
    BACKBONE_CONFIGS = {
        'efficientnet_b0': {'model': 'efficientnet_b0', 'features': 1280, 'img_size': 224},
        'efficientnet_b1': {'model': 'efficientnet_b1', 'features': 1280, 'img_size': 240},
        'efficientnet_b2': {'model': 'efficientnet_b2', 'features': 1408, 'img_size': 260},
        'efficientnet_b3': {'model': 'efficientnet_b3', 'features': 1536, 'img_size': 300},
        'efficientnet_b4': {'model': 'efficientnet_b4', 'features': 1792, 'img_size': 380},
        'convnext_tiny': {'model': 'convnext_tiny', 'features': 768, 'img_size': 224},
        'convnext_small': {'model': 'convnext_small', 'features': 768, 'img_size': 224},
        'convnext_base': {'model': 'convnext_base', 'features': 1024, 'img_size': 224},
        'vit_small_patch16_224': {'model': 'vit_small_patch16_224', 'features': 384, 'img_size': 224},
        'vit_base_patch16_224': {'model': 'vit_base_patch16_224', 'features': 768, 'img_size': 224},
        'vit_large_patch16_224': {'model': 'vit_large_patch16_224', 'features': 1024, 'img_size': 224},
        'swin_tiny_patch4_window7_224': {'model': 'swin_tiny_patch4_window7_224', 'features': 768, 'img_size': 224},
        'swin_small_patch4_window7_224': {'model': 'swin_small_patch4_window7_224', 'features': 768, 'img_size': 224},
    }
    
    def __init__(
        self,
        backbone: str = 'efficientnet_b0',
        num_classes: int = 5,
        pretrained: bool = True,
        head_hidden_dim: int = 512,
        head_dropout: float = 0.3,
        use_gem: bool = True,
        use_multisample_dropout: bool = False,
        freeze_backbone: bool = False,
        freeze_epochs: int = 0,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.freeze_epochs = freeze_epochs
        self.current_epoch = 0
        
        if backbone not in self.BACKBONE_CONFIGS:
            raise ValueError(f"Unknown backbone: {backbone}. Available: {list(self.BACKBONE_CONFIGS.keys())}")
        
        config = self.BACKBONE_CONFIGS[backbone]
        self.img_size = config['img_size']
        self.backbone_features = config['features']
        
        # Create backbone
        self.backbone = timm.create_model(
            config['model'],
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='',  # No pooling, we'll do it in head
        )
        
        # Freeze backbone initially if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        # Classifier head
        self.head = ClassifierHead(
            in_features=self.backbone_features,
            num_classes=num_classes,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            use_gem=use_gem,
            use_multisample_dropout=use_multisample_dropout,
        )
        
        # Temperature scaling parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def set_epoch(self, epoch: int):
        """Set current epoch for progressive unfreezing."""
        self.current_epoch = epoch
        if self.freeze_epochs > 0 and epoch >= self.freeze_epochs:
            self._unfreeze_backbone()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone forward
        features = self.backbone(x)
        
        # Handle different backbone output formats
        if isinstance(features, (list, tuple)):
            features = features[-1]  # Use last feature map
        
        # ViT outputs (B, N, C) - use cls token or mean pool
        if features.dim() == 3 and features.size(1) > 1:
            # Use class token if available (first token), otherwise mean pool
            features = features[:, 0]  # cls token
        
        # Classifier head
        logits = self.head(features)
        
        # Temperature scaling (only during eval for calibration)
        if not self.training:
            logits = logits / self.temperature.clamp(min=0.1)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.dim() == 3 and features.size(1) > 1:
            features = features[:, 0]
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return features


class EnsembleDRClassifier(nn.Module):
    """Ensemble of multiple DR classifiers."""
    
    def __init__(
        self,
        backbones: List[str] = None,
        num_classes: int = 5,
        pretrained: bool = True,
        head_hidden_dim: int = 512,
        head_dropout: float = 0.3,
        aggregation: str = 'mean',  # mean, max, weighted
    ):
        super().__init__()
        self.backbones = backbones or ['efficientnet_b0', 'convnext_tiny', 'vit_small_patch16_224']
        self.num_classes = num_classes
        self.aggregation = aggregation
        
        self.models = nn.ModuleList()
        for backbone in self.backbones:
            model = DRClassifier(
                backbone=backbone,
                num_classes=num_classes,
                pretrained=pretrained,
                head_hidden_dim=head_hidden_dim,
                head_dropout=head_dropout,
            )
            self.models.append(model)
        
        # Learnable weights for weighted aggregation
        if aggregation == 'weighted':
            self.ensemble_weights = nn.Parameter(torch.ones(len(self.backbones)) / len(self.backbones))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_list = []
        for model in self.models:
            logits = model(x)
            logits_list.append(logits)
        
        logits_stack = torch.stack(logits_list, dim=0)  # (num_models, B, C)
        
        if self.aggregation == 'mean':
            return logits_stack.mean(0)
        elif self.aggregation == 'max':
            return logits_stack.max(0)[0]
        elif self.aggregation == 'weighted':
            weights = F.softmax(self.ensemble_weights, dim=0)
            return (logits_stack * weights.view(-1, 1, 1)).sum(0)
        else:
            return logits_stack.mean(0)
    
    def get_individual_logits(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get logits from each model separately."""
        return [model(x) for model in self.models]


# ============================================================================
# Training Utilities
# ============================================================================

@dataclass
class ClassifierConfig:
    """Configuration for DR classifier."""
    backbone: str = 'efficientnet_b0'
    num_classes: int = 5
    pretrained: bool = True
    img_size: int = 224
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    freeze_backbone: bool = False
    freeze_epochs: int = 5
    use_gem: bool = True
    use_multisample_dropout: bool = False
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir: str = "models/classification"
    num_workers: int = 4
    val_split: float = 0.2
    seed: int = 42


class MixupCutmix:
    """Mixup and CutMix augmentation."""
    def __init__(self, mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0, prob: float = 0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
    
    def __call__(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        images, targets = batch
        batch_size = images.size(0)
        
        if np.random.random() > self.prob:
            return images, targets
        
        # Decide between mixup and cutmix
        if np.random.random() < 0.5 and self.mixup_alpha > 0:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            index = torch.randperm(batch_size).to(images.device)
            images = lam * images + (1 - lam) * images[index]
            targets_a, targets_b = targets, targets[index]
            return images, (targets_a, targets_b, lam)
        elif self.cutmix_alpha > 0:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            index = torch.randperm(batch_size).to(images.device)
            
            # Generate random box
            H, W = images.shape[-2:]
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
            
            images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
            
            # Adjust lambda
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
            targets_a, targets_b = targets, targets[index]
            return images, (targets_a, targets_b, lam)
        
        return images, targets


def mixup_cutmix_criterion(
    criterion: nn.Module,
    outputs: torch.Tensor,
    targets: Union[torch.Tensor, Tuple],
) -> torch.Tensor:
    """Compute loss with mixup/cutmix targets."""
    if isinstance(targets, tuple) and len(targets) == 3:
        targets_a, targets_b, lam = targets
        return lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
    return criterion(outputs, targets)


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    mixup_cutmix: Optional[MixupCutmix] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        
        if mixup_cutmix is not None:
            images, targets = mixup_cutmix((images, targets.to(device)))
        else:
            targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = mixup_cutmix_criterion(criterion, outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = mixup_cutmix_criterion(criterion, outputs, targets)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
        # For metrics, use original targets if mixup/cutmix
        if isinstance(targets, tuple):
            targets = targets[0]  # Use first target for metric
        preds = outputs.argmax(1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())
    
    metrics = {
        'loss': total_loss / len(loader.dataset),
        'accuracy': accuracy_score(all_targets, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_targets, all_preds),
        'quadratic_kappa': cohen_kappa_score(all_targets, all_preds, weights='quadratic'),
    }
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item() * images.size(0)
        
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(1)
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    metrics = {
        'loss': total_loss / len(loader.dataset),
        'accuracy': accuracy_score(all_targets, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_targets, all_preds),
        'quadratic_kappa': cohen_kappa_score(all_targets, all_preds, weights='quadratic'),
    }
    return metrics, np.array(all_probs), np.array(all_targets)


def calibrate_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Find optimal temperature for calibration."""
    model.eval()
    logits_list = []
    targets_list = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            logits = model(images)
            logits_list.append(logits.cpu())
            targets_list.append(targets)
    
    logits = torch.cat(logits_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    
    # Optimize temperature
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    def eval_loss():
        optimizer.zero_grad()
        scaled_logits = logits / temperature.clamp(min=0.05)
        loss = F.cross_entropy(scaled_logits, targets)
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    return temperature.item()


# ============================================================================
# Inference Pipeline
# ============================================================================

@dataclass
class InferenceConfig:
    """Configuration for inference."""
    model_path: str
    backbone: str = 'efficientnet_b0'
    num_classes: int = 5
    img_size: int = 224
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tta: bool = True  # Test-time augmentation
    temperature: float = 1.0


class DRInference:
    """Inference pipeline for DR classification."""
    
    DR_LABELS = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR']
    DIABETES_LABELS = ['No Diabetes', 'Diabetes']
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = self._load_model()
        self.transform = self._get_transform()
    
    def _load_model(self) -> nn.Module:
        """Load trained model."""
        checkpoint = torch.load(self.config.model_path, map_location=self.device, weights_only=False)
        
        # Handle both old and new checkpoint formats
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
            model_config = {
                'backbone': checkpoint.get('backbone', self.config.backbone),
                'num_classes': checkpoint.get('classes', self.config.num_classes),
            }
        else:
            state_dict = checkpoint
            model_config = {
                'backbone': self.config.backbone,
                'num_classes': self.config.num_classes,
            }
        
        model = DRClassifier(**model_config)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        # Set temperature if available
        if 'temperature' in checkpoint:
            model.temperature.data = torch.tensor(checkpoint['temperature'])
        elif self.config.temperature != 1.0:
            model.temperature.data = torch.tensor(self.config.temperature)
        
        return model
    
    def _get_transform(self):
        """Get preprocessing transform."""
        import torchvision.transforms as T
        return T.Compose([
            T.Resize((self.config.img_size, self.config.img_size), 
                     interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _tta_transforms(self):
        """Get TTA transforms."""
        import torchvision.transforms as T
        base = self.transform
        return [
            base,  # original
            T.Compose([T.RandomHorizontalFlip(p=1.0), *base.transforms[1:]]),  # flip
            T.Compose([T.RandomRotation(10), *base.transforms[1:]]),  # rotate
        ]
    
    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
        """Predict DR grade from image."""
        # Load image
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert('RGB')
        else:
            img = image.convert('RGB')
        
        # TTA predictions
        if self.config.tta:
            probs_list = []
            for transform in self._tta_transforms():
                tensor = transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = F.softmax(logits, dim=1)
                    probs_list.append(probs.cpu().numpy())
            probs = np.mean(probs_list, axis=0)[0]
        else:
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        pred_idx = int(np.argmax(probs))
        labels = self.DR_LABELS if self.config.num_classes == 5 else self.DIABETES_LABELS
        
        return {
            'level': pred_idx,
            'label': labels[pred_idx],
            'confidence': float(probs[pred_idx]),
            'referable': pred_idx >= 2,
            'probabilities': probs.tolist(),
            'method': f'trained {self.config.backbone} checkpoint',
        }
    
    def predict_batch(self, images: List[Union[str, np.ndarray, Image.Image]]) -> List[Dict]:
        """Predict on batch of images."""
        return [self.predict(img) for img in images]


# ============================================================================
# Convenience Functions
# ============================================================================

def create_classifier(config: ClassifierConfig) -> DRClassifier:
    """Create a DR classifier."""
    return DRClassifier(
        backbone=config.backbone,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        freeze_epochs=config.freeze_epochs,
        use_gem=config.use_gem,
        use_multisample_dropout=config.use_multisample_dropout,
    )


def create_ensemble_classifier(
    backbones: List[str] = None,
    num_classes: int = 5,
    pretrained: bool = True,
) -> EnsembleDRClassifier:
    """Create an ensemble classifier."""
    return EnsembleDRClassifier(
        backbones=backbones,
        num_classes=num_classes,
        pretrained=pretrained,
    )


def load_classifier(model_path: str, **kwargs) -> DRInference:
    """Load a trained classifier for inference."""
    config = InferenceConfig(model_path=model_path, **kwargs)
    return DRInference(config)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test DR classifier")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--input", type=str, help="Input image path")
    args = parser.parse_args()
    
    # Test model creation
    config = ClassifierConfig(backbone=args.backbone, num_classes=args.num_classes)
    model = create_classifier(config)
    print(f"Model: {args.backbone}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward
    dummy = torch.randn(1, 3, config.img_size, config.img_size)
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape: {out.shape}")
    
    if args.input:
        inference = load_classifier("models/classification/best.pt", backbone=args.backbone)
        result = inference.predict(args.input)
        print(f"Prediction: {result}")