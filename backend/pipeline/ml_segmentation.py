"""ML-based segmentation for retinal structures.

Provides U-Net based segmentation for:
- Optic disc
- Fovea
- Blood vessels
- Lesions (microaneurysms, exudates, hemorrhages, neovascularization)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ============================================================================
# U-Net Architecture
# ============================================================================

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 to match x2 size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate for U-Net skip connections."""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """Attention U-Net for retinal segmentation."""
    def __init__(self, n_channels: int = 3, n_classes: int = 7, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Attention gates
        self.att1 = AttentionGate(512 // factor, 512, 256)
        self.att2 = AttentionGate(256 // factor, 256, 128)
        self.att3 = AttentionGate(128 // factor, 128, 64)
        self.att4 = AttentionGate(64, 64, 32)
        
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, self.att1(x5, x4))
        x = self.up2(x, self.att2(x, x3))
        x = self.up3(x, self.att3(x, x2))
        x = self.up4(x, self.att4(x, x1))
        
        logits = self.outc(x)
        return logits


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with ResNet backbone for segmentation."""
    def __init__(self, n_channels: int = 3, n_classes: int = 7, output_stride: int = 16):
        super().__init__()
        # Use a simplified version - in practice, use pretrained ResNet
        self.backbone = self._make_backbone(n_channels)
        self.aspp = self._make_aspp(256, 256)
        self.decoder = self._make_decoder(256, 48, n_classes)
        
    def _make_backbone(self, in_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            self._make_layer(64, 64, 3),
            self._make_layer(64, 128, 4, stride=2),
            self._make_layer(128, 256, 6, stride=2),
            self._make_layer(256, 512, 3, stride=1, dilation=2),
        )
    
    def _make_layer(self, in_c: int, out_c: int, blocks: int, stride: int = 1, dilation: int = 1):
        layers = []
        layers.append(self._res_block(in_c, out_c, stride, dilation))
        for _ in range(1, blocks):
            layers.append(self._res_block(out_c, out_c, 1, dilation))
        return nn.Sequential(*layers)
    
    def _res_block(self, in_c: int, out_c: int, stride: int, dilation: int):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c),
        )
    
    def _make_aspp(self, in_c: int, out_c: int):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
    
    def _make_decoder(self, in_c: int, low_c: int, n_classes: int):
        return nn.Sequential(
            nn.Conv2d(in_c + low_c, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified forward - real implementation would use proper backbone
        features = self.backbone(x)
        x = self.aspp(features)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        return x


# ============================================================================
# Multi-task Segmentation Model
# ============================================================================

class RetinalSegmentationNet(nn.Module):
    """Multi-task segmentation network with shared encoder and task-specific decoders."""
    
    def __init__(self, n_channels: int = 3, tasks: List[str] = None):
        super().__init__()
        self.tasks = tasks or [
            'optic_disc', 'fovea', 'vessels', 
            'microaneurysms', 'exudates', 'hemorrhages', 'neovascularization'
        ]
        self.n_classes = len(self.tasks)
        
        # Shared encoder
        self.encoder = nn.Sequential(
            DoubleConv(n_channels, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024),
        )
        
        # Task-specific decoders
        self.decoders = nn.ModuleDict()
        for task in self.tasks:
            self.decoders[task] = nn.Sequential(
                Up(1024, 512),
                Up(512, 256),
                Up(256, 128),
                Up(128, 64),
                OutConv(64, 1)
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)
        x5 = self.encoder[4](x4)
        
        # Decode for each task
        outputs = {}
        for task in self.tasks:
            decoder = self.decoders[task]
            d1 = decoder[0](x5, x4)
            d2 = decoder[1](d1, x3)
            d3 = decoder[2](d2, x2)
            d4 = decoder[3](d3, x1)
            out = decoder[4](d4)
            outputs[task] = torch.sigmoid(out)
        
        return outputs


# ============================================================================
# Segmentation Pipeline
# ============================================================================

@dataclass
class SegmentationConfig:
    """Configuration for segmentation."""
    model_type: str = "attention_unet"  # attention_unet, multitask, deeplabv3plus
    input_size: Tuple[int, int] = (512, 512)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir: str = "models/segmentation"
    confidence_threshold: float = 0.5
    min_area: Dict[str, int] = None
    
    def __post_init__(self):
        if self.min_area is None:
            self.min_area = {
                'optic_disc': 100,
                'fovea': 50,
                'vessels': 20,
                'microaneurysms': 5,
                'exudates': 10,
                'hemorrhages': 15,
                'neovascularization': 10,
            }


class MLSegmenter:
    """ML-based retinal segmentation pipeline."""
    
    # Class mapping for outputs
    CLASS_NAMES = [
        'optic_disc', 'fovea', 'vessels',
        'microaneurysms', 'exudates', 'hemorrhages', 'neovascularization'
    ]
    
    def __init__(self, config: Optional[SegmentationConfig] = None):
        self.config = config or SegmentationConfig()
        self.device = torch.device(self.config.device)
        self.model = self._create_model()
        self._load_weights()
        self.model.eval()
    
    def _create_model(self) -> nn.Module:
        """Create the segmentation model."""
        if self.config.model_type == "attention_unet":
            model = AttentionUNet(n_channels=3, n_classes=len(self.CLASS_NAMES))
        elif self.config.model_type == "multitask":
            model = RetinalSegmentationNet(n_channels=3, tasks=self.CLASS_NAMES)
        elif self.config.model_type == "deeplabv3plus":
            model = DeepLabV3Plus(n_channels=3, n_classes=len(self.CLASS_NAMES))
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        return model.to(self.device)
    
    def _load_weights(self):
        """Load trained weights if available."""
        model_dir = Path(self.config.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        weight_path = model_dir / f"{self.config.model_type}_retinal.pt"
        if weight_path.exists():
            state_dict = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded segmentation weights from {weight_path}")
        else:
            print(f"No trained weights found at {weight_path}. Using random initialization.")
    
    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocess image for model input."""
        h, w = image.shape[:2]
        # Resize maintaining aspect ratio with padding
        target_h, target_w = self.config.input_size
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad to target size
        padded = np.zeros((target_h, target_w, 3), dtype=np.float32)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        padded = (padded - mean) / std
        
        tensor = torch.from_numpy(padded.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor, (y_offset, x_offset, new_h, new_w)
    
    def _postprocess(self, logits: torch.Tensor, padding_info: Tuple, 
                     original_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Postprocess model outputs to original image size."""
        y_off, x_off, new_h, new_w = padding_info
        orig_h, orig_w = original_shape[:2]
        
        # Remove padding and resize
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (C, H, W)
        
        results = {}
        for i, class_name in enumerate(self.CLASS_NAMES):
            prob_map = probs[i]
            # Crop padding
            prob_map = prob_map[y_off:y_off+new_h, x_off:x_off+new_w]
            # Resize to original
            prob_map = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # Threshold
            mask = prob_map > self.config.confidence_threshold
            
            # Remove small components
            min_area = self.config.min_area.get(class_name, 10)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8)
            cleaned = np.zeros_like(mask)
            for label_idx in range(1, num_labels):
                if stats[label_idx, cv2.CC_STAT_AREA] >= min_area:
                    cleaned[labels == label_idx] = True
            
            results[class_name] = {
                'mask': cleaned,
                'probability_map': prob_map,
                'count': int(np.sum(cleaned)),
                'area': float(np.sum(cleaned) / cleaned.size)
            }
        
        return results
    
    def segment(self, image: np.ndarray) -> Dict[str, dict]:
        """Segment a fundus image."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        original_shape = image.shape
        
        with torch.no_grad():
            tensor, padding_info = self._preprocess(image)
            if isinstance(self.model, RetinalSegmentationNet):
                outputs = self.model(tensor)
                # Convert dict of tensors to stacked tensor for postprocessing
                logits = torch.stack([torch.logit(v.clamp(1e-6, 1-1e-6)) for v in outputs.values()], dim=1)
            else:
                logits = self.model(tensor)
            
            results = self._postprocess(logits, padding_info, original_shape)
        
        # Convert to expected format
        formatted = {}
        for class_name in self.CLASS_NAMES:
            r = results[class_name]
            formatted[class_name] = {
                'mask': r['mask'],
                'count': r['count'],
                'area': r['area']
            }
        
        # Add derived features
        formatted['features'] = {
            'vesselDensity': formatted['vessels']['area'],
            'tortuosity': self._compute_tortuosity(formatted['vessels']['mask']),
            'cupToDiscRatio': self._compute_cdr(
                formatted['optic_disc']['mask'], formatted['vessels']['mask']
            )
        }
        
        return formatted
    
    def _compute_tortuosity(self, vessel_mask: np.ndarray) -> float:
        """Compute vessel tortuosity."""
        # Simplified: use curvature of skeleton
        if vessel_mask.sum() < 10:
            return float('nan')
        # Skeletonize
        skeleton = cv2.ximgproc.thinning(vessel_mask.astype(np.uint8)) if hasattr(cv2, 'ximgproc') else vessel_mask
        # Find contours
        contours, _ = cv2.findContours(skeleton.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return float('nan')
        # Approximate tortuosity as total length / straight-line distance
        total_len = sum(cv2.arcLength(c, False) for c in contours)
        # Bounding box diagonal as reference
        ys, xs = np.where(skeleton)
        if len(xs) < 2:
            return float('nan')
        straight = np.sqrt((xs.max() - xs.min())**2 + (ys.max() - ys.min())**2)
        return float(total_len / (straight + 1e-6))
    
    def _compute_cdr(self, disc_mask: np.ndarray, vessel_mask: np.ndarray) -> float:
        """Estimate cup-to-disc ratio."""
        if disc_mask.sum() < 100:
            return float('nan')
        # Find disc center and radius
        moments = cv2.moments(disc_mask.astype(np.uint8))
        if moments['m00'] == 0:
            return float('nan')
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        radius = int(np.sqrt(moments['m00'] / np.pi))
        
        # Cup region: central area with fewer vessels
        y, x = np.ogrid[:disc_mask.shape[0], :disc_mask.shape[1]]
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        cup_mask = (dist_from_center < radius * 0.5) & (vessel_mask == 0) & disc_mask
        cup_area = cup_mask.sum()
        disc_area = disc_mask.sum()
        
        return float(cup_area / (disc_area + 1e-6))


# ============================================================================
# Classical Segmentation Fallback
# ============================================================================

def classical_segmentation(image: np.ndarray, quality_passed: bool) -> Dict[str, dict]:
    """Classical rule-based segmentation (fallback)."""
    if image.ndim == 3:
        green = image[:, :, 1].astype(np.float32) / 255.0
        bright = image.mean(axis=2).astype(np.float32) / 255.0
    else:
        green = image.astype(np.float32) / 255.0
        bright = green
    
    green = cv2.normalize(green, None, 0, 1, cv2.NORM_MINMAX)
    bright = cv2.normalize(bright, None, 0, 1, cv2.NORM_MINMAX)
    
    # Optic disc - brightest region
    optic_disc = bright > np.percentile(bright, 99)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    optic_disc = cv2.morphologyEx(optic_disc.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    optic_disc = _largest_component(optic_disc)
    
    # Exudates - bright lesions
    exudates = bright > np.percentile(bright, 98)
    exudates = cv2.morphologyEx(exudates.astype(np.uint8), cv2.MORPH_OPEN, 
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    exudates = (exudates > 0).astype(np.uint8) & (optic_disc == 0)
    
    # Hemorrhages and microaneurysms - dark lesions
    dark = 1.0 - green
    hemorrhages = dark > np.percentile(dark, 97)
    hemorrhages = cv2.morphologyEx(hemorrhages.astype(np.uint8), cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    hemorrhages = (hemorrhages > 0).astype(np.uint8)
    
    microaneurysms = dark > np.percentile(dark, 99)
    microaneurysms = cv2.morphologyEx(microaneurysms.astype(np.uint8), cv2.MORPH_OPEN,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    microaneurysms = (microaneurysms > 0).astype(np.uint8) & (optic_disc == 0)
    
    # Vessels
    inverse_green = (255 * (1.0 - green)).astype(np.uint8)
    vessels = cv2.adaptiveThreshold(
        inverse_green, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_OPEN, 
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    vessels = (vessels > 0).astype(np.uint8) & (optic_disc == 0)
    
    # Fovea - darkest region temporal to optic disc (simplified)
    fovea_mask = np.zeros_like(green, dtype=bool)
    if optic_disc.sum() > 0:
        disc_y, disc_x = np.where(optic_disc)
        disc_cy, disc_cx = int(disc_y.mean()), int(disc_x.mean())
        # Fovea is typically temporal (left for right eye, right for left eye)
        # Simplified: search in temporal region
        temporal_region = green[:, disc_cx:] if disc_cx < green.shape[1] // 2 else green[:, :disc_cx]
        if temporal_region.size > 0:
            fovea_y_local, fovea_x_local = np.unravel_index(
                np.argmin(temporal_region), temporal_region.shape)
            fovea_x = fovea_x_local + (disc_cx if disc_cx < green.shape[1] // 2 else 0)
            fovea_y = fovea_y_local
            cv2.circle(fovea_mask, (fovea_x, fovea_y), 20, True, -1)
    
    props = lambda m: {'mask': m.astype(bool), 'count': int(np.sum(m)), 'area': float(np.sum(m) / m.size)}
    
    result = {
        'optic_disc': props(optic_disc),
        'fovea': props(fovea_mask),
        'vessels': props(vessels),
        'microaneurysms': props(microaneurysms),
        'exudates': props(exudates),
        'hemorrhages': props(hemorrhages),
        'neovascularization': props(np.zeros_like(green, dtype=bool)),
    }
    
    result['features'] = {
        'vesselDensity': result['vessels']['area'],
        'tortuosity': float('nan'),
        'cupToDiscRatio': float('nan'),
    }
    
    if not quality_passed:
        result['warning'] = 'Interpret results only after a gradable acquisition.'
    
    return result


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(np.uint8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    result = np.zeros_like(mask, dtype=np.uint8)
    result[labels == largest] = 1
    return result


# ============================================================================
# Convenience Functions
# ============================================================================

def create_segmenter(config: Optional[SegmentationConfig] = None) -> MLSegmenter:
    """Create an ML segmenter."""
    return MLSegmenter(config)


def segment_fundus(image: np.ndarray, 
                    quality_passed: bool = True,
                    use_ml: bool = True,
                    config: Optional[SegmentationConfig] = None) -> Dict[str, dict]:
    """High-level function to segment a fundus image."""
    if use_ml:
        try:
            segmenter = create_segmenter(config)
            return segmenter.segment(image)
        except Exception as e:
            print(f"ML segmentation failed, falling back to classical: {e}")
            return classical_segmentation(image, quality_passed)
    else:
        return classical_segmentation(image, quality_passed)


if __name__ == "__main__":
    # Demo
    import argparse
    parser = argparse.ArgumentParser(description="Test ML segmentation")
    parser.add_argument("--input", type=str, help="Input image path")
    parser.add_argument("--use-ml", action="store_true", help="Use ML models")
    args = parser.parse_args()
    
    if args.input:
        img = cv2.imread(args.input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Synthetic
        h, w = 512, 512
        y, x = np.ogrid[:h, :w]
        mask = ((x - w//2) ** 2 + (y - h//2) ** 2) < (min(h, w) * 0.4) ** 2
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[mask] = [100, 50, 30]
        img[~mask] = [10, 5, 3]
    
    print(f"Input shape: {img.shape}")
    result = segment_fundus(img, quality_passed=True, use_ml=args.use_ml)
    
    for name, data in result.items():
        if name != 'features':
            print(f"  {name}: count={data['count']}, area={data['area']:.4f}")
    print(f"  features: {result['features']}")