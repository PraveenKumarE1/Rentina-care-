"""AI-based image enhancement for retinal images.

Provides super-resolution, denoising, illumination correction, and vessel enhancement
using lightweight deep learning models optimized for fundus images.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps


# ============================================================================
# Model Architectures
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with optional attention."""
    def __init__(self, channels: int, use_attention: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 8, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 8, channels, 1),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_attention:
            attn = self.attention(out)
            out = out * attn
        out += residual
        return self.relu(out)


class UpsampleBlock(nn.Module):
    """Upsample block for super-resolution."""
    def __init__(self, in_channels: int, out_channels: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(in_channels, out_channels * (scale ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.relu(x)


class SuperResolutionNet(nn.Module):
    """Lightweight super-resolution network (ESRGAN-inspired)."""
    def __init__(self, scale: int = 2, num_blocks: int = 16, channels: int = 64):
        super().__init__()
        self.scale = scale
        self.head = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(*[
            ResidualBlock(channels, use_attention=(i % 4 == 3))
            for i in range(num_blocks)
        ])
        self.up = nn.Sequential(
            UpsampleBlock(channels, channels, scale),
            UpsampleBlock(channels, channels, scale // 2) if scale == 4 else nn.Identity(),
        )
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.up(x)
        return self.tail(x)


class DenoisingNet(nn.Module):
    """Lightweight denoising network (DnCNN-inspired)."""
    def __init__(self, channels: int = 64, num_layers: int = 17):
        super().__init__()
        layers = [
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.Conv2d(channels, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.net(x)


class IlluminationNet(nn.Module):
    """Illumination correction network (Retinex-inspired)."""
    def __init__(self, channels: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 4, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels * 4, channels * 2, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        illum = self.decoder(self.encoder(x))
        # Retinex decomposition: reflectance = image / illumination
        corrected = x / (illum + 1e-4)
        return torch.clamp(corrected, 0, 1)


class VesselEnhancementNet(nn.Module):
    """Vessel enhancement using U-Net architecture."""
    def __init__(self, channels: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = self._conv_block(3, channels)
        self.enc2 = self._conv_block(channels, channels * 2)
        self.enc3 = self._conv_block(channels * 2, channels * 4)
        self.enc4 = self._conv_block(channels * 4, channels * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(channels * 8, channels * 16)
        
        # Decoder
        self.dec4 = self._upconv_block(channels * 16, channels * 8)
        self.dec3 = self._upconv_block(channels * 8, channels * 4)
        self.dec2 = self._upconv_block(channels * 4, channels * 2)
        self.dec1 = self._upconv_block(channels * 2, channels)
        
        self.final = nn.Conv2d(channels, 1, 1)
        
        self.pool = nn.MaxPool2d(2)
        
    def _conv_block(self, in_c: int, out_c: int):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_c: int, out_c: int):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 2, stride=2),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.dec4(b) + e4
        d3 = self.dec3(d4) + e3
        d2 = self.dec2(d3) + e2
        d1 = self.dec1(d2) + e1
        
        return torch.sigmoid(self.final(d1))


# ============================================================================
# Enhancement Pipeline
# ============================================================================

@dataclass
class EnhancementConfig:
    """Configuration for enhancement pipeline."""
    enable_sr: bool = True
    enable_denoising: bool = True
    enable_illumination: bool = True
    enable_vessel_enhancement: bool = True
    sr_scale: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir: str = "models/enhancement"


class AIEnhancer:
    """Main enhancement pipeline combining all AI models."""
    
    def __init__(self, config: Optional[EnhancementConfig] = None):
        self.config = config or EnhancementConfig()
        self.device = torch.device(self.config.device)
        self._models = {}
        self._load_models()
    
    def _load_models(self):
        """Load or initialize all enhancement models."""
        model_dir = Path(self.config.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Super-resolution
        if self.config.enable_sr:
            sr_path = model_dir / f"sr_x{self.config.sr_scale}.pt"
            if sr_path.exists():
                self._models['sr'] = SuperResolutionNet(scale=self.config.sr_scale).to(self.device)
                self._models['sr'].load_state_dict(torch.load(sr_path, map_location=self.device))
                self._models['sr'].eval()
        
        # Denoising
        if self.config.enable_denoising:
            denoise_path = model_dir / "denoise.pt"
            if denoise_path.exists():
                self._models['denoise'] = DenoisingNet().to(self.device)
                self._models['denoise'].load_state_dict(torch.load(denoise_path, map_location=self.device))
                self._models['denoise'].eval()
        
        # Illumination correction
        if self.config.enable_illumination:
            illum_path = model_dir / "illumination.pt"
            if illum_path.exists():
                self._models['illumination'] = IlluminationNet().to(self.device)
                self._models['illumination'].load_state_dict(torch.load(illum_path, map_location=self.device))
                self._models['illumination'].eval()
        
        # Vessel enhancement
        if self.config.enable_vessel_enhancement:
            vessel_path = model_dir / "vessel.pt"
            if vessel_path.exists():
                self._models['vessel'] = VesselEnhancementNet().to(self.device)
                self._models['vessel'].load_state_dict(torch.load(vessel_path, map_location=self.device))
                self._models['vessel'].eval()
    
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert numpy image to tensor."""
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy image."""
        img = tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        return np.clip(img * 255, 0, 255).astype(np.uint8)
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Apply all enabled enhancements to an image."""
        if image.dtype != np.float32:
            img = image.astype(np.float32) / 255.0
        else:
            img = image.copy()
        
        # Denoising first (clean input for other models)
        if 'denoise' in self._models:
            with torch.no_grad():
                tensor = self._to_tensor(img)
                img = self._to_numpy(self._models['denoise'](tensor)) / 255.0
        
        # Illumination correction
        if 'illumination' in self._models:
            with torch.no_grad():
                tensor = self._to_tensor(img)
                img = self._to_numpy(self._models['illumination'](tensor)) / 255.0
        
        # Super-resolution
        if 'sr' in self._models:
            with torch.no_grad():
                tensor = self._to_tensor(img)
                img = self._to_numpy(self._models['sr'](tensor)) / 255.0
        
        # Vessel enhancement (returns probability map, overlay on green channel)
        if 'vessel' in self._models:
            with torch.no_grad():
                tensor = self._to_tensor(img)
                vessel_map = self._models['vessel'](tensor).squeeze().cpu().numpy()
                # Enhance green channel with vessel map
                img_enhanced = img.copy()
                img_enhanced[:, :, 1] = np.clip(img_enhanced[:, :, 1] + 0.3 * vessel_map, 0, 1)
                img = img_enhanced
        
        return (img * 255).astype(np.uint8)
    
    def enhance_for_grading(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Enhance image and return metadata about applied enhancements."""
        metadata = {
            'sr_applied': self.config.enable_sr,
            'denoising_applied': self.config.enable_denoising,
            'illumination_applied': self.config.enable_illumination,
            'vessel_enhancement_applied': self.config.enable_vessel_enhancement,
        }
        enhanced = self.enhance(image)
        return enhanced, metadata


# ============================================================================
# Classical Enhancement Fallbacks (when AI models not available)
# ============================================================================

def classical_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """CLAHE enhancement on each channel."""
    if image.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(image)
    enhanced = np.zeros_like(image)
    for c in range(3):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced[:, :, c] = clahe.apply(image[:, :, c])
    return enhanced


def classical_retinex(image: np.ndarray, sigma_list: List[int] = [15, 80, 250]) -> np.ndarray:
    """Multi-scale Retinex with color restoration (MSRCR)."""
    img = image.astype(np.float32) / 255.0
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex += np.log10(img + 1e-4) - np.log10(blurred + 1e-4)
    retinex = retinex / len(sigma_list)
    # Color restoration
    alpha = 125.0
    intensity = img.sum(axis=2, keepdims=True) + 1e-4
    color_restoration = alpha * (np.log10(alpha * img + 1e-4) - np.log10(intensity))
    result = retinex + color_restoration
    # Normalize
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    return (result * 255).astype(np.uint8)


def classical_denoise(image: np.ndarray, h: float = 10, h_color: float = 10) -> np.ndarray:
    """Non-local means denoising."""
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, h, h_color, 7, 21)
    return cv2.fastNlMeansDenoising(image, None, h, 7, 21)


def enhance_fundus_classical(image: np.ndarray, 
                              clahe: bool = True,
                              retinex: bool = False,
                              denoise: bool = False) -> np.ndarray:
    """Classical enhancement pipeline as fallback."""
    img = image.copy()
    if denoise:
        img = classical_denoise(img)
    if clahe:
        img = classical_clahe(img)
    if retinex:
        img = classical_retinex(img)
    return img


# ============================================================================
# Convenience Functions
# ============================================================================

def create_enhancer(config: Optional[EnhancementConfig] = None) -> AIEnhancer:
    """Create an AI enhancer with default or custom config."""
    return AIEnhancer(config)


def enhance_fundus(image: np.ndarray, 
                    use_ai: bool = True,
                    config: Optional[EnhancementConfig] = None) -> np.ndarray:
    """High-level function to enhance a fundus image."""
    if use_ai:
        enhancer = create_enhancer(config)
        return enhancer.enhance(image)
    else:
        return enhance_fundus_classical(image)


if __name__ == "__main__":
    # Demo with synthetic image
    import argparse
    parser = argparse.ArgumentParser(description="Test AI enhancement")
    parser.add_argument("--input", type=str, help="Input image path")
    parser.add_argument("--output", type=str, default="enhanced_demo.png", help="Output path")
    parser.add_argument("--use-ai", action="store_true", help="Use AI models (requires trained weights)")
    args = parser.parse_args()
    
    if args.input:
        img = cv2.imread(args.input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Synthetic fundus-like image
        h, w = 512, 512
        y, x = np.ogrid[:h, :w]
        mask = ((x - w//2) ** 2 + (y - h//2) ** 2) < (min(h, w) * 0.4) ** 2
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[mask] = [100, 50, 30]
        img[~mask] = [10, 5, 3]
        # Add vessels
        for _ in range(20):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
            cv2.line(img, (x1, y1), (x2, y2), (80, 40, 20), np.random.randint(1, 3))
    
    print(f"Input shape: {img.shape}")
    enhanced = enhance_fundus(img, use_ai=args.use_ai)
    print(f"Enhanced shape: {enhanced.shape}")
    cv2.imwrite(args.output, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
    print(f"Saved to {args.output}")