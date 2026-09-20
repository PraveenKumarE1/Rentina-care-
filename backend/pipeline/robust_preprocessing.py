"""Robust retinal image quality assessment, enhancement, and model preprocessing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class QualityResult:
    focus: float
    contrast: float
    vignetting_ratio: float
    field_of_view: float
    snr: float
    passed: bool
    feedback: List[str]


def _to_rgb_float(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    elif array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected a grayscale or RGB image, got shape {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        if array.size and np.nanmax(array) > 1.0:
            array = array / 255.0
    else:
        info = np.iinfo(array.dtype)
        array = array.astype(np.float32) / float(info.max)
    return np.clip(np.nan_to_num(array, nan=0.0), 0.0, 1.0).astype(np.float32)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def _largest_component(mask: np.ndarray, minimum_area: int = 1) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    result = np.zeros_like(mask, dtype=np.uint8)
    if count <= 1:
        return result
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0 or int(areas.max()) < minimum_area:
        return result
    label = 1 + int(np.argmax(areas))
    result[labels == label] = 1
    return result


def estimate_field_of_view(gray: np.ndarray) -> Tuple[float, np.ndarray]:
    height, width = gray.shape
    threshold = max(0.015, float(np.percentile(gray, 2)) + 0.01)
    mask = (gray > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = _largest_component(mask, max(10, int(gray.size * 0.001)))
    if mask.sum() == 0:
        return 0.0, mask.astype(bool)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        ((center_x, center_y), radius) = cv2.minEnclosingCircle(contour.astype(np.float32))
        diameter = 2.0 * radius / min(height, width)
    else:
        diameter = float(np.sqrt(mask.mean()))

    return float(np.clip(diameter, 0.0, 1.5)), mask.astype(bool)


def estimate_vignetting(gray: np.ndarray, fov_mask: np.ndarray) -> float:
    height, width = gray.shape
    center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
    y, x = np.ogrid[:height, :width]
    radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_radius = max(1.0, float(radius.max()))
    normalized_radius = radius / max_radius
    inner = (normalized_radius < 0.45) & fov_mask
    outer = (normalized_radius > 0.80) & fov_mask
    inner_mean = float(gray[inner].mean()) if inner.any() else float(gray.mean())
    outer_mean = float(gray[outer].mean()) if outer.any() else float(gray.mean())
    return float(inner_mean / (outer_mean + 1e-6))


def assess_quality(
    image: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[QualityResult, np.ndarray]:
    rgb = _to_rgb_float(image)
    gray = rgb.mean(axis=2)
    gray8 = _to_uint8(gray)
    defaults = {
        "minFocus": 25.0,
        "minContrast": 0.08,
        "minFovDiameter": 0.45,
        "minSnr": 1.8,
    }
    if thresholds:
        defaults.update({key: float(value) for key, value in thresholds.items()})

    laplacian = cv2.Laplacian(gray8, cv2.CV_32F)
    focus = float(np.var(laplacian))
    contrast = float(np.percentile(gray, 99) - np.percentile(gray, 1))
    field_of_view, fov_mask = estimate_field_of_view(gray)
    vignetting = estimate_vignetting(gray, fov_mask)
    snr = float(gray.mean() / (np.std(gray) + 1e-6))

    passed = (
        focus >= defaults["minFocus"]
        and contrast >= defaults["minContrast"]
        and field_of_view >= defaults["minFovDiameter"]
        and snr >= defaults["minSnr"]
    )
    feedback: List[str] = []
    if focus < defaults["minFocus"]:
        feedback.append("Poor focus - retake")
    if contrast < defaults["minContrast"]:
        feedback.append("Low contrast - retake")
    if field_of_view < defaults["minFovDiameter"]:
        feedback.append("Insufficient field of view - retake")
    if vignetting < 0.65 or vignetting > 1.8:
        feedback.append("Uneven illumination")
    if snr < defaults["minSnr"]:
        feedback.append("Low signal-to-noise ratio - review illumination")
    if not feedback:
        feedback.append("Pass")

    quality = QualityResult(
        focus=focus,
        contrast=contrast,
        vignetting_ratio=vignetting,
        field_of_view=field_of_view,
        snr=snr,
        passed=passed,
        feedback=feedback,
    )
    return quality, rgb


def _illumination_correct(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.mean(axis=2)
    sigma = max(12.0, min(rgb.shape[:2]) / 14.0)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    ratio = (gray + 0.04) / (background + 0.04)
    ratio = np.clip((ratio - float(np.percentile(ratio, 1))) / (float(np.percentile(ratio, 99)) - float(np.percentile(ratio, 1)) + 1e-6), 0.15, 2.5)
    corrected = rgb * ratio[:, :, None]
    return np.clip(corrected, 0.0, 1.0)


def _color_constancy(rgb: np.ndarray) -> np.ndarray:
    channel_mean = rgb.mean(axis=(0, 1), keepdims=True)
    global_mean = float(channel_mean.mean())
    gains = global_mean / (channel_mean + 1e-6)
    return np.clip(rgb * gains, 0.0, 1.0)


def _clahe(rgb: np.ndarray, clip_limit: float = 2.2, tile_size: int = 8) -> np.ndarray:
    image = _to_uint8(rgb)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    channel = image[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    image[:, :, 0] = clahe.apply(channel)
    return cv2.cvtColor(image, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def _denoise_fast(rgb: np.ndarray) -> np.ndarray:
    image = _to_uint8(rgb)
    denoised = cv2.GaussianBlur(image, (3, 3), sigmaX=0.8, sigmaY=0.8)
    return cv2.addWeighted(image, 0.82, denoised, 0.18, 0).astype(np.float32) / 255.0


def _sharpen_fast(rgb: np.ndarray, amount: float = 0.12) -> np.ndarray:
    image = _to_uint8(rgb)
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=0.7, sigmaY=0.7)
    return cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0).astype(np.float32) / 255.0


def _denoise(rgb: np.ndarray) -> np.ndarray:
    """Compatibility wrapper for the enhancement pipeline."""
    return _denoise_fast(rgb)


def _sharpen(rgb: np.ndarray, amount: float = 0.12) -> np.ndarray:
    """Compatibility wrapper for the enhancement pipeline."""
    return _sharpen_fast(rgb, amount=amount)


def enhance_image(
    image: np.ndarray,
    use_ai: bool = False,
    model_dir: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    rgb = _to_rgb_float(image)
    original_shape = rgb.shape
    metadata: Dict[str, object] = {
        "method": "classical_retinex_clahe",
        "ai_applied": False,
        "original_shape": list(original_shape),
        "color_constancy": True,
        "illumination_correction": True,
        "denoising": True,
        "contrast_enhancement": True,
        "sharpening": True,
    }

    if use_ai:
        directory = Path(model_dir) if model_dir else Path(__file__).resolve().parents[2] / "models" / "enhancement"
        required = [directory / "denoise.pt", directory / "illumination.pt"]
        if any(directory.glob("*.pt")) and all(path.exists() for path in required):
            try:
                from .ai_enhancement import AIEnhancer, EnhancementConfig
                config = EnhancementConfig(model_dir=str(directory))
                enhancer = AIEnhancer(config)
                rgb = enhancer.enhance(rgb).astype(np.float32) / 255.0
                metadata.update({"method": "ai_enhancement", "ai_applied": True})
            except Exception as exc:
                metadata["ai_fallback_reason"] = f"No compatible trained enhancement checkpoint: {exc}"

    rgb = _denoise(rgb)
    rgb = _illumination_correct(rgb)
    rgb = _color_constancy(rgb)
    rgb = _clahe(rgb)
    rgb = _sharpen(rgb)
    metadata["output_shape"] = list(rgb.shape)
    return _to_uint8(rgb), metadata


def prepare_model_input(
    image: np.ndarray,
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    rgb = _to_rgb_float(image)
    height, width = rgb.shape[:2]
    scale = min(float(image_size) / height, float(image_size) / width)
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((image_size, image_size, 3), dtype=np.float32)
    y_offset = (image_size - resized_height) // 2
    x_offset = (image_size - resized_width) // 2
    canvas[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    normalized = (canvas - mean_array) / (std_array + 1e-6)
    return normalized.transpose(2, 0, 1).astype(np.float32)


def classical_segmentation(image: np.ndarray, quality_passed: bool = True) -> Dict[str, object]:
    rgb = _to_rgb_float(image)
    green = rgb[:, :, 1]
    bright = rgb.mean(axis=2)
    height, width = green.shape

    disc_candidate = (bright > np.percentile(bright, 99.2)).astype(np.uint8)
    disc_candidate = cv2.morphologyEx(
        disc_candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )
    optic_disc = _largest_component(disc_candidate, max(20, int(green.size * 0.001)))

    exudate_candidate = (bright > np.percentile(bright, 98.5)).astype(np.uint8)
    exudate_candidate = cv2.morphologyEx(
        exudate_candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    exudates = ((exudate_candidate > 0) & (optic_disc == 0)).astype(np.uint8)

    dark = 1.0 - green
    hemorrhage_candidate = (dark > np.percentile(dark, 97.2)).astype(np.uint8)
    hemorrhage_candidate = cv2.morphologyEx(
        hemorrhage_candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    hemorrhages = (hemorrhage_candidate > 0).astype(np.uint8)

    microaneurysm_candidate = (dark > np.percentile(dark, 99.0)).astype(np.uint8)
    microaneurysm_candidate = cv2.morphologyEx(
        microaneurysm_candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    microaneurysms = ((microaneurysm_candidate > 0) & (optic_disc == 0)).astype(np.uint8)

    inverse_green = _to_uint8(1.0 - green)
    vessels = cv2.adaptiveThreshold(
        inverse_green,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )
    vessels = cv2.morphologyEx(
        vessels, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    vessels = ((vessels > 0).astype(np.uint8) & (optic_disc == 0))

    fovea = np.zeros((height, width), dtype=np.uint8)
    disc_pixels = np.where(optic_disc)
    if disc_pixels[0].size:
        disc_y = float(disc_pixels[0].mean())
        disc_x = float(disc_pixels[1].mean())
        left_mean = float(green[:, : max(1, int(disc_x))].mean()) if disc_x > width * 0.15 else 1.0
        right_mean = float(green[:, int(disc_x):].mean()) if disc_x < width * 0.85 else 1.0
        temporal_x = 0 if left_mean < right_mean else width - 1
        radius = max(12, int(min(height, width) * 0.045))
        center_x = max(radius, min(width - radius - 1, int(temporal_x)))
        center_y = max(radius, min(height - radius - 1, int(disc_y)))
        cv2.circle(fovea, (center_x, center_y), radius, 1, -1)

    def properties(mask: np.ndarray) -> Dict[str, object]:
        mask = mask.astype(bool)
        return {"mask": mask, "count": int(mask.sum()), "area": float(mask.mean())}

    result = {
        "opticDisc": properties(optic_disc),
        "fovea": properties(fovea),
        "vessels": properties(vessels),
        "microaneurysms": properties(microaneurysms),
        "exudates": properties(exudates),
        "haemorrhages": properties(hemorrhages),
        "neovascularisation": properties(np.zeros((height, width), dtype=bool)),
    }
    vessel_length = float(vessels.sum())
    vessel_span = float(np.sqrt((width - 1) ** 2 + (height - 1) ** 2))
    result["features"] = {
        "vesselDensity": float(vessels.mean()),
        "tortuosity": vessel_length / (vessel_span + 1e-6),
        "cupToDiscRatio": float("nan"),
    }
    if not quality_passed:
        result["warning"] = "Interpret results only after a gradable acquisition."
    return result
