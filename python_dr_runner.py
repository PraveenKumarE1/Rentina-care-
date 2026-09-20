import argparse
import csv
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pydicom

from backend.pipeline.ml_segmentation_adapter import segment_fundus as ml_segment_fundus
from backend.pipeline.robust_preprocessing import (
    assess_quality,
    classical_segmentation as robust_classical_segmentation,
    enhance_image,
    prepare_model_input,
)

MODEL_CHECKPOINT = os.environ.get(
    "RETINACARE_MODEL_CHECKPOINT",
    os.path.join(os.path.dirname(__file__), "models", "user_retinal_cnn.pt"),
)
DIABETES_MODEL_CHECKPOINT = os.environ.get(
    "RETINACARE_DIABETES_MODEL_CHECKPOINT",
    os.path.join(os.path.dirname(__file__), "models", "diabetes_retinal_risk_cnn.pt"),
)
MAX_PROCESSING_DIMENSION = int(os.environ.get("RETINACARE_MAX_DIMENSION", "1024"))
FAST_INFERENCE = os.environ.get("RETINACARE_FAST", "1").lower() not in {"0", "false", "no"}

try:
    import torch

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.set_num_interop_threads(1)
except Exception:
    torch = None

try:
    cv2.setNumThreads(max(1, min(4, os.cpu_count() or 1)))
except Exception:
    pass

_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.RLock()


@dataclass
class QualityResult:
    focus: float
    contrast: float
    vignetting_ratio: float
    field_of_view: float
    snr: float
    passed: bool
    feedback: List[str]
    enhancement_metadata: Dict[str, object] = field(default_factory=dict)


def _resize_for_processing(image: np.ndarray, max_dimension: int = 1400) -> np.ndarray:
    height, width = image.shape[:2]
    largest_dimension = max(height, width)
    if largest_dimension <= max_dimension:
        return image
    scale = max_dimension / largest_dimension
    return cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _normalize_array(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    if np.issubdtype(array.dtype, np.floating) and array.size and np.nanmax(array) <= 1.0:
        return np.clip(array, 0.0, 1.0)
    low, high = np.percentile(array, (1, 99)) if array.size else (0, 255)
    return np.clip((array - low) / (high - low + 1e-6), 0.0, 1.0)


def load_image(path: Optional[str] = None) -> np.ndarray:
    if path is not None:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".dcm":
            ds = pydicom.dcmread(path)
            pixel_array = ds.pixel_array
            if pixel_array.ndim == 2:
                img = _normalize_array(pixel_array)
                img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            else:
                img = _normalize_array(pixel_array)
                if img.ndim == 3 and img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
                img = (img * 255).astype(np.uint8)
            return _resize_for_processing(img[..., ::-1] if img.ndim == 3 else img)

        if os.path.isdir(path):
            files = []
            for root, _, filenames in os.walk(path):
                for name in filenames:
                    lower = name.lower()
                    if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                        files.append(os.path.join(root, name))
            if not files:
                raise FileNotFoundError(f"No image files found in folder: {path}")
            files.sort()
            image = cv2.imread(files[0], cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read first image: {files[0]}")
            return _resize_for_processing(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            if os.path.exists(path):
                raise ValueError(f"Could not decode image file: {path}")
            raise FileNotFoundError(f"Image not found: {path}")
        return _resize_for_processing(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    rng = np.random.default_rng(42)
    height, width = 256, 256
    y, x = np.ogrid[:height, :width]
    mask = ((x - width // 2) ** 2 + (y - height // 2) ** 2) < (100**2)
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[:, :, :] = 0.2
    img[mask] = 0.65
    img += 0.02 * rng.standard_normal((height, width, 3))
    return np.clip(img, 0, 1)


def quality_assessment(
    image: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
    use_ai: bool = True,
    model_dir: Optional[str] = None,
) -> Tuple[QualityResult, np.ndarray]:
    quality, rgb = assess_quality(image, thresholds)
    enhanced, metadata = enhance_image(
        rgb,
        use_ai=use_ai,
        model_dir=model_dir,
    )
    quality.enhancement_metadata = metadata
    return quality, enhanced


def segmentation(image: np.ndarray, quality: QualityResult, use_ml: bool = True) -> Dict[str, object]:
    return ml_segment_fundus(
        image,
        quality_passed=quality.passed,
        use_ml=use_ml,
    )


def grading(segmentation_result: Dict[str, object], quality: QualityResult) -> Dict[str, object]:
    if not quality.passed:
        return {
            "level": None,
            "label": "Ungradable",
            "confidence": 0.0,
            "referable": False,
            "probabilities": [0.0] * 5,
            "evidence": {},
            "method": "quality gate: no severity prediction for an ungradable image",
            "uncertainty": 1.0,
            "explanation": "Image quality is insufficient for reliable screening. Please retake the image with a proper retinal camera.",
            "lesion_counts": {"microaneurysms": 0, "exudates": 0, "haemorrhages": 0, "neovascularisation": 0},
            "severity_indicators": [],
        }

    ma = int(segmentation_result["microaneurysms"]["count"])
    ex = int(segmentation_result["exudates"]["count"])
    he = int(segmentation_result["haemorrhages"]["count"])
    nv = int(segmentation_result["neovascularisation"]["count"])

    if nv > 0:
        level = 4
    elif he >= 20:
        level = 3
    elif he > 0 or ex > 0:
        level = 2
    elif ma > 0:
        level = 1
    else:
        level = 0

    labels = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]

    total_lesions = ma + ex + he + nv
    confidence = _calculate_nuanced_confidence(level, ma, ex, he, nv, quality)
    probabilities = _generate_probabilities(level, confidence)

    severity_indicators = _build_severity_indicators(level, ma, ex, he, nv)
    explanation = _build_explanation(level, ma, ex, he, nv, confidence, quality)

    return {
        "level": level,
        "label": labels[level],
        "confidence": confidence,
        "referable": level >= 2,
        "probabilities": probabilities,
        "evidence": {
            "microaneurysms": ma,
            "exudates": ex,
            "haemorrhages": he,
            "neovascularisation": nv,
        },
        "method": "interpretable classical lesion proposals",
        "uncertainty": 1.0 - confidence,
        "explanation": explanation,
        "lesion_counts": {"microaneurysms": ma, "exudates": ex, "haemorrhages": he, "neovascularisation": nv},
        "severity_indicators": severity_indicators,
        "total_lesions": total_lesions,
    }


def _calculate_nuanced_confidence(level: int, ma: int, ex: int, he: int, nv: int, quality: QualityResult) -> float:
    base_confidence = {
        0: 0.75,
        1: 0.60,
        2: 0.65,
        3: 0.70,
        4: 0.75,
    }[level]

    lesion_boost = min(0.15, (ma + ex + he + nv) * 0.003)
    quality_factor = 1.0
    if quality.focus < 50:
        quality_factor *= 0.9
    if quality.contrast < 0.15:
        quality_factor *= 0.85
    if quality.snr < 3.0:
        quality_factor *= 0.9

    confidence = base_confidence * quality_factor + lesion_boost
    return float(np.clip(confidence, 0.35, 0.95))


def _generate_probabilities(level: int, confidence: float) -> List[float]:
    probs = [0.02, 0.02, 0.02, 0.02, 0.02]
    remaining = 1.0 - confidence
    other_levels = [i for i in range(5) if i != level]
    for i, other in enumerate(other_levels):
        weight = 1.0 / (abs(other - level) + 1)
        probs[other] = remaining * weight / sum(1.0 / (abs(o - level) + 1) for o in other_levels)
    probs[level] = confidence
    total = sum(probs)
    return [p / total for p in probs]


def _build_severity_indicators(level: int, ma: int, ex: int, he: int, nv: int) -> List[str]:
    indicators = []
    if nv > 0:
        indicators.append(f"Neovascularization detected ({nv} regions) - indicates PDR")
    if he >= 20:
        indicators.append(f"Numerous hemorrhages ({he}) - severe NPDR feature")
    elif he > 0:
        indicators.append(f"Hemorrhages present ({he})")
    if ex > 0:
        indicators.append(f"Exudates present ({ex})")
    if ma > 0:
        indicators.append(f"Microaneurysms present ({ma})")
    if level == 0:
        indicators.append("No diabetic retinal lesions detected")
    return indicators


def _build_explanation(level: int, ma: int, ex: int, he: int, nv: int, confidence: float, quality: QualityResult) -> str:
    labels = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]
    label = labels[level]

    explanations = {
        0: (
            f"No signs of diabetic retinopathy were detected in this retinal image. "
            f"The screening found {ma} microaneurysms, {ex} exudates, and {he} hemorrhages. "
            f"Continue routine diabetes care and annual eye screenings."
        ),
        1: (
            f"Early signs of diabetic retinopathy (Mild NPDR) were detected. "
            f"The screening found {ma} microaneurysms, which are tiny bulges in retinal blood vessels. "
            f"These are the earliest visible changes. Schedule a comprehensive eye exam and discuss diabetes management with your doctor."
        ),
        2: (
            f"Moderate diabetic retinopathy was detected. "
            f"The screening found {he} hemorrhages and/or {ex} exudates (protein/lipid deposits). "
            f"These indicate more significant blood vessel damage. Prompt ophthalmology review is recommended."
        ),
        3: (
            f"Severe diabetic retinopathy was detected. "
            f"The screening found {he} hemorrhages, which exceeds the threshold for severe NPDR (20+ hemorrhages). "
            f"This indicates widespread retinal blood vessel damage. Urgent ophthalmology referral is needed."
        ),
        4: (
            f"Proliferative diabetic retinopathy (PDR) was detected. "
            f"The screening found {nv} areas of neovascularization (abnormal new blood vessel growth). "
            f"This is an advanced stage requiring immediate specialist care to prevent vision loss."
        ),
    }

    base = explanations[level]
    quality_note = ""
    if not quality.passed:
        quality_note = " Note: Image quality issues were detected, which may affect accuracy."
    elif quality.focus < 50:
        quality_note = " Note: Image focus could be improved for more reliable results."
    confidence_note = f" Confidence: {confidence:.0%}."

    return base + quality_note + confidence_note


def _load_torch_model(checkpoint: Dict[str, object], device):
    from train_retinal_model import ResidualRetinalCNN, SmallRetinalCNN

    architecture = str(checkpoint.get("architecture", "small"))
    classes = int(checkpoint.get("classes", 5))
    if architecture == "residual":
        model = ResidualRetinalCNN(classes=classes)
    elif architecture == "small":
        model = SmallRetinalCNN(classes=classes)
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def _prepare_checkpoint_input(image: np.ndarray, image_size: int, normalization: str) -> np.ndarray:
    if normalization == "centered":
        return prepare_model_input(
            image,
            image_size=image_size,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
    return prepare_model_input(image, image_size=image_size)


def _predict_checkpoint(
    image: np.ndarray,
    checkpoint_path: str,
    device,
    use_enhanced_image: bool = False,
    normalization: str = "centered",
):
    import torch
    from train_retinal_model import tta_predict

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    image_size = int(checkpoint.get("image_size", 224))
    classes = int(checkpoint.get("classes", 5))
    model = _load_torch_model(checkpoint, device)
    tensor = torch.from_numpy(
        _prepare_checkpoint_input(image, image_size, normalization)
    ).unsqueeze(0)
    temperature = float(checkpoint.get("temperature", 1.0))
    probabilities = tta_predict(model, tensor, device, temperature=temperature)
    return probabilities, checkpoint


def trained_model_grading(
    path: Optional[str] = None,
    image: Optional[np.ndarray] = None,
    enhanced_image: Optional[np.ndarray] = None,
) -> Optional[Dict[str, object]]:
    if not os.path.exists(MODEL_CHECKPOINT):
        return None
    try:
        import torch

        source = image if image is not None else load_image(path)
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location="cpu", weights_only=False)
        use_enhanced = checkpoint.get("enhancement") == "robust_classical"
        source_for_model = enhanced_image if use_enhanced and enhanced_image is not None else source
        probabilities, checkpoint = _predict_checkpoint(
            source_for_model,
            MODEL_CHECKPOINT,
            torch.device("cpu"),
            use_enhanced_image=use_enhanced,
            normalization=str(checkpoint.get("normalization", "centered")),
        )
        level = int(np.argmax(probabilities))
        labels = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]
        confidence = float(probabilities[level])
        entropy = float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-8, 1.0))))
        max_entropy = np.log(len(probabilities))
        return {
            "level": level,
            "label": labels[level],
            "confidence": confidence,
            "referable": level >= 2,
            "probabilities": probabilities.tolist(),
            "evidence": {},
            "method": f"trained PyTorch {checkpoint.get('architecture', 'CNN')} checkpoint",
            "uncertainty": float(np.clip(entropy / max_entropy, 0.0, 1.0)),
        }
    except Exception:
        return None


def trained_diabetes_risk(
    path: Optional[str] = None,
    image: Optional[np.ndarray] = None,
    enhanced_image: Optional[np.ndarray] = None,
    quality: Optional[QualityResult] = None,
) -> Dict[str, object]:
    unavailable = {
        "available": False,
        "risk": None,
        "confidence": None,
        "message": "Diabetes risk prediction is unavailable: train a separate model using retinal images with verified binary diabetes labels.",
    }
    if quality is not None and not quality.passed:
        unavailable["message"] = "Diabetes risk prediction is unavailable because the retinal image did not pass quality checks."
        return unavailable
    if not os.path.exists(DIABETES_MODEL_CHECKPOINT):
        return unavailable
    try:
        import torch

        source = image if image is not None else load_image(path)
        checkpoint = torch.load(DIABETES_MODEL_CHECKPOINT, map_location="cpu", weights_only=False)
        if checkpoint.get("task") != "diabetes_risk" or int(checkpoint.get("classes", 0)) != 2:
            unavailable["message"] = "Diabetes risk checkpoint is invalid; it must be trained with --task diabetes_risk and verified 0/1 labels."
            return unavailable
        use_enhanced = checkpoint.get("enhancement") == "robust_classical"
        source_for_model = enhanced_image if use_enhanced and enhanced_image is not None else source
        probabilities, _ = _predict_checkpoint(
            source_for_model,
            DIABETES_MODEL_CHECKPOINT,
            torch.device("cpu"),
            use_enhanced_image=use_enhanced,
            normalization=str(checkpoint.get("normalization", "centered")),
        )
        probability = float(probabilities[1])
        return {
            "available": True,
            "risk": "Higher retinal-image diabetes risk" if probability >= 0.5 else "Lower retinal-image diabetes risk",
            "confidence": float(probabilities.max()),
            "probability": probability,
            "message": "Research screening estimate only. Confirm diabetes with A1C or plasma-glucose testing.",
        }
    except Exception:
        unavailable["message"] = "Diabetes risk prediction could not be loaded. Check the separately trained checkpoint."
        return unavailable


def run_pipeline(
    image: Optional[np.ndarray] = None,
    path: Optional[str] = None,
    use_ml: bool = True,
    use_ai_enhancement: bool = True,
) -> Dict[str, object]:
    img = image if image is not None else load_image(path)
    quality, enhanced = quality_assessment(
        img,
        use_ai=use_ai_enhancement,
    )
    segmentation_result = segmentation(enhanced, quality, use_ml=use_ml)
    grading_result = None
    if quality.passed and use_ml:
        grading_result = trained_model_grading(
            path=path,
            image=img,
            enhanced_image=enhanced,
        )
    if grading_result is None:
        grading_result = grading(segmentation_result, quality)
    diabetes_risk = trained_diabetes_risk(
        path=path,
        image=img,
        enhanced_image=enhanced,
        quality=quality,
    ) if quality.passed and use_ml else {
        "available": False,
        "risk": None,
        "confidence": None,
        "message": "Diabetes risk prediction requires a trained verified-label model and a gradable image.",
    }
    summary = f"{('Gradable' if quality.passed else 'Ungradable')}: {grading_result['label']} (confidence {grading_result['confidence']:.2f})"
    return {
        "quality": quality,
        "segmentation": segmentation_result,
        "grading": grading_result,
        "image": enhanced,
        "summary": summary,
        "model_method": grading_result.get("method", "quality gate"),
        "diabetes_risk": diabetes_risk,
        "enhancement": quality.enhancement_metadata,
        "segmentation_method": segmentation_result.get("method", "classical adaptive segmentation"),
    }


def capture_camera_photo(save_path: Optional[str] = None, camera_index: int = 0) -> str:
    if save_path is None:
        save_dir = os.path.join(os.getcwd(), "captures")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "camera_capture.jpg")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}. Check that the camera is available.")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("The camera did not return a frame.")
    if not cv2.imwrite(save_path, frame):
        raise RuntimeError(f"Failed to save captured image to: {save_path}")
    return save_path


def list_supported_files(folder: str) -> List[str]:
    files: List[str] = []
    for root, _, filenames in os.walk(folder):
        for name in filenames:
            lower = name.lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm")):
                files.append(os.path.join(root, name))
    return sorted(files)


def extract_patient_id(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0]
    candidates = [name, name.replace("_", " "), name.replace("-", " ")]
    for candidate in candidates:
        if any(character.isdigit() for character in candidate):
            return candidate
    return name


def build_batch_summary(results: List[Dict[str, object]]) -> Dict[str, object]:
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "quality_pass_count": 0,
            "quality_pass_rate": 0.0,
            "referable_count": 0,
            "referable_rate": 0.0,
            "average_confidence": 0.0,
            "grade_counts": {},
            "model_methods": {},
            "errors": 0,
        }

    quality_pass_count = 0
    referable_count = 0
    confidence_values: List[float] = []
    grade_counts: Dict[str, int] = {}
    model_methods: Dict[str, int] = {}
    errors = 0

    for result in results:
        quality = result.get("quality") or {}
        grading = result.get("grading") or {}
        if isinstance(quality, dict):
            passed = bool(quality.get("passed", False))
        else:
            passed = bool(getattr(quality, "passed", False))
        if passed:
            quality_pass_count += 1

        label = grading.get("label") if isinstance(grading, dict) else getattr(grading, "label", "Ungradable")
        if label is None:
            label = "Ungradable"
        grade_counts[label] = grade_counts.get(label, 0) + 1

        referable = grading.get("referable") if isinstance(grading, dict) else getattr(grading, "referable", False)
        if bool(referable):
            referable_count += 1

        confidence = grading.get("confidence") if isinstance(grading, dict) else getattr(grading, "confidence", 0.0)
        try:
            conf_value = float(confidence)
            if passed and label not in (None, "Ungradable") and conf_value > 0.0:
                confidence_values.append(conf_value)
        except (TypeError, ValueError):
            pass

        model_method = result.get("model_method") or result.get("segmentation_method") or "unknown"
        model_methods[str(model_method)] = model_methods.get(str(model_method), 0) + 1

        if result.get("error"):
            errors += 1

    return {
        "total": total,
        "quality_pass_count": quality_pass_count,
        "quality_pass_rate": quality_pass_count / total,
        "referable_count": referable_count,
        "referable_rate": referable_count / total,
        "average_confidence": sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
        "grade_counts": grade_counts,
        "model_methods": model_methods,
        "errors": errors,
    }


def evaluate_dataset(folder: str, use_ml: bool = True, use_ai_enhancement: bool = True) -> Dict[str, object]:
    files = list_supported_files(folder)
    if not files:
        raise FileNotFoundError(f"No supported image files found in folder: {folder}")

    results: List[Dict[str, object]] = []
    for path in files:
        try:
            result = run_pipeline(
                image=load_image(path),
                use_ml=use_ml,
                use_ai_enhancement=use_ai_enhancement,
            )
            result["filename"] = os.path.basename(path)
            results.append(result)
        except Exception as exc:
            results.append({
                "filename": os.path.basename(path),
                "error": str(exc),
                "quality": {"passed": False},
                "grading": {"label": "Ungradable", "confidence": 0.0, "referable": False},
                "segmentation": {"microaneurysms": {"count": 0}, "exudates": {"count": 0}, "haemorrhages": {"count": 0}},
            })
    return {
        "results": results,
        "summary": build_batch_summary(results),
        "output_dir": os.path.join(folder, "batch_reports"),
    }


def export_batch_report(results: List[Dict[str, object]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "batch_results.csv")
    json_path = os.path.join(output_dir, "batch_results.json")
    fieldnames = [
        "filename", "patient_id", "quality_passed", "summary", "grade_level", "grade_label",
        "referable", "confidence", "uncertainty", "focus", "contrast", "vignetting_ratio",
        "field_of_view", "snr", "microaneurysms", "exudates", "haemorrhages", "model_method",
    ]
    rows: List[Dict[str, object]] = []
    for result in results:
        quality = result.get("quality") or {}
        grading = result.get("grading") or {}
        segmentation = result.get("segmentation") or {}
        rows.append({
            "filename": result.get("filename", ""),
            "patient_id": extract_patient_id(str(result.get("filename", ""))),
            "quality_passed": bool(quality.get("passed", False)) if isinstance(quality, dict) else bool(getattr(quality, "passed", False)),
            "summary": result.get("summary", ""),
            "grade_level": grading.get("level") if isinstance(grading, dict) else getattr(grading, "level", None),
            "grade_label": grading.get("label") if isinstance(grading, dict) else getattr(grading, "label", "Ungradable"),
            "referable": grading.get("referable") if isinstance(grading, dict) else getattr(grading, "referable", False),
            "confidence": grading.get("confidence") if isinstance(grading, dict) else getattr(grading, "confidence", 0.0),
            "uncertainty": grading.get("uncertainty") if isinstance(grading, dict) else getattr(grading, "uncertainty", 0.0),
            "focus": quality.get("focus") if isinstance(quality, dict) else getattr(quality, "focus", 0.0),
            "contrast": quality.get("contrast") if isinstance(quality, dict) else getattr(quality, "contrast", 0.0),
            "vignetting_ratio": quality.get("vignetting_ratio") if isinstance(quality, dict) else getattr(quality, "vignetting_ratio", 0.0),
            "field_of_view": quality.get("field_of_view") if isinstance(quality, dict) else getattr(quality, "field_of_view", 0.0),
            "snr": quality.get("snr") if isinstance(quality, dict) else getattr(quality, "snr", 0.0),
            "microaneurysms": (segmentation.get("microaneurysms", {}) or {}).get("count", 0),
            "exudates": (segmentation.get("exudates", {}) or {}).get("count", 0),
            "haemorrhages": (segmentation.get("haemorrhages", {}) or {}).get("count", 0),
            "model_method": result.get("model_method", ""),
        })
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump({"summary": build_batch_summary(results), "rows": rows}, file, indent=2, default=str)
    print(f"Batch report saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DR Screening System Python runner")
    parser.add_argument("input", nargs="?", default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--save-photo", default=None)
    parser.add_argument("--no-ml", action="store_true")
    parser.add_argument("--no-ai-enhancement", action="store_true")
    args = parser.parse_args()

    print("DR Screening System - robust Python pipeline")
    if args.camera:
        try:
            save_path = capture_camera_photo(args.save_photo)
            result = run_pipeline(
                path=save_path,
                use_ml=not args.no_ml,
                use_ai_enhancement=not args.no_ai_enhancement,
            )
        except Exception as exc:
            raise RuntimeError(f"Camera capture failed: {exc}") from exc
    elif args.batch and args.input and os.path.isdir(args.input):
        batch = evaluate_dataset(
            args.input,
            use_ml=not args.no_ml,
            use_ai_enhancement=not args.no_ai_enhancement,
        )
        results = batch["results"]
        summary = batch["summary"]
        for result in results:
            filename = result.get("filename", "unknown")
            if result.get("error"):
                print(f"[{filename}] ERROR: {result['error']}")
            else:
                print(f"[{filename}] {result['summary']}")
        print("\nBatch summary:")
        print(json.dumps({
            "total": summary["total"],
            "quality_pass_rate": round(summary["quality_pass_rate"], 4),
            "referable_rate": round(summary["referable_rate"], 4),
            "average_confidence": round(summary["average_confidence"], 4),
            "grade_counts": summary["grade_counts"],
            "model_methods": summary["model_methods"],
            "errors": summary["errors"],
        }, indent=2))
        export_batch_report(results, batch["output_dir"])
        return
    else:
        if args.demo or args.input is None:
            result = run_pipeline(
                image=load_image(),
                use_ml=not args.no_ml,
                use_ai_enhancement=not args.no_ai_enhancement,
            )
        else:
            result = run_pipeline(
                path=args.input,
                use_ml=not args.no_ml,
                use_ai_enhancement=not args.no_ai_enhancement,
            )

    from web_ui.server import build_case_review
    print("\n" + build_case_review(result))


if __name__ == "__main__":
    main()
