import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pydicom

MODEL_CHECKPOINT = os.path.join(os.path.dirname(__file__), "models", "user_retinal_cnn.pt")
DIABETES_MODEL_CHECKPOINT = os.path.join(os.path.dirname(__file__), "models", "diabetes_retinal_risk_cnn.pt")


@dataclass
class QualityResult:
    focus: float
    contrast: float
    vignetting_ratio: float
    field_of_view: float
    snr: float
    passed: bool
    feedback: List[str]


def _resize_for_processing(image: np.ndarray, max_dimension: int = 1400) -> np.ndarray:
    height, width = image.shape[:2]
    largest_dimension = max(height, width)
    if largest_dimension <= max_dimension:
        return image
    scale = max_dimension / largest_dimension
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def load_image(path: Optional[str] = None) -> np.ndarray:
    if path is not None:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".dcm":
            ds = pydicom.dcmread(path)
            pixel_array = ds.pixel_array
            if pixel_array.ndim == 2:
                img = cv2.cvtColor(pixel_array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            else:
                img = pixel_array.astype(np.uint8)
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
    h, w = 256, 256
    y, x = np.ogrid[:h, :w]
    mask = ((x - 128) ** 2 + (y - 128) ** 2) < (100 ** 2)
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, :, :] = 0.2
    img[mask] = 0.65
    img += 0.02 * rng.standard_normal((h, w, 3))
    img = np.clip(img, 0, 1)
    return img


def quality_assessment(image: np.ndarray, thresholds: Optional[Dict[str, float]] = None) -> Tuple[QualityResult, np.ndarray]:
    thresholds = thresholds or {"minFocus": 25.0, "minContrast": 0.08, "minFovDiameter": 0.45, "minSnr": 1.8}
    gray = image.astype(np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    gray_for_focus = gray * 255.0 if gray.max() <= 1.0 else gray
    gray = gray / 255.0 if gray.max() > 1.0 else gray

    lap = cv2.Laplacian(gray_for_focus, cv2.CV_32F)
    focus = float(np.var(lap))
    contrast = float(gray.std())

    margin = max(1, int(min(gray.shape) * 0.05))
    top = gray[:margin, :].ravel()
    bottom = gray[-margin:, :].ravel()
    left = gray[:, :margin].ravel()
    right = gray[:, -margin:].ravel()
    border = np.concatenate([top, bottom, left, right])
    center = gray[gray.shape[0] // 4 : 3 * gray.shape[0] // 4, gray.shape[1] // 4 : 3 * gray.shape[1] // 4]
    vignetting = float(center.mean() / (border.mean() + 1e-8))

    fov_mask = gray > 0.05
    fov_diameter = float(max(gray.shape) * np.sqrt(np.mean(fov_mask)) / max(gray.shape))
    snr = float(gray.mean() / (gray.std() + 1e-8))

    passed = (
        focus >= thresholds["minFocus"]
        and contrast >= thresholds["minContrast"]
        and fov_diameter >= thresholds["minFovDiameter"]
        and snr >= thresholds["minSnr"]
    )

    feedback: List[str] = []
    if focus < thresholds["minFocus"]:
        feedback.append("Poor focus - retake")
    if contrast < thresholds["minContrast"]:
        feedback.append("Low contrast - retake")
    if fov_diameter < thresholds["minFovDiameter"]:
        feedback.append("Insufficient field of view - retake")
    if snr < thresholds["minSnr"]:
        feedback.append("Low signal-to-noise ratio - review illumination")
    if vignetting < 0.65 or vignetting > 1.8:
        feedback.append("Uneven illumination")
    if not feedback:
        feedback = ["Pass"]

    quality = QualityResult(
        focus=focus,
        contrast=contrast,
        vignetting_ratio=vignetting,
        field_of_view=fov_diameter,
        snr=snr,
        passed=passed,
        feedback=feedback,
    )

    if not passed and contrast >= thresholds["minContrast"] * 0.7:
        enhanced = np.zeros_like(image)
        for channel in range(image.shape[2]):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced[:, :, channel] = clahe.apply((image[:, :, channel] * 255).astype(np.uint8)) / 255.0
        return quality, enhanced

    return quality, image


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(np.uint8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    result = np.zeros_like(mask, dtype=np.uint8)
    result[labels == largest] = 1
    return result


def segmentation(image: np.ndarray, quality: QualityResult) -> Dict[str, object]:
    rgb = image.astype(np.float32)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    green = rgb[:, :, 1]
    bright = rgb.sum(axis=2) / 3.0

    optic_disc = bright > float(np.percentile(bright, 99))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    optic_disc = cv2.morphologyEx(optic_disc.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    optic_disc = _largest_component(optic_disc)

    exudates = bright > float(np.percentile(bright, 98))
    exudates = cv2.morphologyEx(exudates.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    exudates = (exudates > 0).astype(np.uint8) & (optic_disc == 0)

    dark = 1.0 - green / (green.max() + 1e-8)
    hemorrhages = dark > float(np.percentile(dark, 97))
    hemorrhages = cv2.morphologyEx(hemorrhages.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    hemorrhages = (hemorrhages > 0).astype(np.uint8)

    microaneurysms = dark > float(np.percentile(dark, 99))
    microaneurysms = cv2.morphologyEx(microaneurysms.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    microaneurysms = (microaneurysms > 0).astype(np.uint8) & (optic_disc == 0)

    inverse_green = (255 * (1.0 - green)).astype(np.uint8)
    vessels = cv2.adaptiveThreshold(
        inverse_green,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    vessels = (vessels > 0).astype(np.uint8) & (optic_disc == 0)

    segmentation_result = {
        "opticDisc": {"mask": optic_disc.astype(bool), "count": int(np.sum(optic_disc)), "area": float(np.sum(optic_disc) / optic_disc.size)},
        "fovea": {"mask": np.zeros_like(optic_disc, dtype=bool), "count": 0, "area": 0.0},
        "vessels": {"mask": vessels.astype(bool), "count": int(np.sum(vessels)), "area": float(np.sum(vessels) / vessels.size)},
        "microaneurysms": {"mask": microaneurysms.astype(bool), "count": int(np.sum(microaneurysms)), "area": float(np.sum(microaneurysms) / microaneurysms.size)},
        "exudates": {"mask": exudates.astype(bool), "count": int(np.sum(exudates)), "area": float(np.sum(exudates) / exudates.size)},
        "haemorrhages": {"mask": hemorrhages.astype(bool), "count": int(np.sum(hemorrhages)), "area": float(np.sum(hemorrhages) / hemorrhages.size)},
        "neovascularisation": {"mask": np.zeros_like(optic_disc, dtype=bool), "count": 0, "area": 0.0},
        "features": {
            "vesselDensity": float(np.sum(vessels) / vessels.size),
            "tortuosity": float('nan'),
            "cupToDiscRatio": float('nan'),
        },
    }
    if not quality.passed:
        segmentation_result["warning"] = "Interpret results only after a gradable acquisition."
    return segmentation_result


def grading(segmentation_result: Dict[str, object], quality: QualityResult) -> Dict[str, object]:
    if not quality.passed:
        return {
            "level": None,
            "label": "Ungradable",
            "confidence": 0.0,
            "referable": False,
            "probabilities": [0.0] * 5,
            "evidence": {},
            "method": "interpretable rule-based baseline; configure trained ensemble weights for clinical use",
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
    probabilities = [0.0] * 5
    probabilities[level] = 1.0

    return {
        "level": level,
        "label": labels[level],
        "confidence": 0.55 + 0.1 * (level == 0),
        "referable": level >= 2,
        "probabilities": probabilities,
        "evidence": {
            "microaneurysms": ma,
            "exudates": ex,
            "haemorrhages": he,
            "neovascularisation": nv,
        },
        "method": "interpretable rule-based baseline; configure trained ensemble weights for clinical use",
    }


def trained_model_grading(path: str) -> Optional[Dict[str, object]]:
    if not os.path.exists(MODEL_CHECKPOINT):
        return None
    try:
        import torch
        from PIL import Image, ImageOps
        from train_retinal_model import SmallRetinalCNN
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location="cpu", weights_only=False)
        image_size = int(checkpoint.get("image_size", 160))
        model = SmallRetinalCNN(classes=int(checkpoint.get("classes", 5)))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        image = ImageOps.fit(Image.open(path).convert("RGB"), (image_size, image_size), method=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(((array - 0.5) / 0.5).transpose(2, 0, 1)).unsqueeze(0).contiguous()
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
        level = int(torch.argmax(probabilities).item())
        labels = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]
        return {
            "level": level,
            "label": labels[level],
            "confidence": float(probabilities[level].item()),
            "referable": level >= 2,
            "probabilities": probabilities.tolist(),
            "evidence": {},
            "method": "trained PyTorch CNN checkpoint",
        }
    except Exception:
        return None


def trained_diabetes_risk(path: str, quality: QualityResult) -> Dict[str, object]:
    """Run only a separately trained, verified-label diabetes-risk checkpoint.

    This is a research screening signal, never a clinical diabetes diagnosis.
    """
    unavailable = {
        "available": False,
        "risk": None,
        "confidence": None,
        "message": "Diabetes risk prediction is unavailable: train a separate model using retinal images with verified binary diabetes labels.",
    }
    if not quality.passed:
        unavailable["message"] = "Diabetes risk prediction is unavailable because the retinal image did not pass quality checks."
        return unavailable
    if not os.path.exists(DIABETES_MODEL_CHECKPOINT):
        return unavailable
    try:
        import torch
        from PIL import Image, ImageOps
        from train_retinal_model import SmallRetinalCNN
        checkpoint = torch.load(DIABETES_MODEL_CHECKPOINT, map_location="cpu", weights_only=False)
        if checkpoint.get("task") != "diabetes_risk" or int(checkpoint.get("classes", 0)) != 2:
            unavailable["message"] = "Diabetes risk checkpoint is invalid; it must be trained with --task diabetes_risk and verified 0/1 labels."
            return unavailable
        image_size = int(checkpoint.get("image_size", 160))
        model = SmallRetinalCNN(classes=2)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        image = ImageOps.fit(Image.open(path).convert("RGB"), (image_size, image_size), method=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(((array - 0.5) / 0.5).transpose(2, 0, 1)).unsqueeze(0).contiguous()
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
        probability = float(probabilities[1].item())
        return {
            "available": True,
            "risk": "Higher retinal-image diabetes risk" if probability >= 0.5 else "Lower retinal-image diabetes risk",
            "confidence": float(max(probabilities).item()),
            "probability": probability,
            "message": "Research screening estimate only. Confirm diabetes with A1C or plasma-glucose testing.",
        }
    except Exception:
        unavailable["message"] = "Diabetes risk prediction could not be loaded. Check the separately trained checkpoint."
        return unavailable


def run_pipeline(image: Optional[np.ndarray] = None, path: Optional[str] = None) -> Dict[str, object]:
    img = image if image is not None else load_image(path)
    quality, enhanced = quality_assessment(img)
    segmentation_result = segmentation(enhanced, quality)
    grading_result = trained_model_grading(path) if path and quality.passed else None
    if grading_result is None:
        grading_result = grading(segmentation_result, quality)
    model_method = grading_result.get("method", "interpretable rule-based baseline")
    diabetes_risk = trained_diabetes_risk(path, quality) if path else {
        "available": False, "risk": None, "confidence": None,
        "message": "Diabetes risk prediction requires a retinal image file and a separately trained verified-label model.",
    }
    summary = f"{('Gradable' if quality.passed else 'Ungradable')}: {grading_result['label']} (confidence {grading_result['confidence']:.2f})"
    return {
        "quality": quality,
        "segmentation": segmentation_result,
        "grading": grading_result,
        "image": enhanced,
        "summary": summary,
        "model_method": model_method,
        "diabetes_risk": diabetes_risk,
    }


def capture_camera_photo(save_path: Optional[str] = None, camera_index: int = 0) -> str:
    if save_path is None:
        save_dir = os.path.join(os.getcwd(), "captures")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "camera_capture.jpg")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}. Check that the laptop camera is available.")

    print("Opening laptop camera... hold still for a moment.")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("The camera did not return a frame.")

    success = cv2.imwrite(save_path, frame)
    if not success:
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
    candidates = [
        name,
        name.replace("_", " "),
        name.replace("-", " "),
    ]
    for candidate in candidates:
        if any(ch.isdigit() for ch in candidate):
            return candidate
    return name


def export_batch_report(results: List[Dict[str, object]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "batch_results.csv")
    json_path = os.path.join(output_dir, "batch_results.json")

    fieldnames = [
        "filename",
        "patient_id",
        "quality_passed",
        "summary",
        "grade_level",
        "grade_label",
        "referable",
        "confidence",
        "focus",
        "contrast",
        "vignetting_ratio",
        "field_of_view",
        "snr",
        "microaneurysms",
        "exudates",
        "haemorrhages",
    ]

    rows: List[Dict[str, object]] = []
    for result in results:
        patient_id = extract_patient_id(str(result["filename"]))
        rows.append({
            "filename": result["filename"],
            "patient_id": patient_id,
            "quality_passed": result["quality"].passed,
            "summary": result["summary"],
            "grade_level": result["grading"].get("level"),
            "grade_label": result["grading"].get("label"),
            "referable": result["grading"].get("referable"),
            "confidence": result["grading"].get("confidence"),
            "focus": result["quality"].focus,
            "contrast": result["quality"].contrast,
            "vignetting_ratio": result["quality"].vignetting_ratio,
            "field_of_view": result["quality"].field_of_view,
            "snr": result["quality"].snr,
            "microaneurysms": result["segmentation"]["microaneurysms"]["count"],
            "exudates": result["segmentation"]["exudates"]["count"],
            "haemorrhages": result["segmentation"]["haemorrhages"]["count"],
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)

    print(f"\nBatch report saved to: {output_dir}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print("\nSummary table:")
    headers = ["Patient ID", "Filename", "Passed", "Grade", "Referable", "Conf", "MA", "Ex", "He"]
    data_rows = []
    for row in rows:
        data_rows.append([
            row["patient_id"],
            row["filename"],
            str(row["quality_passed"]),
            str(row["grade_label"]),
            str(row["referable"]),
            f"{float(row['confidence'] or 0.0):.2f}",
            str(row["microaneurysms"]),
            str(row["exudates"]),
            str(row["haemorrhages"]),
        ])
    widths = [max(len(str(h)), max((len(str(r[i])) for r in data_rows), default=0)) for i, h in enumerate(headers)]
    header_line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    divider = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(divider)
    for row in data_rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))

    valid_rows = [row for row in rows if row["quality_passed"] is True]
    if valid_rows:
        best = max(valid_rows, key=lambda r: float(r["confidence"] or 0.0))
        worst = min(valid_rows, key=lambda r: float(r["confidence"] or 0.0))
        print("\nBest image:", best["filename"], "confidence=", f"{float(best['confidence'] or 0.0):.2f}")
        print("Worst image:", worst["filename"], "confidence=", f"{float(worst['confidence'] or 0.0):.2f}")
    else:
        print("\nBest image: N/A (no gradable cases)")
        print("Worst image: N/A (no gradable cases)")


def main() -> None:
    parser = argparse.ArgumentParser(description="DR Screening System Python runner")
    parser.add_argument("input", nargs="?", default=None, help="Image file, DICOM file, or folder of images to upload")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo image instead of real input")
    parser.add_argument("--batch", action="store_true", help="Process all supported images in the given folder")
    parser.add_argument("--camera", action="store_true", help="Use the laptop camera to capture a photo before running the model")
    parser.add_argument("--save-photo", default=None, help="Output filename for a captured camera image")
    args = parser.parse_args()

    print("DR Screening System - Python runner")
    print("Using PyTorch, OpenCV, SimpleITK, pydicom, scikit-learn, OpenModelica-compatible stack")

    if args.camera:
        try:
            save_path = capture_camera_photo(args.save_photo)
            print(f"Camera photo saved to: {save_path}")
            result = run_pipeline(path=save_path)
            print("\nResult:")
            print(result["summary"])
            print("Quality passed:", result["quality"].passed)
            print("Feedback:", result["quality"].feedback)
            print("Grade level:", result["grading"]["level"])
            print("Referable:", result["grading"]["referable"])
            print("Lesion counts:", result["segmentation"]["microaneurysms"]["count"], result["segmentation"]["exudates"]["count"], result["segmentation"]["haemorrhages"]["count"])
            return
        except Exception as exc:
            raise RuntimeError(f"Camera capture failed: {exc}") from exc

    image_source = args.input
    if args.demo or image_source is None:
        demo = load_image()
        result = run_pipeline(demo)
        print("\nResult:")
        print(result["summary"])
        print("Quality passed:", result["quality"].passed)
        print("Feedback:", result["quality"].feedback)
        print("Grade level:", result["grading"]["level"])
        print("Referable:", result["grading"]["referable"])
        print("Lesion counts:", result["segmentation"]["microaneurysms"]["count"], result["segmentation"]["exudates"]["count"], result["segmentation"]["haemorrhages"]["count"])
        return

    if args.batch and os.path.isdir(image_source):
        files = list_supported_files(image_source)
        if not files:
            raise FileNotFoundError(f"No supported image files found in folder: {image_source}")
        print(f"\nBatch processing {len(files)} files from: {image_source}")
        results = []
        for path in files:
            try:
                image = load_image(path)
                result = run_pipeline(image)
                result["filename"] = os.path.basename(path)
                results.append(result)
                print(f"\n[{os.path.basename(path)}] {result['summary']}")
                print(f"  quality_passed={result['quality'].passed} feedback={result['quality'].feedback}")
                print(f"  grade_level={result['grading']['level']} referable={result['grading']['referable']}")
            except Exception as exc:
                print(f"\n[{os.path.basename(path)}] ERROR: {exc}")
        export_batch_report(results, os.path.join(image_source, "batch_reports"))
        return

    image = load_image(image_source)
    result = run_pipeline(image)

    print("\nResult:")
    print(result["summary"])
    print("Quality passed:", result["quality"].passed)
    print("Feedback:", result["quality"].feedback)
    print("Grade level:", result["grading"]["level"])
    print("Referable:", result["grading"]["referable"])
    print("Lesion counts:", result["segmentation"]["microaneurysms"]["count"], result["segmentation"]["exudates"]["count"], result["segmentation"]["haemorrhages"]["count"])


if __name__ == "__main__":
    main()
