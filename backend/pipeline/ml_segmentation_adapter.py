"""Optional ML segmentation adapter and stable classical fallback."""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .robust_preprocessing import classical_segmentation


def segment_fundus(
    image: np.ndarray,
    quality_passed: bool = True,
    use_ml: bool = True,
    model_dir: Optional[str] = None,
) -> Dict[str, object]:
    if not use_ml:
        return classical_segmentation(image, quality_passed)

    directory = Path(model_dir) if model_dir else Path(__file__).resolve().parents[2] / "models" / "segmentation"
    checkpoint = directory / "attention_unet_retinal.pt"
    if not checkpoint.exists():
        return classical_segmentation(image, quality_passed)

    try:
        from .ml_segmentation import MLSegmenter, SegmentationConfig
        segmenter = MLSegmenter(SegmentationConfig(model_dir=str(directory)))
        result = segmenter.segment(image)
        if "hemorrhages" in result and "haemorrhages" not in result:
            result["haemorrhages"] = result.pop("hemorrhages")
        if not quality_passed:
            result["warning"] = "Interpret results only after a gradable acquisition."
        result["method"] = "attention U-Net"
        return result
    except Exception:
        return classical_segmentation(image, quality_passed)
