import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import python_dr_runner


def test_build_batch_summary_handles_mixed_results():
    results = [
        {
            "filename": "good_1.png",
            "quality": {"passed": True},
            "grading": {"level": 0, "label": "No DR", "confidence": 0.93, "referable": False},
            "segmentation": {"microaneurysms": {"count": 1}, "exudates": {"count": 0}, "haemorrhages": {"count": 0}},
        },
        {
            "filename": "bad_1.png",
            "quality": {"passed": False},
            "grading": {"level": None, "label": "Ungradable", "confidence": 0.0, "referable": False},
            "segmentation": {"microaneurysms": {"count": 0}, "exudates": {"count": 0}, "haemorrhages": {"count": 0}},
        },
        {
            "filename": "good_2.png",
            "quality": {"passed": True},
            "grading": {"level": 2, "label": "Moderate NPDR", "confidence": 0.81, "referable": True},
            "segmentation": {"microaneurysms": {"count": 4}, "exudates": {"count": 9}, "haemorrhages": {"count": 7}},
        },
    ]

    summary = python_dr_runner.build_batch_summary(results)

    assert summary["total"] == 3
    assert summary["quality_pass_rate"] == pytest.approx(2 / 3)
    assert summary["referable_rate"] == pytest.approx(1 / 3)
    assert summary["average_confidence"] == pytest.approx((0.93 + 0.81) / 2)
    assert summary["grade_counts"]["Moderate NPDR"] == 1
