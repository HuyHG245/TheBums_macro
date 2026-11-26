"""Unit checks for PRIVATE timestamp suggestions; not AI model validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "developer"))
from video_review_index import detect_frame, load_index, load_templates


def test_detect_known_template_on_a_synthetic_frame():
    rng = np.random.default_rng(1234)
    template = rng.integers(0, 256, size=(20, 32), dtype=np.uint8)
    frame = np.zeros((100, 140, 3), dtype=np.uint8)
    frame[22:42, 57:89] = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    result = detect_frame(frame, {"retry_text": template}, scales=(1.0,))
    assert len(result) == 1
    assert result[0]["label"] == "retry_text"
    assert result[0]["score"] > .99
    assert result[0]["bbox"] == [57, 22, 32, 20]


def test_load_templates_ignores_missing_and_flat(tmp_path):
    folder = tmp_path / "image"
    folder.mkdir()
    assert load_templates(tmp_path, ("retry_text",)) == {}
    assert cv2.imwrite(str(folder / "retry_text.png"), np.zeros((20, 20), dtype=np.uint8))
    assert load_templates(tmp_path, ("retry_text",)) == {}
    with pytest.raises(ValueError, match="Unapproved"):
        load_templates(tmp_path, ("not_a_real_label",))


def test_index_rejects_wrong_video_and_out_of_range(tmp_path):
    digest = hashlib.sha256(b"original-private-video").hexdigest()
    index = tmp_path / "review.json"
    metadata = {
        "review_schema": 1, "source_sha256": digest,
        "events": [{"label": "retry_text", "seconds": 3.5,
                    "bbox": [10, 20, 90, 40], "score": .9}]
    }
    index.write_text(json.dumps(metadata), encoding="utf-8")
    assert len(load_index(index, source_sha256=digest, duration=10)) == 1
    with pytest.raises(ValueError, match="does not match"):
        load_index(index, source_sha256=hashlib.sha256(b"copy").hexdigest(), duration=10)
    with pytest.raises(ValueError, match="Invalid suggested event"):
        load_index(index, source_sha256=digest, duration=2)


def test_context_only_markers_not_training_labels():
    from video_review_index import CONTEXT_ONLY, TARGET_LABELS
    from hybrid_dataset import allowed_labels
    assert "result_completed" in CONTEXT_ONLY
    assert "result_failed" in CONTEXT_ONLY
    assert "result_completed" not in allowed_labels()
    assert "result_failed" not in allowed_labels()
    assert "retry_text" in TARGET_LABELS
