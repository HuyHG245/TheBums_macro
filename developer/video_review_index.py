"""Offline review index for AOT:R recordings: suggest timestamps, not labels.

This utility reads a local video and original recognition templates. It writes
ONLY metadata (no video frames, crops or credentials) under datasets/hybrid/.
Every suggested box must be reviewed in the private Video Dataset Collector
before it becomes a training sample. A match score is NOT ground truth.

Example:
  python developer/video_review_index.py --video "recording.mp4"
  python developer/video_dataset_collector.py --video "recording.mp4"
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT / "datasets" / "hybrid"
# Separate context-only images from labels allowed by the training pipeline.
CONTEXT_ONLY = ("result_completed", "result_failed", "finish", "retry_button")
TARGET_LABELS = (
    "retry_text", "alt_retry_button", "rc_reconnect",
    "loot_scroll", "loot_body", "loot_defence", "loot_offence",
    "loot_support", "loot_vessel", "loot_attack_serum",
    "loot_armor_serum", "loot_female_serum", "loot_beast_serum",
    "loot_colossal_serum",
)
DEFAULT_LABELS = CONTEXT_ONLY + TARGET_LABELS


def fingerprint(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_templates(assets: Path, labels=DEFAULT_LABELS):
    templates = {}
    for label in labels:
        if label not in DEFAULT_LABELS:
            raise ValueError("Unapproved template label: " + label)
        path = Path(assets) / "image" / (label + ".png")
        if not path.is_file():
            continue
        template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if template is None or min(template.shape) < 8:
            continue
        if float(template.std()) < 2.0:
            # Flat templates produce unreliable CCOEFF scores.
            continue
        templates[label] = template
    return templates


def detect_frame(frame: np.ndarray, templates: dict, scales=(0.85, 1.0, 1.15)):
    """Return tentative label/score/rectangle; never interpreted as truth."""
    if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("Expected BGR frame")
    grey = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    height, width = grey.shape[:2]
    results = []
    for label, template in templates.items():
        best = None
        for scale in scales:
            if scale <= 0 or scale > 4:
                raise ValueError("Invalid scale")
            tw = max(1, round(template.shape[1] * scale))
            th = max(1, round(template.shape[0] * scale))
            if min(tw, th) < 8 or tw > width or th > height:
                continue
            resized = cv2.resize(template, (tw, th), interpolation=cv2.INTER_AREA)
            if float(resized.std()) < 2.0:
                continue
            corr = cv2.matchTemplate(grey, resized, cv2.TM_CCOEFF_NORMED)
            _, maximum, _, location = cv2.minMaxLoc(corr)
            if not np.isfinite(maximum):
                continue
            if best is None or maximum > best["score"]:
                best = {"label": label, "score": round(float(maximum), 5),
                        "bbox": [int(location[0]), int(location[1]), tw, th]}
        if best is not None:
            results.append(best)
    return results


def scan_video(video: Path, templates: dict, *, sample_fps=1.0,
               threshold=0.82, min_gap=2.5, max_samples=1800):
    """Scan at bounded rate, keep sparse candidate times in a single session."""
    if not 0.2 <= sample_fps <= 4 or not 0.5 <= threshold <= 1.0:
        raise ValueError("Unsafe sample rate or matching threshold")
    if not 0 <= min_gap <= 60 or not 1 <= max_samples <= 20000:
        raise ValueError("Invalid scan limits")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError("Cannot open video: " + str(video))
    events = []
    last_by_label = {}
    sampled = 0
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0 or not count:
            raise ValueError("Video FPS or frame count unavailable")
        step = max(1, round(fps / sample_fps))
        for index in range(0, count, step):
            if sampled >= max_samples:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            sampled += 1
            seconds = index / fps
            for candidate in detect_frame(frame, templates):
                label = candidate["label"]
                if candidate["score"] < threshold:
                    continue
                if seconds - last_by_label.get(label, -1e6) < min_gap:
                    continue
                candidate["seconds"] = round(seconds, 3)
                candidate["context_only"] = label in CONTEXT_ONLY
                events.append(candidate)
                last_by_label[label] = seconds
    finally:
        capture.release()
    events.sort(key=lambda item: (item["seconds"], -item["score"], item["label"]))
    return {"review_schema": 1, "source_sha256": fingerprint(video),
            "sample_fps": sample_fps, "threshold": threshold,
            "sampled_frames": sampled, "event_count": len(events), "events": events,
            "warning": "Suggestions are NOT verified labels. Review ROI and privacy before saving."}


def load_index(path: Path, *, source_sha256: str, duration: float):
    """Validate local review metadata; reject a mismatched recording."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("review_schema") != 1 or data.get("source_sha256") != source_sha256:
        raise ValueError("Review index does not match the selected video")
    raw = data.get("events")
    if not isinstance(raw, list) or len(raw) > 20000:
        raise ValueError("Invalid review events")
    output = []
    for event in raw:
        if not isinstance(event, dict) or event.get("label") not in DEFAULT_LABELS:
            raise ValueError("Invalid suggested event")
        seconds, rect, score = event.get("seconds"), event.get("bbox"), event.get("score")
        if (not isinstance(seconds, (float, int)) or not np.isfinite(seconds) or
                not 0 <= seconds <= duration + 1.0 or not isinstance(rect, list) or len(rect) != 4 or
                any(type(v) is not int for v in rect) or
                rect[0] < 0 or rect[1] < 0 or min(rect[2:]) < 8 or
                not isinstance(score, (float, int)) or not 0 <= score <= 1):
            raise ValueError("Invalid suggested event geometry or score")
        output.append(event)
    return sorted(output, key=lambda item: item["seconds"])


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=PROJECT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--min-gap", type=float, default=2.5)
    parser.add_argument("--labels", default="", help="Comma-separated subset of available template names")
    args = parser.parse_args()
    labels = tuple(v.strip() for v in args.labels.split(",") if v.strip()) or DEFAULT_LABELS
    templates = load_templates(args.assets, labels)
    if not templates:
        parser.error("No usable legacy PNG templates found")
    result = scan_video(args.video, templates, sample_fps=args.fps,
                        threshold=args.threshold, min_gap=args.min_gap)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / ("review_" + result["source_sha256"][:12] + ".json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Offline review index:", output)
    print("Candidates:", result["event_count"], "(NOT labeled training data)")
    print("Open the same video with developer/video_dataset_collector.py")


if __name__ == "__main__":
    cli()
