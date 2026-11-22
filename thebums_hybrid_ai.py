"""Opt-in, offline CPU ONNX verification of borderline legacy OpenCV matches.

This module NEVER captures the display, modifies templates, or sends input.
The caller MUST perform the existing OpenCV recognition first and retain its
spatial and multi-frame safety checks. No network/API use or auto-training.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging
import math

import cv2
import numpy as np

LOG = logging.getLogger(__name__)

# Small windows below the original thresholds: opt-in support, not a new detector.
# Score floors prevent user-configured thresholds from permitting very weak matches.
GATES = {
    'retry_text': (.04, .88),
    'alt_retry_button': (.04, .88),
    'rc_reconnect': (.035, .865),
    **{f'rc_{name}': (.035, .80) for name in (
        'slot_a', 'select', 'play_small', 'codes', 'lobby', 'ps_input',
        'generate', 'play_large', 'missions', 'outskirts',
        'outskirts_selected', 'create', 'start')},
    **{f'loot_{name}': (.045, .77) for name in (
        'scroll', 'body', 'defence', 'offence', 'support', 'vessel',
        'attack_serum', 'armor_serum', 'female_serum', 'beast_serum',
        'colossal_serum')},
}


@dataclass(frozen=True)
class HybridDecision:
    accepted: bool
    cv_score: float
    ai_score: float | None = None
    reason: str = ''


class CpuOnnxBackend:
    """Explicitly configured local image classifier: RGB NCHW float32.

    Model must be validated AND probability-calibrated on real gameplay data
    outside this application. A softmax score is not itself a calibrated score.
    """
    def __init__(self, model_dir: str | Path):
        root = Path(model_dir).resolve()
        meta = json.loads((root / 'hybrid_manifest.json').read_text(encoding='utf-8'))
        if meta.get('schema_version') != 1 or meta.get('validated') is not True or meta.get('calibrated') is not True:
            raise ValueError('Model needs schema_version=1, validated=true, calibrated=true')
        filename = meta.get('model')
        if not isinstance(filename, str) or not filename.endswith('.onnx') or Path(filename).name != filename:
            raise ValueError('Model must be a local .onnx filename in models/')
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        names = meta.get('labels')
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
            raise ValueError('labels must be a non-empty string list')
        if len(set(names)) != len(names) or '__background__' not in names or any(
                n not in GATES and n != '__background__' for n in names):
            raise ValueError('labels must include __background__ and allowlisted template names')
        dims = meta.get('input_hw', [128, 128])
        if (not isinstance(dims, list) or len(dims) != 2 or
                any(type(n) is not int or not 32 <= n <= 512 for n in dims)):
            raise ValueError('input_hw must be a pair of integers between 32 and 512')
        mean, std = meta.get('mean', [0.485, 0.456, 0.406]), meta.get('std', [0.229, 0.224, 0.225])
        if any(not isinstance(a, list) or len(a) != 3 for a in (mean, std)):
            raise ValueError('mean and std must have three components')
        for channel in mean + std:
            if isinstance(channel, bool) or not isinstance(channel, (int, float)) or not math.isfinite(channel):
                raise ValueError('Invalid normalization coefficient')
        if any(v <= 0 for v in std):
            raise ValueError('std components must be positive')
        try:
            import onnxruntime as ort  # optional; Standard Mode has no ORT requirement
        except ImportError as exc:
            raise RuntimeError('Install optional CPU dependency: pip install onnxruntime') from exc
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=opts,
                                            providers=['CPUExecutionProvider'])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError('Expected one ONNX input and one output')
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.labels = {n: i for i, n in enumerate(names)}
        self.size = (dims[1], dims[0])
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

    def score(self, crop_bgr: np.ndarray, label: str) -> float | None:
        index = self.labels.get(label)
        if index is None:
            return None
        image = cv2.resize(crop_bgr, self.size, interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose((image - self.mean) / self.std, (2, 0, 1))[None]
        raw = np.asarray(self.session.run([self.output_name], {self.input_name: chw})[0])
        if raw.shape != (1, len(self.labels)) or not np.all(np.isfinite(raw)):
            raise ValueError('Model must output finite [1,N] class probabilities')
        probs = raw[0]
        if np.any(probs < 0) or np.any(probs > 1) or not np.isclose(probs.sum(), 1, atol=0.02):
            raise ValueError('Model output must be calibrated probabilities (not logits)')
        return float(probs[index])


class HybridAdvisor:
    """Only verify a CV-generated box close to a pre-existing threshold.

    Candidate is a legacy (center_x, center_y, cv_score, width, height) tuple.
    The optional ROI is a rectangle (left, top, right, bottom) in frame pixels.
    A candidate outside ROI, unvalidated backend, or backend error fails closed.
    """
    def __init__(self, backend, *, min_ai_score: float = 0.98):
        self.backend = backend
        self.min_ai_score = float(min_ai_score)
        if not .95 <= self.min_ai_score <= 1:
            raise ValueError('Use strict AI confidence >= 0.95')

    def verify(self, frame: np.ndarray, label: str, candidate: tuple,
               threshold: float, roi: tuple[int, int, int, int] | None = None) -> HybridDecision:
        if label not in GATES:
            return HybridDecision(False, 0, reason='label not allowlisted')
        try:
            cx, cy, cv_score, tw, th = candidate
            cv_score = float(cv_score)
            gap, floor = GATES[label]
            if (not np.isfinite(cv_score) or not 0 <= cv_score <= 1 or
                    not 0 < threshold <= 1 or cv_score >= threshold or
                    cv_score < max(floor, threshold - gap)):
                return HybridDecision(False, cv_score, reason='not borderline')
            if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
                return HybridDecision(False, cv_score, reason='invalid frame')
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in (cx, cy, tw, th)):
                return HybridDecision(False, cv_score, reason='invalid bounding box')
            if tw <= 0 or th <= 0:
                return HybridDecision(False, cv_score, reason='invalid bounding box')
            x0, y0 = round(cx - tw / 2), round(cy - th / 2)
            x1, y1 = x0 + round(tw), y0 + round(th)
            h, w = frame.shape[:2]
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 <= x0 or y1 <= y0:
                return HybridDecision(False, cv_score, reason='box outside frame')
            if roi is not None:
                rx0, ry0, rx1, ry1 = roi
                if x0 < rx0 or x1 > rx1 or y0 < ry0 or y1 > ry1:
                    return HybridDecision(False, cv_score, reason='box outside allowed ROI')
            # No API calls, no photo files, just an in-memory CPU crop.
            score = self.backend.score(frame[y0:y1, x0:x1].copy(), label)
            if score is None or not isinstance(score, (int, float, np.floating)) or not math.isfinite(float(score)):
                return HybridDecision(False, cv_score, reason='AI score unavailable')
            score = float(score)
            if not 0 <= score <= 1:
                return HybridDecision(False, cv_score, reason='invalid AI probability')
            return HybridDecision(score >= self.min_ai_score, cv_score, score,
                                  'AI verified' if score >= self.min_ai_score else 'AI rejected')
        except Exception as exc:
            LOG.warning('Hybrid AI error for %s (fail closed): %s', label, type(exc).__name__)
            return HybridDecision(False, float(candidate[2]) if len(candidate) > 2 else 0,
                                  reason='AI backend failed')
