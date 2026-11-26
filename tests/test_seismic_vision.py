"""Synthetic compositing from the supplied ready icon; no real cooldown labels."""
from pathlib import Path
import cv2
import numpy as np
import pytest
from raid_vision import seismic_ready

ROOT = Path(__file__).resolve().parents[1] / 'image'
TEMPLATE = cv2.imread(str(ROOT/'seismic_shock.png'))


def frame_with(icon, x=520, y=480):
    frame = np.full((600, 1000, 3), 25, np.uint8)
    h, w = icon.shape[:2]
    frame[y:y+h, x:x+w] = icon
    return frame


@pytest.mark.parametrize('scale', [.45, .55, .65, .75, .85, 1., 1.15, 1.30])
def test_ready_at_scales(scale):
    icon = cv2.resize(TEMPLATE, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    assert seismic_ready(frame_with(icon), {'normal': TEMPLATE}) is True


def test_dimmed_icon_not_ready_even_with_high_shape_similarity():
    dim = (TEMPLATE.astype(float)*.45).astype(np.uint8)
    assert seismic_ready(frame_with(dim), {'normal': TEMPLATE}) is False


def test_unrelated_heatwave_icon_not_ready():
    heat = cv2.imread(str(ROOT/'heatwave.png'))
    assert seismic_ready(frame_with(heat), {'normal': TEMPLATE}) is False


def test_icon_outside_hud_not_ready():
    assert seismic_ready(frame_with(TEMPLATE, y=100), {'normal': TEMPLATE}) is False


def test_missing_template_preserves_legacy_fallback():
    assert seismic_ready(frame_with(TEMPLATE), {}) is None
