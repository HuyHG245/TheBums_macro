"""Regression using only the dialog cropped from the user's 2026-11-26 image."""
from pathlib import Path
import cv2
import numpy as np
from thebums_reconnect import ImageMatcher, find_scaled_reconnect, find_leave_disconnect

ROOT = Path(__file__).resolve().parents[1]


def frame():
    crop = cv2.imread(str(ROOT / 'tests/fixtures/idle_278_dialog.png'))
    canvas = np.zeros((1079, 1919, 3), np.uint8)
    canvas[390:660,750:1170] = crop
    return canvas


def normalize(f):
    scale = f.shape[0] / 1080
    return cv2.resize(f, None, fx=1/scale, fy=1/scale,
                      interpolation=cv2.INTER_CUBIC) if abs(scale-1)>.01 else f


def test_idle_278_original_button():
    f = frame()
    m = ImageMatcher(str(ROOT/'image')).find(f, 'reconnect')
    assert m and m.source == 'opencv' and m.confidence >= .90
    assert abs(m.x-1052)<5 and abs(m.y-615)<5
    assert find_leave_disconnect(f) is None


def test_idle_278_scaled_capture_variants():
    original = frame()
    matcher = ImageMatcher(str(ROOT/'image'))
    for f, expected in ((original[23:1032],(1126,633)),
                        (original[:1032],(1101,643)),
                        (cv2.resize(original,(1650,927)),(1054,616))):
        g=normalize(f)
        m=matcher.find(g,'reconnect')
        assert m and m.source=='opencv_scaled_dialog'
        assert abs(m.x-expected[0])<=5 and abs(m.y-expected[1])<=5
        assert find_leave_disconnect(g) is None


def test_scaled_fallback_requires_title_and_button():
    matcher=ImageMatcher(str(ROOT/'image'))
    tpl=matcher.templates['reconnect']
    for y0,y1,x0,x1 in ((400,455,870,1040),(590,640,960,1140)):
        f=frame()
        f[y0:y1,x0:x1]=0
        assert find_scaled_reconnect(normalize(f[23:1032]),tpl) is None
    assert find_scaled_reconnect(np.zeros_like(frame()),tpl) is None
