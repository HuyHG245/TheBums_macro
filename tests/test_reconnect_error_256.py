from pathlib import Path
import cv2
import numpy as np
import pytest
from thebums_reconnect import find_leave_disconnect, find_error_256_leave, ImageMatcher
ROOT=Path(__file__).resolve().parents[1]
def frame():
    f=np.zeros((1079,1919,3),np.uint8)
    f[400:655,760:1160]=cv2.imread(str(ROOT/'tests/fixtures/error_256_dialog.png'))
    return f
@pytest.mark.parametrize('variant', ['original','client','window','resize'])
def test_256_prefers_leave(variant):
    f=frame()
    if variant=='client': f=f[23:1032]
    elif variant=='window': f=f[:1032]
    elif variant=='resize': f=cv2.resize(f,(1650,927))
    f=cv2.resize(f,None,fx=1080/f.shape[0],fy=1080/f.shape[0])
    hit=find_leave_disconnect(f)
    assert hit and hit.source=='opencv_error_256'
    assert .44*f.shape[1]<hit.x<.46*f.shape[1]
    assert .55*f.shape[0]<hit.y<.61*f.shape[0]
def test_256_requires_body_and_leave():
    for box in [(490,560,780,1140),(605,630,830,910),(410,446,885,1040)]:
        f=frame(); y,Y,x,X=box;f[y:Y,x:X]=0
        assert find_error_256_leave(f) is None
    assert find_error_256_leave(np.zeros_like(frame())) is None
