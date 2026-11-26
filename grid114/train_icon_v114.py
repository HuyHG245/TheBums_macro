import cv2
import numpy as np


def icon(tile):
    top=tile[:max(1,round(.64*len(tile)))]
    return cv2.resize(top,(16,12),interpolation=cv2.INTER_AREA).astype('float32')/255

def scores(feature, refs):
    a=feature.reshape(12,48)
    return np.array([max(float(cv2.matchTemplate(a,ref.reshape(12,48),cv2.TM_CCOEFF_NORMED)[0,0]) for ref in cls) for cls in refs])
