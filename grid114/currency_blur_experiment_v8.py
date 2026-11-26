"""Experimental blur fallback; review candidates only. Never write to session ledger."""
import cv2
import numpy as np
from .currency_v7 import fit, raid_review

def cleanup(image):
    a=image.copy()
    h,w=a.shape[:2]
    lo=a.min(2).astype(float)
    hi=a.max(2).astype(float)
    a[(hi-lo)>45]=0
    a[:,:3]=0
    a[:,-3:]=0
    for y in range(round(h*.55),h):
        if (a[y].min(1)>65).sum()>.55*w:
            a[y]=0
    return a

def fit_blur_bank(records, original):
    augmented=[]
    for source,label,value,image in records:
        for sigma in (.65,1.05):
            augmented.append((source,label,value,cleanup(cv2.GaussianBlur(image,(5,5),sigma))))
    extra=fit(augmented)
    return {k:np.concatenate([v,extra[k]]) if k in extra else v.copy() for k,v in original.items()}

def review_candidate(image, label, original, blurred):
    value,detail=raid_review(image,original,label)
    if value is not None:
        return value,"v7",detail
    value,detail=raid_review(cleanup(image),blurred,label)
    return value,"blur_review" if value is not None else "unresolved",detail
