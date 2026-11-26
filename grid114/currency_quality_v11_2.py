"""Offline quality-aware candidate routing. Never writes currency ledger."""
import cv2
import numpy as np
import torch
from . import currency_ctc_v11 as v11
from . import currency_degradation_v11_1 as v111

def quality_features(a):
    # Text zone only, excluding colored rarity stripe and crop borders.
    h,w=a.shape[:2]
    g=a.min(2).astype(np.float32)
    g=g[round(.16*h):round(.77*h),2:-2]
    if g.size==0:return [0.]*7
    hi=float(np.percentile(g,95))
    g=np.clip((g-35)/max(50,hi-35),0,1)
    active=g>.3
    if not active.any():return [hi,0,0,0,0,0,0]
    xs=np.flatnonzero(active.any(0))
    g=g[:,max(0,xs[0]-1):min(g.shape[1],xs[-1]+2)]
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    return [hi,float(cv2.Laplacian(g,cv2.CV_32F).var()),float(np.mean(gx**2+gy**2)),float(np.mean((g>.15)&(g<.55))),float(np.mean(g>.7)),float(np.mean(g>.3)),float(np.std(g))]

def quality_mode(a):
    f=quality_features(a)
    return "render" if f[1] <= 0.390150249004364 or f[5] > 0.27247025072574615 else "normal"

@torch.no_grad()
def predict(model11,model111,a):
    mode=quality_mode(a)
    if mode=="render":
        return v111.predict(model111,a),mode
    model11.eval()
    return v11.greedy(model11(torch.from_numpy(v11.raw_tensor(a)[None])))[0],mode

def review(model11,model111,a):
    mode=quality_mode(a)
    r=v111.review(model111,a) if mode=="render" else v11.review_candidate_ctc(model11,a)
    return dict(r,mode=mode,ledger_allowed=False)

def hybrid_review(image,label,original_bank,blur_bank,model11,model111):
    from .currency_blur_experiment_v8 import review_candidate
    primary,method,_=review_candidate(image,label,original_bank,blur_bank)
    if primary is not None:
        return dict(candidate=primary,method="opencv_"+method,review_only=True,ledger_allowed=False)
    result=review(model11,model111,image)
    return dict(result,method="ai_fallback")
