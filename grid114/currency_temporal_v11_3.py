"""Offline frame selection and review. No visibility proof or ledger writes."""
from dataclasses import dataclass
from collections import defaultdict
import hashlib, math
import cv2
import numpy as np
from .currency_quality_v11_2 import quality_features, quality_mode
from .currency_v7 import preprocess

def geometry(image):
    """Conservative suspicion flags, not a detector of invisible prefix loss."""
    if image is None or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 10:
        return {'flags':['invalid_crop'],'span':0,'gap':0,'sharpness':0}
    h,w=image.shape[:2]
    prep=preprocess(image,.46)
    if prep is None:return {'flags':['no_text'],'span':0,'gap':0,'sharpness':0}
    _,m=prep
    # Exclude the lower rarity stripe and any rows dominated by a horizontal border.
    m=m.copy();m[round(.77*h):]=False;m[m.sum(1)>.65*w]=False
    # Discard isolated one-pixel speckles only for geometry; preserve original crop for decoding.
    n,components,stats,_=cv2.connectedComponentsWithStats(m.astype(np.uint8),8)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]==1:m[components==i]=False
    xs=np.flatnonzero(m.any(0));f=quality_features(image);flags=[]
    if not len(xs):return {'flags':['no_text'],'span':0,'gap':0,'sharpness':f[1]}
    gap=int(np.diff(xs).max()-1) if len(xs)>1 else 0
    if gap>.30*h:flags.append('separated_text_or_overlay')
    if xs[0]<=2 or xs[-1]>=w-3:flags.append('text_near_crop_edge')
    if quality_mode(image)!='normal':flags.append('low_quality')
    return dict(flags=flags,span=int(xs[-1]-xs[0]+1),gap=gap,sharpness=f[1])

@dataclass(frozen=True)
class Capture:
    event:str
    slot:str
    currency:str
    stream:str
    seconds:float
    frame_id:str
    crop_hash:str
    candidate:int|None
    flags:tuple
    sharpness:float

def digest(image):
    return hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()

def select(captures,min_frames=3,min_span=.4):
    """Review consensus across one timebase at a time; do not infer visibility."""
    def result(status,value=None,best=None,n=0):
        return dict(status=status,candidate=value,best_frame=best,distinct_clear_crops=n,review_only=True,ledger_allowed=False)
    if min_frames<3 or not math.isfinite(min_span) or min_span<=0:raise ValueError('invalid thresholds')
    if not captures:return result('no_evidence')
    identities={(c.event,c.slot,c.currency) for c in captures}
    if len(identities)!=1 or any(not k for k in next(iter(identities))) or captures[0].currency not in ('gold','gem'):return result('mixed_or_invalid_identity')
    if any(not c.stream or not c.frame_id or not c.crop_hash or not math.isfinite(c.seconds) or c.seconds<0 or not math.isfinite(c.sharpness) or (c.candidate is not None and (type(c.candidate) is not int or c.candidate<0)) for c in captures):return result('invalid_metadata')
    # A repeated frame identifier with conflicting evidence is not a new observation.
    by_id=defaultdict(set)
    for c in captures:by_id[c.stream,c.frame_id].add((c.crop_hash,c.candidate,c.seconds,c.flags))
    if any(len(x)>1 for x in by_id.values()):return result('conflicting_frame_metadata')
    # Do not suppress disagreements simply by preferring a sharper frame.
    values={c.candidate for c in captures if c.candidate is not None}
    if len(values)>1:return result('conflicting_candidates')
    clear=[c for c in captures if c.candidate is not None and not c.flags]
    by_stream=defaultdict(list)
    for c in clear:by_stream[c.stream].append(c)
    eligible=[];max_n=0
    for stream,items in by_stream.items():
        hs=set();ts=set();ids=set();unique=[]
        for c in sorted(items,key=lambda c:(-c.sharpness,c.seconds,c.frame_id)):
            if c.crop_hash in hs or c.seconds in ts or c.frame_id in ids:continue
            unique.append(c);hs.add(c.crop_hash);ts.add(c.seconds);ids.add(c.frame_id)
        max_n=max(max_n,len(unique))
        if len(unique)>=min_frames and max(ts)-min(ts)>=min_span:
            eligible.extend(unique)
    if not eligible:return result('insufficient_distinct_clear_evidence',n=max_n)
    best=max(eligible,key=lambda c:c.sharpness)
    return result('consistent_review',best.candidate,best.stream+':'+best.frame_id,max_n)
