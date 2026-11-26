"""Offline experimental sequence decoder; no session writes."""
from functools import lru_cache
import collections
import cv2
import numpy as np
ALIGNMENTS = {
('S01','gold'):[13,20,28,35,38,45,51,58],('S07','gold'):[10,15,20,25,28,34,37,44],('S08','gold'):[12,21,30,39,43,52,60,70],('S10','gold'):[12,17,22,26,28,33,38,43],('S10','gem'):[26,30,36,42],('S11','gold'):[8,13,18,24,26,32,38,44],('S12','gold'):[14,22,28,33,37,45,53,62],('S12','gem'):[42,48,54,62],('S13','gold'):[14,22,28,36,39,46,53,62],('S14','gold'):[14,22,28,35,38,45,53,62],('S15','gold'):[16,24,30,37,40,46,54,62],('S20','gold'):[16,22,29,37,40,46,53,60],('S20','gem'):[40,46,54,62],('S21','gold'):[10,15,19,26,31,39,42,49,55,61],('S24','gold'):[14,17,21,28,34,40,43,49,55,62],('S25','gold'):[14,22,29,36,39,46,53,62],('S25','gem'):[40,46,54,62],('S28','gold'):[14,22,29,37,40,46,54,62]}
def preprocess(image,t):
 if not isinstance(image,np.ndarray) or image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8 or min(image.shape[:2])<10:return None
 h,w=image.shape[:2];g=image.min(2).astype(np.float32);g[:round(h*.15)]=0;g[round(h*.85):]=0;g[:,:2]=0;g[:,-2:]=0
 hi=np.percentile(g[g>70],90) if (g>70).any() else 0
 if hi<90:return None
 m=g>=max(78,hi*t)
 if (m.sum(1)>=2).sum()<3:return None
 g=np.clip((g-max(35,hi*.21))/(hi*.79),0,1);g[~m]*=.20
 return g,m
def glyph(g,m,l,r):
 ys,xs=np.where(m[:,l:r]);
 if not len(xs):return None
 y0,y1=ys.min(),ys.max()+1;x0,x1=xs.min()+l,xs.max()+1+l
 a=g[y0:y1,x0:x1];raw=m[y0:y1,x0:x1].astype('float32')
 return np.r_[cv2.resize(a,(12,18)).ravel(),cv2.resize(raw,(12,18)).ravel(),np.log((x1-x0)/(y1-y0)),(y1-y0)/g.shape[0],y1/g.shape[0]]
def fit(records,excluded=None):
 bank=collections.defaultdict(list)
 for s,label,quantity,image in records:
  if s==excluded or (s,label) not in ALIGNMENTS:continue
  cuts=ALIGNMENTS[s,label];chars=f'{quantity:,}';assert len(cuts)==len(chars)+1,(s,label)
  for t in (.46,.52,.60,.69):
   prep=preprocess(image,t)
   if prep is None:continue
   g,m=prep
   for c,l,r in zip(chars,cuts,cuts[1:]):
    p=glyph(g,m,l,r)
    if p is not None and ((c==',' and p[-2]<.34) or (c!=',' and p[-2]>.23)):bank[c].append(p)
 return {c:np.asarray(v) for c,v in bank.items()}
def decode(image,bank,t=.52):
 prep=preprocess(image,t)
 if prep is None:return None
 g,m=prep;h,w=m.shape;xx=np.flatnonzero(m.any(0));l,r=int(xx[0]),int(xx[-1])+1
 if not .025<=(w-r)/h<=.42:return None
 @lru_cache(None)
 def choices(x):
  out=[]
  for e in range(x+1,min(r,x+round(h*.61))+1):
   p=glyph(g,m,x,e)
   if p is None or not m[:,e-1].any():continue
   for c,q in bank.items():
    if c==',' and (p[-2]>.34 or p[-1]<.55):continue
    if c!=',' and (p[-2]<.25 or p[-2]>.75 or p[-3]>np.log(1.3)):continue
    a,b=p[216:432],q[:,216:432];dice=1-(2*np.minimum(a,b).sum(1)+1e-5)/(a.sum()+b.sum(1)+1e-5)
    d=float(np.min(.58*np.mean(abs(p[:216]-q[:,:216]),axis=1)+.28*dice+np.sum(abs(p[-3:]-q[:,-3:])*[.09,.20,.15],axis=1)))
    if d>.29:continue
    cut=0 if e==r or not m[:,e].any() else .022+.013*min(m[:,e-1].sum(),m[:,e].sum())
    out.append((e,c,d+cut+.025,d))
  return out
 results={}
 for n in range(1,10):
  pattern=f'{10**(n-1):,}'
  @lru_cache(None)
  def solve(x,i):
   while x<r and not m[:,x].any():x+=1
   if i==len(pattern):return [(0.,'',0.)] if x==r else []
   if x==r:return []
   rem=len(pattern)-i
   if r-x<rem or r-x>rem*round(h*.67)+3:return []
   options=[]
   for e,c,cost,d in choices(x):
    if (pattern[i]==',')!=(c==',') or (i==0 and c=='0'):continue
    for v,s,mx in solve(e,i+1):options.append((v+cost,c+s,max(mx,d)))
   unique={}
   for v,s,mx in sorted(options):
    if s not in unique:unique[s]=(v,s,mx)
    if len(unique)>=2:break
   return list(unique.values())
  for cost,s,mx in solve(l,0):results[s]=(cost,s,mx)
 if not results:return None
 ranked=sorted(results.values());cost,s,mx=ranked[0];margin=ranked[1][0]-cost if len(ranked)>1 else 1.
 return {'text':s,'value':int(s.replace(',','')),'cost':cost/len(s),'margin':margin,'max_glyph':mx}

def review(image,bank):
 """Conservative candidate only, not an authorization to add to the ledger."""
 results=[decode(image,bank,t) for t in (.46,.52,.60)]
 passing=[x for x in results if x and x['cost']<=.19 and x['margin']>=.025 and x['max_glyph']<=.23]
 counts=collections.Counter(x['value'] for x in passing)
 if not counts:return None,results
 value,n=counts.most_common(1)[0]
 # Disagreement even among weak but nonempty hypotheses is reason to withhold.
 if n<2 or any(x and x['value']!=value for x in results):return None,results
 return value,results

def load_bank(path):
 with np.load(path,allow_pickle=False) as data:
  return {(',' if c=='comma' else c):data[c] for c in data.files}

def raid_review(image,bank,label):
 """Explicit *observed* raid profile, not a universal currency range.
 Unseen digit lengths and ambiguity are deferred for review, not set to zero.
 Still not safe for automatic ledger writes: complete-prefix occlusion can look valid.
 """
 if label not in ('gold','gem'):return None,[]
 value,results=review(image,bank)
 return profile_guard(image,label,value),results

def profile_guard(image,label,value):
 if label not in ('gold','gem'):return None
 allowed={'gold':(6,7),'gem':(3,)}[label]
 if value is None or len(str(value)) not in allowed:return None
 if label=='gem':
  prep=preprocess(image,.52)
  if prep is None:return None
  _,mask=prep; xs=np.flatnonzero(mask.any(0)); gaps=np.diff(xs)-1
  if len(gaps) and max(gaps)/mask.shape[0]>.18:return None
 return value
