"""V11.1 render-degradation fine-tuning, offline review only."""
import cv2, numpy as np, torch
from torch import nn
from . import currency_ctc_v11 as base

def degrade(a,rng,strong=True):
    h,w=a.shape[:2]
    scale=float(rng.uniform(.48,.78) if strong else rng.uniform(.78,1.0))
    interp=int(rng.choice([cv2.INTER_LINEAR,cv2.INTER_AREA]))
    b=cv2.resize(a,(max(4,round(w*scale)),max(4,round(h*scale))),interpolation=interp)
    b=cv2.resize(b,(w,h),interpolation=int(rng.choice([cv2.INTER_LINEAR,cv2.INTER_CUBIC])))
    if rng.random()<.5:
        b=cv2.GaussianBlur(b,(3,3),float(rng.uniform(.25,.65)))
    return b

def raw_tensor(a,t=.46):
    # Ignore colored rarity stripe instead of letting it affect brightness normalization.
    b=a.copy()
    lo=b.min(2).astype(np.int16);hi=b.max(2).astype(np.int16)
    b[(hi-lo)>55]=0
    return base.raw_tensor(b,t)

def sample_real(im,text,rng):
    b=degrade(im,rng) if rng.random()<.68 else im.copy()
    b=np.clip(b.astype(np.float32)*rng.uniform(.72,1.18)+rng.normal(0,1.4,b.shape),0,255).astype(np.uint8)
    return raw_tensor(b,float(rng.choice([.38,.42,.46,.52]))),text

def sample_synth(bank,rng):
    x,s=base.synth(bank,rng)
    if rng.random()<.6:
        x=degrade(x[0],rng)[None]
    return x.astype(np.float32),s

def train_steps(model,opt,bank,real,rng,steps=100,batch_size=48):
    model.train()
    loss_fn=nn.CTCLoss(blank=0,zero_infinity=True)
    losses=[]
    for _ in range(steps):
        samples=[sample_synth(bank,rng) if rng.random()<.64 else sample_real(*real[int(rng.integers(len(real)))],rng) for _ in range(batch_size)]
        x,target,lens,ss=base.collate(samples)
        logp=model(x)
        loss=loss_fn(logp,target,torch.full((len(ss),),logp.shape[0],dtype=torch.long),lens)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))

@torch.no_grad()
def predict(model,a,threshold=.46):
    model.eval()
    return base.greedy(model(torch.from_numpy(raw_tensor(a,threshold)[None])))[0]

@torch.no_grad()
def review(model,a):
    texts=[predict(model,a,t) for t in (.40,.46,.52)]
    vals=[base.canonical_value(s) for s in texts]
    return {'candidate':vals[0] if vals[0] is not None and len(set(vals))==1 else None,
            'texts':texts,'review_only':True,'ledger_allowed':False}
