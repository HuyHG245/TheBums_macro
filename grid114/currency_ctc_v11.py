"""Grid v11 CPU-only CTC experiment. Review only; no game or ledger writes."""
import argparse, csv, random, json
from pathlib import Path
import numpy as np
import cv2
import torch
from torch import nn
from .currency_v7 import load_bank, preprocess

CHARS="0123456789,"
W,H=96,32
def tensorize(g):
    return cv2.resize(g.astype(np.float32),(W,H),interpolation=cv2.INTER_LINEAR)[None]
def raw_tensor(im,t=.46):
    p=preprocess(im,t)
    if p is None:return np.zeros((1,H,W),np.float32)
    return tensorize(p[0])
class CurrencyCTC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),
            nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
            nn.MaxPool2d((2,1)),nn.Conv2d(32,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU(),
            nn.MaxPool2d((2,1)),nn.Conv2d(48,64,(4,3),padding=(0,1)),nn.ReLU())
        self.head=nn.Sequential(nn.Conv1d(64,64,3,padding=1),nn.ReLU(),nn.Conv1d(64,12,1))
    def forward(self,x):
        return self.head(self.conv(x).squeeze(2)).permute(2,0,1).log_softmax(2)

def synth(bank,rng):
    # Source glyphs from development-only bank; generate arbitrary integer strings.
    n=int(rng.choice([1,2,3,4,5,6,7,8],p=[.025,.055,.24,.06,.16,.27,.16,.03]))
    val=int(rng.integers(10**(n-1),10**n))
    text=f"{val:,}"
    h,w=23,66
    height=float(rng.uniform(8,11))
    if n>=7:height*=.85
    base=int(rng.integers(14,18))
    patches=[]
    for c in text:
        rows=bank[c]
        p=rows[int(rng.integers(len(rows)))]
        gh=max(2,round(height*(.33 if c==',' else rng.uniform(.88,1.06))))
        gw=max(1,round(gh*np.exp(p[-3])*rng.uniform(.88,1.12)))
        patch=cv2.resize(p[:216].reshape(18,12).astype(np.float32),(gw,gh),interpolation=cv2.INTER_LINEAR)
        gap=int(rng.choice([0,1,1,2]))
        patches.append((c,patch,gap))
    width=sum(p.shape[1]+gap for _,p,gap in patches)-patches[-1][2]
    canvas=np.zeros((h,max(w,width+6)),np.float32)
    x=canvas.shape[1]-int(rng.integers(2,5))-width
    for c,p,gap in patches:
        ph,pw=p.shape
        y=base-ph+(1 if c==',' else 0)
        canvas[y:y+ph,x:x+pw]=np.maximum(canvas[y:y+ph,x:x+pw],p)
        x+=pw+gap
    # compress wider strings instead of clipping digits
    if canvas.shape[1]!=w:canvas=cv2.resize(canvas,(w,h),interpolation=cv2.INTER_AREA)
    if rng.random()<.72:
        canvas=cv2.GaussianBlur(canvas,(3,3),float(rng.uniform(.25,.95)))
    canvas=np.clip(canvas*rng.uniform(.75,1.35)+rng.normal(0,.014,canvas.shape),0,1)
    return tensorize(canvas),text
def real_sample(im,text,rng):
    a=im.copy()
    if rng.random()<.6:a=cv2.GaussianBlur(a,(3,3),float(rng.uniform(.3,.8)))
    a=np.clip(a.astype(np.float32)*rng.uniform(.83,1.15),0,255).astype(np.uint8)
    return raw_tensor(a,float(rng.choice([.40,.46,.52,.60]))),text
def collate(samples):
    xs,ss=zip(*samples)
    x=torch.from_numpy(np.stack(xs)).float()
    target=torch.tensor([CHARS.index(c)+1 for s in ss for c in s],dtype=torch.long)
    lengths=torch.tensor([len(s) for s in ss],dtype=torch.long)
    return x,target,lengths,ss
def greedy(logp):
    paths=logp.argmax(2).cpu().numpy().T
    return [''.join(CHARS[c-1] for j,c in enumerate(path) if c and (j==0 or c!=path[j-1])) for path in paths]
def train_steps(model,optimizer,bank,real,rng,steps=100,batch_size=64):
    model.train()
    loss_fn=nn.CTCLoss(blank=0,zero_infinity=True)
    losses=[]
    for i in range(steps):
        samples=[synth(bank,rng) if rng.random()<.78 else real_sample(*real[int(rng.integers(len(real)))],rng) for _ in range(batch_size)]
        x,target,lens,ss=collate(samples)
        logits=model(x)
        loss=loss_fn(logits,target,torch.full((len(ss),),logits.shape[0],dtype=torch.long),lens)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))
@torch.no_grad()
def evaluate(model,samples):
    model.eval()
    outs=[]
    for start in range(0,len(samples),64):
        x,_,_,ss=collate(samples[start:start+64])
        pred=greedy(model(x))
        outs.extend(zip(ss,pred))
    return outs

def canonical_value(text):
    import re
    if not re.fullmatch(r'(?:[1-9][0-9]{0,2})(?:,[0-9]{3})*|0',text):
        return None
    return int(text.replace(',',''))

@torch.no_grad()
def review_candidate_ctc(model,image):
    """Only a candidate; never a ledger decision. Occlusion is not solved."""
    model.eval()
    texts=[greedy(model(torch.from_numpy(raw_tensor(image,t)[None])))[0] for t in (.40,.46,.52)]
    values=[canonical_value(t) for t in texts]
    return {'candidate':values[0] if values[0] is not None and len(set(values))==1 else None,
            'texts':texts,'review_only':True,'ledger_allowed':False}
