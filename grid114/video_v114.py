"""Offline or shared-frame consumer. No capture thread or ledger mutations."""
import argparse, collections, hashlib, json, subprocess, time
from pathlib import Path
import cv2
import numpy as np
import torch
from .result_grid_replay import grid
from .currency_temporal_v11_3 import Capture, geometry, digest, select
from .currency_quality_v11_2 import hybrid_review
from .currency_ctc_v11 import CurrencyCTC
from .currency_v7 import load_bank,preprocess
from .train_icon_v114 import icon, scores
from .panel_v114 import match
from .perks import read_perks

ROOT=Path(__file__).resolve().parent

class Reader:
    def __init__(self):
        torch.set_num_threads(1)
        self.gate=dict(np.load(ROOT/'v114/icon_candidate.npz'))
        self.panel=dict(np.load(ROOT/'v114/panel_candidate.npz'))
        self.recent=collections.OrderedDict()
        self.models=[]
        for name in ('currency_v11_ctc/currency_ctc_state.pt','currency_v11_1/currency_state.pt'):
            model=CurrencyCTC()
            model.load_state_dict(torch.load(ROOT/'results'/name,weights_only=True))
            self.models.append(model.eval())
        self.banks=[load_bank(ROOT/'results'/p) for p in ('currency_v7/glyph_bank.npz','currency_v8_friend/blur_bank.npz')]

    def classify(self,tile):
        s=scores(icon(tile),self.gate['refs'])
        logits=self.gate['coef']@s+self.gate['intercept']
        p=np.exp(logits-logits.max());p/=p.sum()
        k=p.argmax()
        return str(self.gate['classes'][k]),float(p[k]),float(max(s))

    def read(self,crop,label):
        key=(label,digest(crop))
        if key not in self.recent:
            self.recent[key]=hybrid_review(crop,label,*self.banks,*self.models)
            if len(self.recent)>128:self.recent.popitem(last=False)
        self.recent.move_to_end(key)
        return self.recent[key]

    def quantity(self,frame,box,label,text_bounds=None):
        x,y,w,h=box
        left,right=text_bounds if text_bounds is not None else (x+2,x+w-2)
        if not x<=left<right<=x+w:raise ValueError('text bounds must lie in tile')
        crop=frame[y+round(.67*h):y+h,left:right]
        g=geometry(crop)
        def rejected():
            return dict(candidate=None,method='quality_guard',review_only=True,ledger_allowed=False)
        # Only boundary suspicion is recoverable by expanding the actual ROI.
        # Never spend inference time on already-rejected render/invalid crops.
        if any(flag!='text_near_crop_edge' for flag in g['flags']):
            return crop,g,rejected()
        value=self.read(crop,label)
        # A caller-supplied sub-ROI can omit a complete digit without touching the edge.
        if left>x+2 or right<x+w-2:
            full=frame[y+round(.67*h):y+h,x+2:x+w-2]
            full_g=geometry(full)
            complete=self.read(full,label) if not full_g['flags'] else rejected()
            partial=value['candidate']; recovered=complete['candidate']
            agrees=partial is None or partial==recovered or (
                left>x+2 and recovered is not None and str(recovered).endswith(str(partial)))
            if recovered is not None and agrees and not full_g['flags']:
                return full,dict(full_g,expanded=True),complete
            g['flags'].append('unverified_subroi')
            return crop,g,rejected()
        if g['flags']==['text_near_crop_edge']:
            # Read actual surrounding pixels. Reject disagreement or boundary clipping.
            left=max(0,x-round(.05*w))
            wide=frame[y+round(.67*h):y+h,left:min(frame.shape[1],x+w+2)]
            wide_g=geometry(wide)
            if not wide_g['flags']:
                wide_r=self.read(wide,label)
                if wide_r['candidate'] is not None and value['candidate'] in (None,wide_r['candidate']):
                    crop,g,value=wide,wide_g,wide_r
                    g['expanded']=True
        if g['flags']:
            # A rejected candidate must never enter temporal consensus as a number.
            value=rejected()
        return crop,g,value

def finish_event(rows, terminal_frame=None):
    results=[]
    if not rows:
        return results
    terminal_frame = rows[-1]['frame'] if terminal_frame is None else terminal_frame
    for slot in sorted({r['slot'] for r in rows}):
        seq=[r for r in rows if r['slot']==slot]
        last=seq[-1]; key=(last['event'],slot,last['label'])
        captures=[Capture(*key,r['stream'],r['t'],str(r['frame']),r['hash'],r['value'],tuple(r['flags']),r['sharpness']) for r in seq]
        result=select(captures)
        # The watchdog can sample a stable reward screen less than once per second.
        # Use the actual last three observations; the session already splits at 2 s.
        tail = seq[-3:]
        times = [r['t'] for r in tail]
        reason = None
        if len({r['label'] for r in seq}) != 1:
            reason = 'slot_changed_label'
        elif len(tail) < 3 or times[-1] - times[0] < .3999:
            # At higher frame rates include the rest of the former terminal window.
            tail = [r for r in seq if last['t'] - r['t'] <= .65]
            times = [r['t'] for r in tail]
            if len(tail) < 3 or times[-1] - times[0] < .3999:
                reason = 'short_terminal_history'
        if reason is None:
            if last['frame'] != terminal_frame:
                reason = 'slot_absent_on_last_frame'
            elif any(not 0 < b - a < 2 for a, b in zip(times, times[1:])):
                reason = 'discontinuous_samples'
            elif any(r['flags'] or r['value'] is None for r in tail):
                reason = 'unclear_terminal_read'
        if reason:
            result.update(consensus_status=result['status'], status='uncertain_terminal_panel', candidate=None, best_frame=None,
                          terminal_reason=reason)
        if result['candidate'] is not None:
            clear=[r for r in tail if r['value']==result['candidate']]
            # Rank only fully eligible terminal observations. Sharpness normalized within this window.
            peak=max(r['sharpness'] for r in clear)
            def rank(r):
                confidence=(r['icon_conf']+r.get('read_stability',1.))/2
                neighborhood=[v for v in seq if abs(v['t']-r['t'])<=.41]
                agreement=sum(v['value']==r['value'] and not v['flags'] for v in neighborhood)/len(neighborhood)
                return (.30*confidence+.25*r.get('edge_clearance',float(r['span']>0))
                        +.20*r['sharpness']/max(peak,1e-8)+.15*agreement+.10*r['roi_quality'])
            best=max(clear,key=rank);result.update(best_frame=best['frame'],score=rank(best))
        results.append(dict(event=key[0],slot=slot,label=key[2],**result))
    return results

class Session:
    def __init__(self,reader,source):
        self.reader=reader;self.source=source;self.seq=0;self.last_seen=None;self.pending=[];self.events=[];self.last_t=-1;self.last_panel_frame=None

    def flush(self):
        if self.pending:self.events.extend(finish_event(self.pending,self.last_panel_frame))
        self.pending=[];self.last_seen=None;self.last_panel_frame=None

    def observe(self,frame,index,t):
        if t<=self.last_t:raise ValueError('timestamps must increase in a session')
        self.last_t=t
        if self.last_seen is not None and t-self.last_seen>=2:self.flush()
        self.perk_slots = {}
        rows=[]; tiles=[]
        header=match(frame,self.reader.panel['tpl'])
        if self.reader.panel['slope']*header+self.reader.panel['bias']<0:
            return rows,tiles
        self.last_panel_frame=index
        located=grid(frame)
        self.perk_slots = read_perks(frame, located, self.reader.banks[0])
        for row in located:
            if len(row)<3:continue
            for x,y,w,h in row:
                tile=frame[y:y+h,x:x+w]
                label,conf,corr=self.reader.classify(tile)
                if label=='other' or conf<.75 or corr<.70:continue
                if self.last_seen is None:self.seq+=1
                self.last_seen=t
                crop,g,value=self.reader.quantity(frame,(x,y,w,h),label)
                # Geometric slot key is independent of the number of other detected tiles.
                slot=f'{round((x+w/2)/frame.shape[1]*32)}-{round((y+h/2)/frame.shape[0]*24)}'
                rows.append(dict(stream=self.source,event=f'{self.source}:{self.seq}',slot=slot,frame=index,t=t,label=label,
                    box=[x,y,w,h],icon_conf=conf,panel_correlation=header,correlation=corr,value=value['candidate'],method=value['method'],hash=digest(crop),
                    roi_quality=min(1.,len(row)/5),**g))
                # Read a slight contrast perturbation only after localization, to rank
                # threshold-sensitive digits lower. A disagreement never changes a digit.
                if g['flags'] or value['candidate'] is None:
                    rows[-1].update(read_stability=0.,edge_clearance=0.)
                    tiles.append(crop)
                    continue
                alternate=cv2.convertScaleAbs(crop,alpha=.94)
                check=self.reader.read(alternate,label)
                same=value['candidate'] is not None and value['candidate']==check['candidate']
                prep=preprocess(crop,.46)
                xs=np.flatnonzero(prep[1][:round(.77*len(crop))].any(0)) if prep is not None else []
                edge=min(int(xs[0]),crop.shape[1]-1-int(xs[-1])) if len(xs) else 0
                rows[-1].update(read_stability=float(same),edge_clearance=min(1.,edge/max(1,.22*len(crop))))
                if not same:rows[-1]['flags']=[*rows[-1]['flags'],'unstable_digit_read']
                tiles.append(crop)
        self.pending.extend(rows)
        return rows,tiles
