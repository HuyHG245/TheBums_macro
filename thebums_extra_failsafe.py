"""Time-only level-2 watchdog and verified Roblox Leave sequence.
No AI, no fixed click locations, no background input thread.
"""
from dataclasses import dataclass
import math
import time
import cv2
import numpy as np

# Crops from the supplied 1918x1078 demonstration, white foreground only.
# Relative search regions are deliberately wider than the demo positions.
SPECS = {
    'menu': ('fs_roblox_menu.png', (0., 0., .065, .15), .87),
    'hamburger': ('fs_menu_hamburger.png', (.02, 0., .14, .15), .88),
    'leave': ('fs_menu_leave.png', (.12, .48, .52, .95), .88),
    'resume': ('fs_menu_resume.png', (.48, .48, .89, .95), .88),
    'title': ('fs_confirm_dialog.png', (.12, .16, .88, .65), .88),
    'confirm': ('fs_confirm_leave.png', (.16, .25, .51, .75), .88),
    'cancel': ('fs_confirm_cancel.png', (.50, .25, .84, .75), .88),
}
SCALES = (.75,.8,.85,.9,.95,1.,1.025,1.05,1.07,1.10,1.15,1.20,1.25)

@dataclass(frozen=True)
class Hit:
    x: int
    y: int
    score: float
    width: int
    height: int

class LeaveError(RuntimeError):
    pass


def level2_due(started, timeout, now, *, active=True, blocked=False):
    """Deadline anchored to the same mission start as level 1; no rescue reset."""
    return (active and not blocked and started is not None and timeout is not None
            and math.isfinite(timeout) and timeout > 0 and now - started >= 2 * timeout)


def white_mask(frame):
    # White/gray UI text, independent of game scenery behind the translucent modal.
    b,g,r=cv2.split(frame)
    lo=cv2.min(cv2.min(b,g),r)
    hi=cv2.max(cv2.max(b,g),r)
    lum=np.clip((lo.astype(np.float32)-115)/115,0,1)
    chroma=np.clip(1-(hi.astype(np.float32)-lo)/65,0,1)
    return (255*lum*chroma).astype(np.uint8)


class LeaveMatcher:
    def __init__(self, asset_path):
        self.templates = {}
        for key, (name, _, _) in SPECS.items():
            im = cv2.imread(str(asset_path(name)), cv2.IMREAD_COLOR)
            if im is None:
                raise LeaveError('Missing Leave template: '+name)
            mask = white_mask(im)
            if np.count_nonzero(mask) < 15 or np.all(mask == 255):
                raise LeaveError('Invalid Leave template: '+name)
            self.templates[key] = [(scale, white_mask(cv2.resize(im, None, fx=scale, fy=scale,
                                                     interpolation=cv2.INTER_CUBIC))) for scale in SCALES]

    def find(self, frame, key, mask=None):
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return None
        h,w=frame.shape[:2]
        _,(x0,y0,x1,y1),minimum=SPECS[key]
        xa,ya,xb,yb=int(w*x0),int(h*y0),int(w*x1),int(h*y1)
        roi = (white_mask(frame[ya:yb,xa:xb]) if mask is None else mask[ya:yb,xa:xb])
        best = None
        for _,tpl in self.templates[key]:
            th,tw=tpl.shape
            if th>roi.shape[0] or tw>roi.shape[1]:continue
            _,score,_,(x,y)=cv2.minMaxLoc(cv2.matchTemplate(roi,tpl,cv2.TM_CCOEFF_NORMED))
            if score>=minimum and (best is None or score>best.score):
                best=Hit(xa+x+tw//2,ya+y+th//2,float(score),tw,th)
        return best

    def identify(self, frame):
        """Prefer specific states. Leave/Confirm require TWO distinct controls."""
        mask=white_mask(frame)
        title=self.find(frame,'title',mask)
        if title:
            left=self.find(frame,'confirm',mask);right=self.find(frame,'cancel',mask)
            if left and right and left.x < right.x and abs(left.y-right.y)<max(left.height,right.height)*1.5 and left.y>title.y:
                # Blue confirmation button, not the white word Leave in the cancel button.
                h,w=frame.shape[:2];dy=max(6,left.height//2);dx=max(10,left.width//2)
                crop=frame[max(0,left.y-dy):min(h,left.y+dy),max(0,left.x-dx):min(w,left.x+dx)]
                b,g,r=cv2.split(crop.astype(np.int16))
                if crop.size and np.mean((b>r+25)&(b>g+15))>.30:
                    return 'confirm',left
            return None,None # never fall back to menu when a modal is visible
        leave=self.find(frame,'leave',mask);resume=self.find(frame,'resume',mask)
        if leave and resume and leave.x<resume.x and abs(leave.y-resume.y)<max(leave.height,resume.height)*1.5:
            return 'leave',leave
        menu=self.find(frame,'menu',mask)
        hamburger=self.find(frame,'hamburger',mask)
        if (menu and hamburger and 25 <= hamburger.x-menu.x <= 100
                and abs(menu.y-hamburger.y)<=20):
            return 'menu',menu
        return None,None


def same(a,b):
    return a is not None and b is not None and abs(a.x-b.x)<=15 and abs(a.y-b.y)<=15


class LeaveNavigator:
    """Each stage is stable over two captures and freshly checked before input.
    A clicked stage must advance; a stale screen never causes repeated clicks.
    """
    def __init__(self, matcher, capture, click, check, handoff, progress=lambda _:None,
                 clock=time.monotonic, sleep=time.sleep, step_timeout=15., exit_timeout=65.):
        self.matcher,self.capture,self.click,self.check=matcher,capture,click,check
        self.handoff,self.progress,self.clock,self.sleep=handoff,progress,clock,sleep
        self.step_timeout,self.exit_timeout=step_timeout,exit_timeout

    def observe(self):
        self.check()
        frame=self.capture()
        stage,hit=self.matcher.identify(frame)
        if stage is not None:return stage,hit
        if self.handoff(frame):return 'handoff',None
        return None,None

    def stable(self, allowed, timeout):
        end=self.clock()+timeout;prev=None
        while self.clock()<end:
            stage,hit=self.observe()
            if stage in allowed and prev and prev[0]==stage and (stage=='handoff' or same(prev[1],hit)):
                # Fresh third observation immediately before input/handoff.
                fresh,fresh_hit=self.observe()
                if fresh==stage and (stage=='handoff' or same(hit,fresh_hit)):
                    return fresh,fresh_hit
            prev=(stage,hit) if stage in allowed else None
            self.sleep(.25)
        raise LeaveError('No verified next state: '+', '.join(allowed))

    def run(self):
        order=('menu','leave','confirm','handoff')
        allowed=order
        for _ in range(4):
            stage,hit=self.stable(allowed,self.exit_timeout if allowed==('handoff',) else self.step_timeout)
            if stage=='handoff':
                self.progress('Roblox exit/recovery screen verified')
                return True
            self.progress('Click verified '+stage)
            fresh,fresh_hit=self.observe()
            if fresh=='handoff':
                allowed=('handoff',)
                continue
            if fresh!=stage or not same(hit,fresh_hit):
                raise LeaveError('Target changed before click: '+stage)
            self.click(fresh_hit)
            allowed=order[order.index(stage)+1:]
        raise LeaveError('Leave sequence did not complete')
