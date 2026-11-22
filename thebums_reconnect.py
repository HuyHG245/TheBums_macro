"""Image-driven reconnect state machine; *only* the macro owner supplies UI input.

This module deliberately does not import tkinter or Roblox-specific APIs. Matching
is deterministic and testable with previously captured screenshots. Every action
has an independently observed gate before the next action is permitted.
"""
from __future__ import annotations

import dataclasses
import os
import time
from typing import Callable, Optional

import cv2
import numpy as np


class ReconnectError(RuntimeError):
    pass


class NewDisconnect(ReconnectError):
    pass


@dataclasses.dataclass(frozen=True)
class Match:
    name: str
    x: int
    y: int
    confidence: float
    width: int
    height: int
    source: str = "opencv"


# ROIs are fractions of the frame dimensions, not screen pixels. They are
# restricted by the full screenshots the user provided, to reject duplicate
# labels on different panels (e.g. Redeem vs VIP Servers or two START buttons).
REGIONS = {
    'slot_a': (.20,.20,.43,.45),
    'select': (.18,.52,.45,.72),
    'play_small': (.34,.46,.65,.65),
    'codes': (.32,.66,.67,.86),
    'lobby': (.30,.35,.70,.78),
    'ps_input': (.01,.76,.17,.98),
    'generate': (.01,.83,.17,.99),
    'play_large': (.72,.62,.99,.94),
    'missions': (.10,.24,.40,.86),
    'outskirts': (.58,.47,.78,.74),
    'outskirts_selected': (.28,.25,.60,.45),
    'create': (.40,.71,.66,.83),
    'start': (.46,.70,.69,.83),
}
THRESHOLDS = {name:.82 for name in REGIONS}
THRESHOLDS.update({'reconnect':.90, 'slot_a':.86, 'select':.85,
                   'start':.88, 'generate':.85, 'ps_input':.87})


# The English Roblox disconnect/Home UI anchors below were extracted from the
# user-supplied recording (22 Nov 2026). They are compact binary glyph masks
# held inline so that all original recognition images stay byte-for-byte intact.
# OpenCV alone handles this navigation; there is no OCR, API or remote request.
# If Roblox changes these labels the route MUST fail closed (no blind click).
_CLIENT_UI_B64 = {
    'disconnect_title': 'iVBORw0KGgoAAAANSUhEUgAAAI8AAAAhCAAAAADFHqfRAAAA80lEQVRYw+1V7Q7DIAjsNb7/K9/SxSIofixZarJxP5ozICBfPY5A4B/AJ3w0TlIbAjInvvm4RWNnL2jsqUqyR5TM7AnIyQ92dm3yy4275FQNlVkhMPw6QBqxsOOtxjv5rU43GRTXkE9W4q1OXdghV7fYXGanGGl5LKCIDJ+yXHmhvgUxBCfcSf94W4jeSrCvg33ElQy6zysSrPaP+KCkCB+0O7uzwdnUnOPNDHhL9PH5gqaUkbNkkFYzqGuSwX52aFUtjqPxhoeOhNP81H1WxwZNJg2EmqGVLO1nWFo+lyEhgwSJSnVLS7b/CQKBQCAQ+F28ABtyVUI/bHCmAAAAAElFTkSuQmCC',
    'leave_button': 'iVBORw0KGgoAAAANSUhEUgAAAFMAAAAkCAAAAADQ/lmlAAAAfUlEQVRIx+2U3Q6AIAhG/Vrv/8qnmWKu1irsyuEFIIMz+Zki/X6WFMxgBjOY0uy1ry9ichMwpWwWWf0OpqVnLUFpMwfWUzuVTDMLyi6e2tUb4up3MGkIdmFtZGDuOsFtczWyS2r5uvGf4nl4Hv0uUYdNv2MfmPHXBTOYszE3syQgV0TkLe8AAAAASUVORK5CYII=',
    'home_game_title': 'iVBORw0KGgoAAAANSUhEUgAAAJoAAAAqCAAAAABuXh2+AAABbElEQVRYw+1X0ZKDIAxknf7/L+9ZQ4EEFLRwN3OTfUBLQ1hCAmsIDodjIcivhr+0sxBwPIu2mjCgwyg+xYyYQ20K8KGPMC9qu0fKsuWNMk0KQJotRy72iT2q+FonTbsTbPWC44qRfheN3iyZl8H+qxxqJyd2vailETgSJLbZHyrDdx/IkoDlpp2c2g3lWpW3NlIlSVnCSLVOqNBWQfLrY+SJk02N39FKF5j9Lnd0LCB4UK9b7YQqZfdGjk6kt8zt6OtMqpw8pYaSH2O7NzGUaLBAl5lx4nA4HP9cyQ7eobQHb0fgcvEh+qrl2l1J+yvUDoXDpCeJkEVqKR4L/cooXm9r2AfXe6lheKEvoLTufQ17+9sg3dhRu+YbnDa1QB3IK607J2qilZm/M0yEzhIOq8vAVCn/UslsLSFpj4teBs3MsOsySHoWTS3LDw2Gca07gRrkIKBKvVLLIm0z0lWAFcwcDodjgXBdKGEH8AOBXKgkUyC3lAAAAABJRU5ErkJggg==',
    'game_detail_title': 'iVBORw0KGgoAAAANSUhEUgAAAawAAAAvCAAAAACa8KGyAAACRElEQVR42u2Z63bjIAyEPT55/1eenr2cBoME4ua4u/P9aFOXKCOBJCDHIYQQQgghhBBCCCGEEOLRIH9A62EnnDHBQuCEOfqOrnSG8zGL8LK8IywVNynatwbwIDlDnP9RFeFPd+Blu/XBFAKbpfr5zeSWyeJTPOe6EODbNeJfyyy8Z4zfP3B5/Z5W5NOM6maFefhwsVltDswscsRU8c7MIDxXkg5lyKGtqk9Wb89iNMPI6++DzhutuboM4lhGFwLipox3Fkk9WmUs21xWs8Y2GPQf0B9lDOJgZOhPX8sUncRnzZVuWdMehsogkhaPJL/T179GIO0DSQWBWwSTmpF0D0T9qApom0o/HVlgkRkoVTbkrPGwL7MY3mkAeZFD+tzbIqCojBis56UA1xR/Y5VmVnYkgz1mnYe7zll0HqF5mbBwe4bQiEIFsm4zIW23h2UZ5PD8dA0g1i+Q6GQyPN0zKjecE16e/9XP+vhxjH2zxNvjeuMNRtfZ9eOH59ityLXj5zcm3CNvLefI4g1cL6CxB8J0YuGG6D4s484sAn9ZsIjy2UJkGezMQvoz8t4s1lS2TrfbPTxDq9gRYVwfsH3vwMmUcgT0GiHdUNdUsl59uHM1vuzSkhZ5Zhtc/Plv0RPyZo3s6hRJDGa+CrQExLuWKfVoq7Q/8lqP0zG4P7NQLXYwejYihRDDtcwTMFMI4fyFymm81eO2dLvzGJktFMdyIBA9dIfYjcXEvQBMGWipRGD1TnsohBBCCCGEEEIIIYQQQoh7+ALC6uwzbzcgDwAAAABJRU5ErkJggg==',
}
_CLIENT_UI_CACHE = {}


def _ui_template(name):
    if name not in _CLIENT_UI_CACHE:
        from base64 import b64decode
        encoded = np.frombuffer(b64decode(_CLIENT_UI_B64[name]), dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if decoded is None or decoded.std() < 5:
            raise ReconnectError('Embedded Roblox UI anchor could not be decoded')
        _CLIENT_UI_CACHE[name] = decoded
    return _CLIENT_UI_CACHE[name]


def _find_ui_text(frame, name, roi, threshold=.82):
    """Match an anchored, locally embedded UI label with bounded scaling."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
        return None
    h, w = frame.shape[:2]
    if h < 480 or w < 800:
        return None
    x0,y0,x1,y1 = roi
    xa,ya,xb,yb = max(0,int(w*x0)),max(0,int(h*y0)),min(w,int(w*x1)),min(h,int(h*y1))
    if xb<=xa or yb<=ya:
        return None
    image = cv2.cvtColor(frame[ya:yb,xa:xb,:3], cv2.COLOR_BGR2GRAY)
    image = cv2.threshold(image, 185, 255, cv2.THRESH_BINARY)[1]
    original = _ui_template(name)
    best=None
    for scale in (1.0, .9, 1.1, .8, 1.2):
        tw=max(8,round(original.shape[1]*scale)); th=max(8,round(original.shape[0]*scale))
        if image.shape[0]<th or image.shape[1]<tw:
            continue
        tpl = original if scale==1.0 else cv2.resize(
            original,(tw,th),interpolation=cv2.INTER_NEAREST)
        # A nearly uniform patch must never be taken as a positive match.
        if tpl.std()<5:
            continue
        _,score,_,at = cv2.minMaxLoc(cv2.matchTemplate(
            image,tpl,cv2.TM_CCOEFF_NORMED))
        if best is None or score>best.confidence:
            best=Match(name,xa+at[0]+tw//2,ya+at[1]+th//2,
                       float(score),tw,th)
    return best if best is not None and best.confidence>=threshold else None


def find_leave_disconnect(frame):
    """Distinct Roblox Leave-only disconnect dialog; require *both* text labels.

    This never mistakes the original 'Reconnect' button for a Leave dialog.
    The result points to the white Leave button, not the central text.
    """
    title=_find_ui_text(frame,'disconnect_title',(.33,.30,.68,.53),.85)
    if title is None:
        return None
    leave=_find_ui_text(frame,'leave_button',(.32,.49,.69,.73),.86)
    if leave is None:
        return None
    h,w=frame.shape[:2]
    dy=leave.y-title.y
    if not (.12*h < dy < .23*h) or abs(leave.x-title.x)>.055*w:
        return None
    # Confirm the target really lies within a pale Leave-button surface.
    y,x=leave.y,leave.x
    patch=frame[max(0,y-6):y+7,max(0,x-35):x+36,:3]
    if patch.size==0 or np.mean(np.all(patch>185,axis=2))<.60:
        return None
    return Match('leave',leave.x,leave.y,min(title.confidence,leave.confidence),
                 leave.width,leave.height,'opencv')


def find_roblox_home_game(frame):
    """Locate the AOT:R card text within Roblox Home / Continue."""
    label=_find_ui_text(frame,'home_game_title',(.02,.20,.97,.97),.84)
    if not label:
        return None
    # Move upwards from the text into the matching game's thumbnail.
    cy=label.y-round(95*label.height/42)
    if cy < 20:
        return None
    return Match('home_game',label.x,cy,label.confidence,label.width,label.height)


def find_roblox_game_play(frame):
    """Find the BLUE Play button with the WHITE right-pointing triangle.

    This is the button on the Attack on Titan Revolution detail page in the
    user's recording, *not* a generic PLAY text/icon elsewhere in Roblox.
    A verified game title, a blue button, AND its white triangle are required.
    """
    title = _find_ui_text(frame, 'game_detail_title', (.12, .07, .90, .70), .82)
    if title is None:
        return None

    h, w = frame.shape[:2]
    expected_x = round(title.x - .21 * title.width)
    expected_y = round(title.y + 2.05 * title.height)
    xa = max(0, expected_x - round(.42 * title.width))
    xb = min(w, expected_x + round(.42 * title.width))
    ya = max(0, expected_y - round(.90 * title.height))
    yb = min(h, expected_y + round(.90 * title.height))
    if xa >= xb or ya >= yb:
        return None

    crop = frame[ya:yb, xa:xb, :3]
    # Roblox's bright blue Play button: HSV blue, not gray loading skeletons.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (100, 110, 115), (130, 255, 255))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE,
                            np.ones((3, 3), dtype=np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(blue, 8)
    options = sorted(range(1, count), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]),
                     reverse=True)
    for i in options:
        x, y, bw, bh, area = map(int, stats[i])
        px, py = xa + x + bw // 2, ya + y + bh // 2
        if not (.35 * title.width <= bw <= .85 * title.width):
            continue
        if not (.65 * title.height <= bh <= 1.65 * title.height):
            continue
        if abs(px - expected_x) > .23 * title.width:
            continue
        if abs(py - expected_y) > .65 * title.height:
            continue
        if area / max(bw * bh, 1) < .65:
            continue

        button = crop[y:y + bh, x:x + bw]
        white = cv2.inRange(button, (210, 210, 210), (255, 255, 255))
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            tx, ty, tw, th = cv2.boundingRect(contour)
            if not (.045 * bw <= tw <= .19 * bw):
                continue
            if not (.25 * bh <= th <= .78 * bh):
                continue
            if abs((tx + tw / 2) - bw / 2) > .16 * bw:
                continue
            if abs((ty + th / 2) - bh / 2) > .20 * bh:
                continue
            perimeter = cv2.arcLength(contour, True)
            points = cv2.approxPolyDP(contour, .04 * perimeter, True).reshape(-1, 2)
            if len(points) != 3:
                continue
            # The apex must point RIGHT, with a near-vertical left edge.
            vertices = sorted(points.tolist(), key=lambda p: p[0])
            left_a, left_b, apex = vertices
            if abs(left_a[0] - left_b[0]) > .35 * tw:
                continue
            if apex[0] - max(left_a[0], left_b[0]) < .50 * tw:
                continue
            if abs(left_a[1] - left_b[1]) < .60 * th:
                continue
            if abs(apex[1] - (left_a[1] + left_b[1]) / 2) > .30 * th:
                continue
            return Match('game_play', px, py, title.confidence, bw, bh,
                         'opencv_blue_play_triangle')
    return None


class RobloxLeaveNavigator:
    """Leave -> Roblox Home game card -> game Play -> Slot A verification.

    Deliberately never guesses a fixed Home card position. Every click needs
    two successive OpenCV detections and a fresh pre-click verification.
    """
    def __init__(self, matcher, capture, click, check, progress):
        self.matcher=matcher
        self.capture=capture
        self.click=click
        self.check=check
        self.progress=progress

    def _find(self, frame, state):
        if state=='leave':return find_leave_disconnect(frame)
        if state=='home_game':return find_roblox_home_game(frame)
        if state=='game_play':return find_roblox_game_play(frame)
        if state=='slot':return self.matcher.slot_a_select(frame)
        raise ValueError('Unknown navigation state')

    def _wait(self, state, timeout, stable=2):
        end=time.monotonic()+timeout
        count=0
        last=None
        while time.monotonic()<end:
            self.check()
            hit=self._find(self.capture(),state)
            if hit and last and abs(hit.x-last.x)<20 and abs(hit.y-last.y)<20:
                count+=1
            else:
                count=1 if hit else 0
            last=hit
            if hit and count>=stable:
                return hit
            time.sleep(.3)
        return None

    def _wait_any(self, states, timeout, stable=2):
        """Observe one screenshot at a time and accept any stable next screen.

        Leave may open Roblox Home OR AOT:R detail directly. Never assume
        which one and never click based on a transitional frame.
        """
        states = tuple(states)
        end = time.monotonic() + timeout
        history = {state: (0, None) for state in states}
        while time.monotonic() < end:
            self.check()
            frame = self.capture()
            for state in states:
                hit = self._find(frame, state)
                count, last = history[state]
                if hit and last and abs(hit.x-last.x) < 20 and abs(hit.y-last.y) < 20:
                    count += 1
                else:
                    count = 1 if hit else 0
                history[state] = (count, hit)
                if hit and count >= stable:
                    return state, hit
            time.sleep(.3)
        return None, None

    def _click_verified(self, state, title, next_state, timeout):
        self.progress(0,10,title,'Detecting',1)
        hit=self._wait(state,timeout)
        if not hit:
            raise ReconnectError(f'{title}: {state} not found; no blind click')
        self.check()
        fresh=self._find(self.capture(),state)
        if not fresh or abs(fresh.x-hit.x)>20 or abs(fresh.y-hit.y)>20:
            raise ReconnectError(f'{title}: target moved before clicking')
        self.progress(0,10,title,'Clicking',1)
        self.click(fresh)
        self.progress(0,10,title,'Verifying',1)
        if isinstance(next_state, str):
            next_states = (next_state,)
        else:
            next_states = tuple(next_state)
        observed, next_hit = self._wait_any(next_states, timeout)
        if not next_hit:
            raise ReconnectError(f'{title}: {" / ".join(next_states)} not confirmed')
        self.progress(0,10,title,'Confirmed',1)
        return observed, next_hit

    def run(self, first_frame=None):
        """Handle both Leave outcomes (Home/detail) plus already-open screens.

        The game detail page is never treated as Roblox Home. Every target is
        confirmed over multiple captures, then rechecked immediately before
        input. Slot A is the hand-off to the existing Image Reconnect steps.
        """
        frame = first_frame if first_frame is not None else self.capture()
        if self.matcher.slot_a_select(frame):
            return True
        if find_leave_disconnect(frame):
            next_state, _ = self._click_verified(
                'leave', 'Leave disconnected game',
                ('slot', 'game_play', 'home_game'), timeout=65)
        elif find_roblox_game_play(frame):
            next_state = 'game_play'
        elif find_roblox_home_game(frame):
            next_state = 'home_game'
        else:
            next_state, _ = self._wait_any(
                ('slot', 'game_play', 'home_game'), timeout=25)
        if next_state == 'slot':
            return True
        if next_state == 'home_game':
            next_state, _ = self._click_verified(
                'home_game', 'Select Attack on Titan Revolution',
                ('game_play', 'slot'), timeout=50)
            if next_state == 'slot':
                return True
        if next_state == 'game_play':
            self._click_verified('game_play', 'Play Attack on Titan Revolution',
                                 'slot', timeout=100)
            return True
        raise ReconnectError('Roblox Home / AOT:R game page / Slot A not verified')



class ImageMatcher:
    def __init__(self, image_dir: str, advisor=None):
        self.advisor = advisor
        self.templates = {}
        for name in (*REGIONS, 'reconnect'):
            path = os.path.join(image_dir, 'rc_'+name+'.png')
            tpl = cv2.imread(path, cv2.IMREAD_COLOR)
            if tpl is None:
                raise FileNotFoundError(path)
            self.templates[name] = tpl

    def find(self, frame: np.ndarray, name: str,
             region: Optional[tuple[float,float,float,float]]=None,
             threshold: Optional[float]=None) -> Optional[Match]:
        if frame is None or frame.ndim != 3:
            return None
        h,w=frame.shape[:2]
        x0,y0,x1,y1=region or REGIONS.get(name,(0,0,1,1))
        xa,ya=max(0,int(x0*w)),max(0,int(y0*h))
        xb,yb=min(w,int(x1*w)),min(h,int(y1*h))
        image=frame[ya:yb,xa:xb]
        tpl=self.templates[name]
        if image.shape[0]<tpl.shape[0] or image.shape[1]<tpl.shape[1]:
            return None
        # New input screenshots are resized to the same 1080p reference by
        # the macro's capture engine. Do not silently relax confidence.
        res=cv2.matchTemplate(image,tpl,cv2.TM_CCOEFF_NORMED)
        _lo,score,_loc,point=cv2.minMaxLoc(res)
        limit = threshold if threshold is not None else THRESHOLDS.get(name,.85)
        tw,th=tpl.shape[1],tpl.shape[0]
        result = Match(name,xa+point[0]+tw//2,ya+point[1]+th//2,score,tw,th)
        if score >= limit:
            return result  # authoritative original OpenCV path
        if self.advisor is None:
            return None
        # AI only assesses the original ROI-constrained candidate. The
        # reconnect state machine still requires stable repeat observations.
        candidate=(result.x,result.y,score,tw,th)
        decision=self.advisor.verify(frame,'rc_'+name,candidate,limit,
                                      roi=(xa,ya,xb,yb))
        return dataclasses.replace(result,source='hybrid_ai') if decision.accepted else None

    def slot_a_select(self,frame):
        """Return SELECT only if geometrically inside and below the SLOT A card."""
        anchor=self.find(frame,'slot_a')
        if anchor is None:
            return None
        h,w=frame.shape[:2]
        # SELECT in the same slot column, below anchor: never B/C.
        region=(max(0,(anchor.x-120)/w), min(1,(anchor.y+80)/h),
                min(1,(anchor.x+120)/w), min(1,(anchor.y+380)/h))
        selected=self.find(frame,'select',region=region)
        return selected if selected and abs(selected.x-anchor.x)<120 else None


@dataclasses.dataclass(frozen=True)
class Action:
    name: str
    target: str
    next_marker: str
    timeout: float = 50


class ImageReconnectFlow:
    """Run a *verified* UI route. Only accepts advancement after next marker.

    capture: returns latest BGR screen frame in the application's 1080p space
    click: converts a Match into physical on-screen click (provided by main.py)
    check: throws on Stop; waits on Pause; must be called frequently
    disconnect: raises NewDisconnect if the disconnect page reappears mid-route
    progress: status message for user UI; must never include PS Code
    enter_code: types private PS code, without logging it
    mission_ready: checks the existing skill/HUD marker to confirm gameplay
    """
    def __init__(self, matcher, capture, click, check, disconnect, progress,
                 enter_code, mission_ready, server_mode='public', ps_code=''):
        self.matcher=matcher;self.capture=capture;self.click=click
        self.check=check;self.disconnect=disconnect;self.progress=progress
        self.enter_code=enter_code;self.mission_ready=mission_ready
        self.mode=server_mode;self.ps_code=ps_code
        if self.mode not in ('public','ps_code','gamepass'):
            raise ValueError('Unsupported server mode')
        if self.mode=='ps_code' and (not ps_code or any(ord(c)<32 for c in ps_code) or len(ps_code)>128):
            raise ValueError('Valid PS Code required')

    def _sleep(self, seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            self.check()
            time.sleep(min(.12, max(0,until-time.monotonic())))

    def _wait_for(self,name,timeout=55,quiet=False,stable=2,allow_disconnect=False):
        expiry=time.monotonic()+timeout; count=0;last=None
        while time.monotonic()<expiry:
            self.check()
            frame=self.capture()
            if not allow_disconnect and self.disconnect(frame):
                raise NewDisconnect('Reconnect appeared again during navigation')
            match=(self.matcher.slot_a_select(frame) if name=='slot_select'
                   else self.matcher.find(frame,name))
            if match is not None and last is not None and abs(last.x-match.x)<16 and abs(last.y-match.y)<16:
                count+=1
            else:count=1 if match else 0
            last=match
            if match and count>=stable:
                return match
            self._sleep(.3)
        return None

    def _step(self,index,total,name,target,next_marker,timeout=50):
        """Retry one action only when the same target remains visible.
        A button being clicked is not success. Next-screen marker is success.
        """
        for attempt in range(1,4):
            self.progress(index,total,name,'Detecting',attempt)
            hit=self._wait_for(target,timeout=timeout)
            if hit is None:
                raise ReconnectError(f'{name}: {target} not detected')
            # Verify *fresh* screenshot immediately before interacting.
            fresh=self.capture()
            if self.disconnect(fresh):raise NewDisconnect('Reconnect returned')
            still=(self.matcher.slot_a_select(fresh) if target=='slot_select'
                   else self.matcher.find(fresh,target))
            if not still or abs(still.x-hit.x)>16 or abs(still.y-hit.y)>16:
                continue
            self.progress(index,total,name,'Clicking',attempt)
            self.check();self.click(still)
            if name=='Enter PS Code':
                self.check();self.enter_code(self.ps_code)
            self.progress(index,total,name,'Verifying',attempt)
            if name=='Enter PS Code':
                # GENERATE was already visible before typing the PS Code.
                # Its continued presence is NOT evidence the code was entered.
                # Verify the original three-dash placeholder disappeared,
                # and that Generate is still in the VIP Servers panel.
                expiry=time.monotonic()+min(15,timeout)
                verified=0
                while time.monotonic()<expiry:
                    self.check()
                    frame=self.capture()
                    if self.disconnect(frame):raise NewDisconnect('Reconnect returned')
                    if (self.matcher.find(frame,'ps_input') is None and
                            self.matcher.find(frame,'generate') is not None):
                        verified+=1
                    else:verified=0
                    if verified>=2:break
                    self._sleep(.3)
                next_confirmed=verified>=2
            else:
                next_confirmed=bool(self._wait_for(next_marker, timeout=timeout))
            if next_confirmed:
                self.progress(index,total,name,'Confirmed',attempt)
                return True
            self.progress(index,total,name,'Retrying',attempt)
        raise ReconnectError(f'{name}: next screen {next_marker} not confirmed')

    def _ensure_vip_servers(self,index,total):
        """Do not click CODES if VIP SERVERS is already open in the menu.

        The user's full screenshot shows a separate CODES/REDEEM panel and
        VIP SERVERS controls simultaneously. Their positions must not be
        mistaken for a screen transition caused by clicking CODES.
        """
        self.progress(index,total,'Locate VIP Servers','Detecting',1)
        if self._wait_for('generate',timeout=2,stable=2):
            self.progress(index,total,'Locate VIP Servers','Confirmed',1)
            return
        # If game UI changes and VIP Servers is hidden, a CODES click may
        # expose it; verify the VIP-only GENERATE control appears afterwards.
        self._step(index,total,'Open VIP Servers','codes','generate',timeout=15)

    def run(self):
        route=[('Select Slot A','slot_select','play_small')]
        if self.mode=='public':
            route += [('Open Public Servers','play_small','lobby'),
                      ('Join Lobby','lobby','play_large')]
        elif self.mode=='ps_code':
            route += [('Ensure VIP Servers','codes','generate'),
                      ('Enter PS Code','ps_input','generate'),
                      # Generate returns to the LOBBY / TRADE choice screen.
                      # Do not wait for the large PLAY button until LOBBY is clicked.
                      ('Generate Private Server','generate','lobby'),
                      ('Join Lobby','lobby','play_large')]
        else:
            # Preserve the original Gamepass path until its post-Generate
            # screen has been verified; the reported bug concerns PS Code.
            route += [('Ensure VIP Servers','codes','generate'),
                      ('Generate Gamepass Server','generate','play_large')]
        route += [('Open Missions Menu','play_large','missions'),
                  ('Select Missions','missions','outskirts'),
                  # CREATE can already be visible with SHIGANSHINA selected.
                  # Verify the *central* OUTSKIRTS title changes first.
                  ('Select Outskirts','outskirts','outskirts_selected'),
                  ('Create Mission','create','start')]
        total=len(route)+1
        for n,(label,target,gate) in enumerate(route,1):
            if label=='Ensure VIP Servers':
                self._ensure_vip_servers(n,total)
            else:
                self._step(n,total,label,target,gate,timeout=75 if (target=='generate' or gate=='play_large') else 40)
        # START must be the modal's START, not the button behind the modal.
        self.progress(total,total,'Start Mission','Detecting',1)
        button=self._wait_for('start',timeout=35)
        if not button:raise ReconnectError('Mission START is not visible in the room popup')
        self.check();self.click(button)
        self.progress(total,total,'Waiting for Mission HUD','Verifying',1)
        expiry=time.monotonic()+100
        while time.monotonic()<expiry:
            self.check();frame=self.capture()
            if self.disconnect(frame):raise NewDisconnect('Disconnected while mission was loading')
            if self.mission_ready(frame):
                self.progress(total,total,'Mission Confirmed','Confirmed',1)
                return True
            self._sleep(.4)
        raise ReconnectError('Mission HUD not confirmed after START')
