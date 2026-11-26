"""Button matching on normalized Roblox frames, with a restricted search per control."""
import cv2


def locate(frame, templates, bounds, minimum, scales=(1.0, .85, 1.15, .70, 1.30)):
    h, w = frame.shape[:2]
    left, top, right, bottom = bounds
    x, y = int(w*left), int(h*top)
    area = frame[y:int(h*bottom), x:int(w*right)]
    for template in templates.values():
        for scale in scales:
            tpl = template if scale == 1 else cv2.resize(template, None, fx=scale, fy=scale)
            th, tw = tpl.shape[:2]
            if th > area.shape[0] or tw > area.shape[1]:
                continue
            _, score, _, pos = cv2.minMaxLoc(cv2.matchTemplate(area, tpl, cv2.TM_CCOEFF_NORMED))
            if score >= minimum:
                return (x+pos[0]+tw//2, y+pos[1]+th//2, score, tw, th)
    return None


def locate_skip(frame, templates, bounds, minimum, scales=(1., .85, 1.15, .70, 1.30, .55, 1.50)):
    """Coarse half-resolution search; accept only a refined full-color match.

    Avoid a large multi-scale full-resolution scan on every combat tick.
    """
    if not templates:
        return None
    h,w=frame.shape[:2]
    left,top,right,bottom=bounds
    x,y=int(w*left),int(h*top)
    area=frame[y:int(h*bottom),x:int(w*right)]
    if min(area.shape[:2])<2:
        return None
    small=cv2.resize(area,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
    for template in templates.values():
        for scale in scales:
            tpl=template if scale==1 else cv2.resize(template,None,fx=scale,fy=scale)
            th,tw=tpl.shape[:2]
            coarse=cv2.resize(tpl,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
            if coarse.shape[0]>small.shape[0] or coarse.shape[1]>small.shape[1]:
                continue
            _,score,_,(cx,cy)=cv2.minMaxLoc(cv2.matchTemplate(small,coarse,cv2.TM_CCOEFF_NORMED))
            if score<.62:
                continue
            x0,y0=max(0,cx*2-4),max(0,cy*2-4)
            local=area[y0:min(area.shape[0],cy*2+th+5),x0:min(area.shape[1],cx*2+tw+5)]
            if local.shape[0]<th or local.shape[1]<tw:
                continue
            _,score,_,(dx,dy)=cv2.minMaxLoc(cv2.matchTemplate(local,tpl,cv2.TM_CCOEFF_NORMED))
            if score>=minimum:
                return (x+x0+dx+tw//2,y+y0+dy+th//2,score,tw,th)
    return None


def seismic_ready(frame, templates, minimum=.84):
    """Positive evidence of the bright Skill 5 icon in the lower HUD.

    False means the ready icon was not observed, not proof of cooldown.
    None means no template is available. Normalized correlation alone would
    accept a uniformly dimmed icon, so also check the gold foreground level.
    The photometric gate is a candidate heuristic, not calibrated in-game.
    """
    if not templates:
        return None
    h, w = frame.shape[:2]
    area = frame[int(h*.67):, int(w*.31):int(w*.82)]
    small = cv2.resize(area, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    for template in templates.values():
        for scale in (1., .55, .65, .75, .85, 1.15, 1.30, .45):
            tpl = template if scale == 1 else cv2.resize(template, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            th, tw = tpl.shape[:2]
            if th > area.shape[0] or tw > area.shape[1]:
                continue
            coarse = cv2.resize(tpl, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
            if coarse.shape[0] > small.shape[0] or coarse.shape[1] > small.shape[1]:
                continue
            _, approximate, _, (cx, cy) = cv2.minMaxLoc(cv2.matchTemplate(small, coarse, cv2.TM_CCOEFF_NORMED))
            if approximate < .63:
                continue
            # Refine at native resolution; account for the half-pixel grid.
            left, top = max(0, cx*2-4), max(0, cy*2-4)
            local = area[top:min(area.shape[0], cy*2+th+5), left:min(area.shape[1], cx*2+tw+5)]
            if local.shape[0] < th or local.shape[1] < tw:
                continue
            _, score, _, (dx, dy) = cv2.minMaxLoc(cv2.matchTemplate(local, tpl, cv2.TM_CCOEFF_NORMED))
            x, y = left+dx, top+dy
            if score < minimum:
                continue
            gold = (tpl[:,:,2] > 130) & (tpl[:,:,1] > 90) & (tpl[:,:,2].astype('int16') - tpl[:,:,0] > 35)
            if not gold.any():
                continue
            patch = area[y:y+th, x:x+tw]
            # Brightness and hue evidence at the actual matching pixels.
            if (patch[:,:,2][gold].mean() >= .72 * tpl[:,:,2][gold].mean()
                    and patch[:,:,1][gold].mean() >= .72 * tpl[:,:,1][gold].mean()
                    and (patch[:,:,2].astype('int16') - patch[:,:,0])[gold].mean() >= 30):
                return True
    return False


class ColossalHUD:
    """Slot-local HUD observer. Slot 2/3 identifies form, not Heatwave readiness.

    Geometry is learned from the supplied 1080p playtest: slots are 84.5px
    apart, at a common scale. No evidence is represented by None, not cooldown.
    Cached geometry is validated on every observation; no full-screen search.
    """
    scales = tuple(sorted((i / 100 for i in range(45, 151, 5)), key=lambda s: abs(s-1.)))

    def __init__(self, templates, heat_threshold=.80):
        self.templates = {k: next(iter(templates.get(k, {}).values()), None)
                          for k in ('form2', 'form3', 'heat_ready', 'seismic_local')}
        self.geometry = None
        self.scaled = {}
        self.heat_threshold = float(heat_threshold)

    def template(self, key, scale):
        cache_key = (key, scale)
        if cache_key not in self.scaled:
            tpl = self.templates[key]
            self.scaled[cache_key] = (None if tpl is None else cv2.resize(
                tpl, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR))
        return self.scaled[cache_key]

    def patch(self, frame, key, x, y, scale):
        tpl = self.template(key, scale)
        if tpl is None:
            return None, None, 0.
        h,w = tpl.shape[:2]
        left, top = int(round(x-w/2)), int(round(y-h/2))
        pad = max(4, int(8*scale))
        x0,y0 = max(0,left-pad),max(0,top-pad)
        area = frame[y0:min(frame.shape[0],top+h+pad),x0:min(frame.shape[1],left+w+pad)]
        if area.shape[0]<h or area.shape[1]<w:
            return None,tpl,0.
        _,score,_,(dx,dy)=cv2.minMaxLoc(cv2.matchTemplate(area,tpl,cv2.TM_CCOEFF_NORMED))
        return area[dy:dy+h,dx:dx+w],tpl,score

    @staticmethod
    def brightness(patch,tpl):
        if patch is None or tpl is None:
            return None
        gold=(tpl[:,:,2]>140)&(tpl[:,:,1]>90)&(tpl[:,:,2].astype('int16')-tpl[:,:,0]>35)
        if not gold.any():
            return None
        return float(patch[:,:,2][gold].mean()/tpl[:,:,2][gold].mean())

    def validate(self, frame, geom):
        x,y,s=geom
        a,t,sa=self.patch(frame,'form2',x,y,s)
        b,u,sb=self.patch(frame,'form3',x+84*s,y,s)
        # A permanently unused slot 2 is a strong form anchor. Slot 3 or
        # slot 4 corroborates weaker matches; slot 3 can be on cooldown.
        h,v,sh=self.patch(frame,'heat_ready',x+169*s,y,s)
        if sa>=.88 or (sa>=.73 and max(sb,sh)>=.76):
            return True
        return sb>=.88 and sh>=.76

    def find(self, frame):
        if self.geometry and self.validate(frame,self.geometry):
            return self.geometry
        self.geometry=None
        h,w=frame.shape[:2]
        # Slots 2/3 straddle the center of the screen. Restrict the vertical
        # search to the lower HUD, including window/taskbar variation.
        x0,y0=int(w*.34),int(h*.76)
        area=frame[y0:int(h*.96),x0:int(w*.58)]
        if not area.size:
            return None
        small=cv2.resize(area,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
        candidates=[]
        for key, offset in (('form2',0),('form3',84)):
            for scale in self.scales:
                tpl=self.template(key,scale)
                if tpl is None:
                    continue
                coarse=cv2.resize(tpl,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
                th,tw=coarse.shape[:2]
                if th>small.shape[0] or tw>small.shape[1]:
                    continue
                _,score,_,(dx,dy)=cv2.minMaxLoc(cv2.matchTemplate(small,coarse,cv2.TM_CCOEFF_NORMED))
                if score>=.64:
                    candidates.append((score,(x0+2*dx+tpl.shape[1]/2-offset*scale,
                                               y0+2*dy+tpl.shape[0]/2,scale)))
        best = None
        for _,geom in sorted(candidates,reverse=True)[:12]:
            if self.validate(frame,geom):
                x,y,s=geom
                scores=[self.patch(frame,key,x+offset*s,y,s)[2]
                        for key,offset in (('form2',0),('form3',84),('heat_ready',169))]
                # Use both form anchors to refine scale rather than accepting
                # the first coarse hit. The heat score is only a tie-breaker.
                score=max(scores[:2])+.65*min(scores[:2])+.10*scores[2]
                if best is None or score>best[0]:
                    best=(score,geom)
        if best:
            self.geometry=best[1]
            return self.geometry
        return None

    def state(self,frame,key,x,y,s):
        patch,tpl,score=self.patch(frame,key,x,y,s)
        ratio=self.brightness(patch,tpl)
        if ratio is None:
            return None
        minimum = self.heat_threshold if key == 'heat_ready' else .76
        if score>=minimum and ratio>=.72:
            return True
        # A dimmed slot in a verified Colossal HUD, not generic absence.
        if ratio<=.55 and score>=.25:
            return False
        return None

    def observe(self,frame):
        geom=self.find(frame)
        if geom is None:
            return False,None,None
        x,y,s=geom
        return (True,self.state(frame,'heat_ready',x+169*s,y,s),
                self.state(frame,'seismic_local',x+253*s,y,s))
