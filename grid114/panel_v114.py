import cv2


def match(frame,tpl):
    gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(800,450),interpolation=cv2.INTER_AREA)
    search=gray[155:337,215:585]
    return max(float(cv2.matchTemplate(search,cv2.resize(tpl,None,fx=s,fy=s),cv2.TM_CCOEFF_NORMED).max()) for s in (.85,1.,1.15))
