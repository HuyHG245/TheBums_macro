"""Developer-only offline template tester (not included in the user EXE).

Usage:
  python developer/detection_lab.py --screenshot screen.png --template image/retry_text.png --threshold .92
Optional: --roi 0.4,0.56,0.79,0.94 --output /path/annotated.png
"""
import argparse
from pathlib import Path
import cv2


def match(screenshot, template, region=(0,0,1,1)):
    height,width=screenshot.shape[:2]
    x0,y0,x1,y1=region
    a,b,c,d=int(x0*width),int(y0*height),int(x1*width),int(y1*height)
    crop=screenshot[b:d,a:c]
    if crop.shape[1]<template.shape[1] or crop.shape[0]<template.shape[0]:
        raise ValueError('Template larger than the supplied search region')
    _mn,score,_mnloc,(x,y)=cv2.minMaxLoc(cv2.matchTemplate(crop,template,cv2.TM_CCOEFF_NORMED))
    return score,(a+x,b+y,template.shape[1],template.shape[0])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screenshot',required=True)
    parser.add_argument('--template',required=True)
    parser.add_argument('--threshold',type=float,default=.92)
    parser.add_argument('--roi',default='0,0,1,1')
    parser.add_argument('--output')
    args=parser.parse_args()
    src=cv2.imread(args.screenshot,cv2.IMREAD_COLOR)
    tpl=cv2.imread(args.template,cv2.IMREAD_COLOR)
    if src is None or tpl is None:parser.error('Screenshot/template cannot be loaded')
    try:region=tuple(float(x) for x in args.roi.split(','))
    except ValueError:parser.error('ROI must contain x0,y0,x1,y1 fractions')
    if len(region)!=4 or not (0<=region[0]<region[2]<=1 and 0<=region[1]<region[3]<=1):
        parser.error('Invalid normalized ROI')
    score,(x,y,w,h)=match(src,tpl,region)
    accepted=score>=args.threshold
    print(f'Match score: {score:.4f} | Threshold: {args.threshold:.4f} '
          f'| Accepted: {accepted} | Rectangle: {(x,y,w,h)}')
    if args.output:
        cv2.rectangle(src,(x,y),(x+w,y+h),(0,255,0) if accepted else (0,0,255),2)
        cv2.putText(src,f'{score:.3f}',(x,max(15,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.6,
                    (0,255,0) if accepted else (0,0,255),2)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(args.output,src):parser.error('Failed writing annotated output')
        print('Annotated screenshot:',args.output)

if __name__=='__main__':main()
