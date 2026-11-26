"""Developer-only contact sheets from existing candidate indexes.

All boxes are unverified. No full frames, no auto-labeling, no dataset writes.
Compare exact crops and close v9 lookalikes for triage, not ground truth.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np

from ai_studio import digest, require_space, video_source_hint, SPLITS, IMAGE_TYPES
from hybrid_dataset import allowed_labels
from video_review_index import load_index
from video_dataset_collector import get_registry

COLS, ROWS, TILE_W, TILE_H = 4, 3, 350, 245


def feature(crop):
    image = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def pixel_digest(crop):
    return hashlib.sha256(str(crop.shape).encode() + crop.tobytes()).hexdigest()


def iou(a, b):
    x, y, w, h = a
    u, v, q, r = b
    overlap = max(0, min(x+w, u+q)-max(x,u)) * max(0, min(y+h,v+r)-max(y,v))
    return overlap / max(1, w*h+q*r-overlap)


def baseline_features(root):
    rows = []
    for split in SPLITS:
        for p in sorted((root / split).rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_TYPES:
                try:
                    image = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
                except (OSError, cv2.error):
                    continue
                if image is not None:
                    rows.append((p.relative_to(root).as_posix(), feature(image)))
    return rows


def nearest(crop, baseline):
    if not baseline:
        return None, None
    f = feature(crop)
    # This deliberately uses color as well as spatial shape, since fragment
    # icons may look similar in grayscale. It is NOT an identity decision.
    path, distance = min(((p, float(np.abs(f-img).mean())) for p, img in baseline),
                         key=lambda pair: pair[1])
    return path, round(distance, 5)


def collect(video, root, baseline):
    sha = digest(video)
    index = root / ("review_" + sha[:12] + ".json")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError("Cannot open recording: " + video.name)
    out, omitted = [], []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or frames <= 0:
            raise ValueError("Cannot determine recording duration")
        events = load_index(index, source_sha256=sha, duration=frames/fps)
        source = get_registry(root).get(sha)
        hint = video_source_hint(root, video)
        if source:
            known = (source["split"], source["session"])
            if hint is not None and known != hint:
                raise ValueError("Existing registry conflicts with baseline provenance: " + video.name)
        else:
            known = hint
        grouped = defaultdict(list)
        for event in events:
            if not event.get("context_only") and event["label"] in allowed_labels():
                grouped[event["seconds"]].append(event)
        for seconds, proposals in sorted(grouped.items()):
            frame_no = round(seconds * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok:
                omitted.append({"seconds":seconds,"reason":"frame unreadable","events":len(proposals)})
                continue
            height, width = frame.shape[:2]
            for event in proposals:
                x, y, w, h = event["bbox"]
                if min(w,h)<12 or max(w,h)>1024 or x+w>width or y+h>height:
                    omitted.append({"seconds":seconds,"reason":"invalid crop geometry","label":event["label"]})
                    continue
                crop = frame[y:y+h,x:x+w].copy()
                matches = [other["label"] for other in proposals if other is not event
                           and other["label"] != event["label"] and iou(event["bbox"], other["bbox"]) >= .55]
                closest, dist = nearest(crop, baseline)
                out.append({"source":sha,"video":video.name,"seconds":seconds,
                            "frame_index":frame_no,"bbox":event["bbox"],"label":event["label"],
                            "score":event["score"],"overlap_labels":sorted(set(matches)),
                            "baseline_nearest":closest,"baseline_distance":dist,
                            "split":known[0] if known else None,"session":known[1] if known else None,
                            "pixel_sha256":pixel_digest(crop),"crop":crop})
    finally:
        cap.release()
    return out, omitted


def draw_tile(canvas, entry, x, y):
    crop = entry["crop"]
    def line(s, dy, scale=.48, color=(30,35,40)):
        cv2.putText(canvas, str(s), (x+10,y+dy), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)
    cv2.rectangle(canvas,(x+1,y+1),(x+TILE_W-2,y+TILE_H-2),(194,201,209),1)
    line(entry["id"]+"  "+entry["label"],22,.54)
    line(f"CV {entry['score']:.3f} | {entry['seconds']:.3f}s | {entry['source'][:8]}",43)
    box_w,box_h=TILE_W-22,120
    ratio=min(box_w/crop.shape[1],box_h/crop.shape[0])
    new_w,new_h=max(1,round(crop.shape[1]*ratio)),max(1,round(crop.shape[0]*ratio))
    zoom=cv2.resize(crop,(new_w,new_h),interpolation=cv2.INTER_NEAREST if ratio>1 else cv2.INTER_AREA)
    dx=x+(TILE_W-new_w)//2
    dy=y+49+(box_h-new_h)//2
    canvas[dy:dy+new_h,dx:dx+new_w]=zoom
    overlap=entry["overlap_labels"]
    line("Overlap: "+(", ".join(overlap) if overlap else "-")[:42],189,.43,
         (45,60,170) if overlap else (30,35,40))
    dist=entry["baseline_distance"]
    close = dist is not None and dist <= .055
    line("v9 nearest: "+(f"{dist:.3f} (similar)" if close else f"{dist:.3f}" if dist is not None else "n/a"),209,.46)
    line("Source: "+(entry["split"] or "UNRESOLVED")+" | Not verified",229,.46)


def export(root, videos, output):
    root, output=Path(root),Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh export directory")
    require_space(output.parent, 256*1024**2)
    base=baseline_features(root)
    entries, skipped, seen_sources=[],[],set()
    for video in map(Path,videos):
        sha=digest(video)
        if sha in seen_sources:
            print("Skip exact duplicate recording:", video.name, flush=True)
            continue
        seen_sources.add(sha)
        if not (root/("review_"+sha[:12]+".json")).is_file():
            skipped.append({"source":sha,"reason":"review index missing"})
            print("No candidate index:",video.name,flush=True)
            continue
        rows, errors=collect(video,root,base)
        entries.extend(rows)
        skipped.extend({"source":sha,**e} for e in errors)
        print(video.name,":",len(rows),"unverified crops",flush=True)
        if len(entries)>2000:
            raise ValueError("Export limit 2000 crops; select fewer recordings")
    if not entries:
        raise ValueError("No valid candidate crops. Run Find candidates first.")
    # Do not collapse different classes, sources, or similar-but-not-identical
    # images. Exact repeats of the SAME label in the SAME recording are grouped.
    unique={}
    for e in entries:
        key=(e["source"],e["label"],e["pixel_sha256"])
        if key in unique:
            old=unique[key]
            old["also_at"].append({"seconds":e["seconds"],"bbox":e["bbox"],"score":e["score"]})
            old["overlap_labels"]=sorted(set(old["overlap_labels"]+e["overlap_labels"]))
        else:
            e["also_at"]=[]
            unique[key]=e
    rows=sorted(unique.values(),key=lambda e:(not bool(e["overlap_labels"]),e["label"],e["source"],e["seconds"]))
    stage=Path(tempfile.mkdtemp(prefix="review_export_",dir=output.parent))
    try:
        for idx,e in enumerate(rows,1):
            e["id"]=f"C{idx:04d}"
        for offset in range(0,len(rows),COLS*ROWS):
            batch=rows[offset:offset+COLS*ROWS]
            canvas=np.full((ROWS*TILE_H+48,COLS*TILE_W,3),248,np.uint8)
            cv2.putText(canvas,"THE BUMS AI STUDIO | UNVERIFIED CROPS | Inspect privacy before sharing",
                        (12,29),cv2.FONT_HERSHEY_SIMPLEX,.65,(30,35,40),1,cv2.LINE_AA)
            for j,e in enumerate(batch):
                draw_tile(canvas,e,(j%COLS)*TILE_W,48+(j//COLS)*TILE_H)
            ok,png=cv2.imencode(".png",canvas)
            if not ok:
                raise OSError("PNG encoding failed")
            (stage/f"sheet_{offset//(COLS*ROWS)+1:03d}.png").write_bytes(png.tobytes())
        manifest={"schema":1,"warning":"Unverified suggestions; no training data was added.",
                  "total_candidates":len(entries),"unique_candidates":len(rows),
                  "skipped":skipped,"counts":dict(Counter(e["label"] for e in rows)),
                  "candidates":[{k:v for k,v in e.items() if k!="crop"} for e in rows]}
        (stage/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    print(f"Exported {len(rows)} numbered crops from {len(entries)} candidates. "
          f"Exact repeats grouped: {len(entries)-len(rows)}. Skipped records: {len(skipped)}.")
    print("No training data was added. Inspect sheets for private data before sharing.")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--video",action="append",required=True)
    args=parser.parse_args()
    export(args.root,args.video,args.out)


if __name__=="__main__":
    main()
