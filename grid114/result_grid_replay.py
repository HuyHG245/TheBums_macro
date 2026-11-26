import cv2


def square_boxes(frame, mask, flexible=False):
    h, w = frame.shape[:2]
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        in_panel = .24*w < x < .79*w and .48*h < y < .77*h and .032*w < bw < .040*w
        if flexible:
            in_panel = .38*h < y < .87*h and .035*h < bw < .14*h
        if in_panel and .91 < bw/bh < 1.09:
            boxes.append((x,y,bw,bh))
    kept = []
    for box in sorted(boxes, key=lambda b: -b[2]*b[3]):
        if all(abs(box[0]-b[0]) > .012*w or abs(box[1]-b[1]) > .015*h for b in kept):
            kept.append(box)
    return kept


def grid(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    anchors = square_boxes(frame, cv2.inRange(hsv, (10,60,100), (45,255,255)))
    kept = list(anchors)
    colored = cv2.inRange(hsv, (0,45,80), (179,255,255))
    candidates = square_boxes(frame, colored, flexible=True)
    edges = cv2.Canny(frame, 45, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    candidates.extend(square_boxes(frame, edges, flexible=True))
    for candidate in candidates:
        x, y, width, height = candidate
        if any(abs(x-ax) < width/3 and abs(y-ay) < height/3 for ax, ay, _, _ in kept):
            continue
        # Result tiles share a row or a column across the two reward rows.
        neighbors = [other for other in candidates if abs(other[2]-width) < width/8
                     and abs(y-other[1]) < height/10 and .95*width < abs(x-other[0]) < 1.3*width]
        if neighbors or any((abs(y-ay) < height/8 and abs(width-aw) < width/8) or
               (abs(x-ax) < width/8 and .9*height < abs(y-ay) < 1.4*height)
               for ax, ay, aw, _ in anchors):
            kept.append(candidate)
    # A wrapped row can contain just one tile, including in cropped screenshots.
    first_pass = list(kept)
    for x, y, width, height in candidates:
        if any(abs(x-ax) < width/3 and abs(y-ay) < height/3 for ax, ay, _, _ in kept):
            continue
        if any(abs(x-ax) < width/8 and abs(width-aw) < width/8
               and .95*height < y-ay < 1.35*height for ax, ay, aw, _ in first_pass):
            kept.append((x, y, width, height))
    h = frame.shape[0]
    rows = []
    for box in sorted(kept, key=lambda b: b[1]):
        if not rows or box[1]-rows[-1][0][1] > .02*h:
            rows.append([])
        rows[-1].append(box)
    return [sorted(row) for row in rows]
