"""Read labelled perk icons and their printed quantities on an Obtained grid."""
from pathlib import Path
import cv2
import numpy as np
from .currency_v7 import review

LABELS = ('epic_perk', 'legendary_perk', 'mythic_perk')
REFERENCES = np.load(Path(__file__).with_name('v114') / 'perk_icons.npz')


def feature(tile):
    h, w = tile.shape[:2]
    roi = tile[round(.15*h):round(.72*h), round(.18*w):round(.82*w)]
    return cv2.resize(roi, (32, 28), interpolation=cv2.INTER_AREA)


def number_patch(tile):
    h, w = tile.shape[:2]
    roi = tile[round(.68*h):round(.94*h), round(.54*w):round(.97*w)]
    gray = roi.min(2)
    return cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA)


def number(tile, bank):
    h, w = tile.shape[:2]
    primary, _ = review(tile[round(.67*h):, round(.45*w):-2], bank)
    patch = number_patch(tile)
    if patch.std() < 6:
        return None
    scores = sorted(((max(float(cv2.matchTemplate(patch, ref, cv2.TM_CCOEFF_NORMED)[0, 0])
                          for ref in REFERENCES[k]), int(k[1:]))
                     for k in REFERENCES.files if k[0] == 'n' and k[1:].isdigit()), reverse=True)
    match = scores[0][1] if scores[0][0] >= .85 and scores[0][0]-scores[1][0] >= .07 else None
    if primary is not None and match is not None and primary != match:
        return None
    return primary if primary is not None else match


def read_perks(frame, rows, bank):
    slots = {label: [] for label in LABELS}
    for row in rows:
        for x, y, w, h in row:
            tile = frame[y:y+h, x:x+w]
            sample = feature(tile)
            scores = sorted(((float(cv2.matchTemplate(sample, REFERENCES[label],
                             cv2.TM_CCOEFF_NORMED)[0, 0]), label) for label in LABELS), reverse=True)
            score, label = scores[0]
            if score < .82 or score - scores[1][0] < .08:
                continue
            negative = max(float(cv2.matchTemplate(sample, ref, cv2.TM_CCOEFF_NORMED)[0, 0])
                           for ref in REFERENCES['negatives'])
            if score - negative < .08:
                continue
            quantity = number(tile, bank)
            slots[label].append((x // 8, y // 8, quantity))
    return {k: tuple(v) for k, v in slots.items() if v and all(s[2] is not None for s in v)}
