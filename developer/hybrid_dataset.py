"""Developer-only dataset audit and opt-in ROI capture for shared Hybrid AI.

Usage:
    python developer/hybrid_dataset.py audit --root datasets/hybrid
    python developer/hybrid_dataset.py capture --root datasets/hybrid --split train \
        --session pc1_20261123 --label retry_text --roi 100,200,180,65 --count 30

Only explicitly provided CROPPED screen rectangles are saved. Never collect
private-server codes, chat, usernames, Discord tokens or full desktop screenshots.
Capture does not send game input or upload any user data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

SPLITS = ('train', 'val', 'test')
BACKGROUND = '__background__'
VALID_SESSION = re.compile(r'^[A-Za-z0-9_-]{4,80}$')
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}


def allowed_labels():
    # Import only the stable label table, not main.py (which initializes GUI/input).
    from thebums_hybrid_ai import GATES
    return set(GATES) | {BACKGROUND}


def enumerate_samples(root: Path):
    """Enumerate all labels from independent recording sessions.

    Folder format: <root>/<split>/<session_id>/<label>/*.png.
    A session_id identifies a gameplay recording/device/time block, NOT a frame.
    """
    root = Path(root)
    samples = []
    issues = []
    known = allowed_labels()
    if not root.is_dir():
        return [], [f'Dataset folder missing: {root}']
    hash_owners = {}
    session_splits = {}
    for split in SPLITS:
        base = root / split
        if not base.is_dir():
            issues.append(f'Missing split: {split}')
            continue
        for session in sorted(base.iterdir()):
            if not session.is_dir():
                continue
            if not VALID_SESSION.fullmatch(session.name):
                issues.append(f'Invalid session ID: {session.name}')
                continue
            if session.name in session_splits and session_splits[session.name] != split:
                issues.append(f'Session reused across splits: {session.name}')
            session_splits[session.name] = split
            for folder in sorted(session.iterdir()):
                if not folder.is_dir():
                    continue
                if folder.name not in known:
                    issues.append(f'Unknown label {folder.name} in {split}/{session.name}')
                    continue
                for path in sorted(folder.iterdir()):
                    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if im is None or min(im.shape[:2]) < 12:
                        issues.append(f'Unreadable or tiny crop: {path}')
                        continue
                    if max(im.shape[:2]) > 1024:
                        issues.append(f'Crop >1024 pixels; inspect for private data: {path}')
                        continue
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    owner = hash_owners.get(digest)
                    if owner and owner != (split, session.name, folder.name):
                        issues.append(f'Duplicate image across records: {path} vs {owner}')
                    else:
                        hash_owners[digest] = (split, session.name, folder.name)
                    samples.append((split, session.name, folder.name, path))
    if not samples:
        issues.append('No labeled crops found')
    return samples, issues


def audit(root: Path, *, for_release=False):
    samples, errors = enumerate_samples(root)
    counts = Counter((split, label) for split, _, label, _ in samples)
    sessions = {split: {session for s, session, _, _ in samples if s == split}
                for split in SPLITS}
    labels = sorted({label for _, _, label, _ in samples} - {BACKGROUND})
    if not labels:
        errors.append('No gameplay target labels; background-only data cannot train a useful classifier')
    for split in SPLITS:
        if not any(s == split and lab == BACKGROUND for s, _, lab, _ in samples):
            errors.append(f'Missing real hard-negative background crops in {split}')
        for label in labels:
            if not counts.get((split, label)):
                errors.append(f'Missing {label} in {split}')
    # The developer can train a preview with smaller data, but a shared, publicly
    # enabled model needs independently gathered real negatives and held-out sessions.
    if for_release:
        minimum = {'train': 60, 'val': 30, 'test': 50}
        for split in SPLITS:
            if len(sessions[split]) < (2 if split == 'train' else 1):
                errors.append(f'Insufficient independent sessions in {split}')
            for label in labels:
                if counts[(split, label)] < minimum[split]:
                    errors.append(f'Insufficient real {split}/{label}: {counts[(split, label)]} < {minimum[split]}')
            negatives = counts[(split, BACKGROUND)]
            required_neg = {'train': 200, 'val': 100, 'test': 500}[split]
            if negatives < required_neg:
                errors.append(f'Insufficient hard negatives in {split}: {negatives} < {required_neg}')
    return {'samples': samples, 'errors': errors, 'counts': counts,
            'sessions': sessions, 'labels': [BACKGROUND] + labels}


def capture(root: Path, split: str, session: str, label: str, roi: str,
            count: int, interval: float):
    if split not in SPLITS or label not in allowed_labels() or not VALID_SESSION.fullmatch(session):
        raise ValueError('Invalid split/session/label')
    try:
        x, y, width, height = (int(v) for v in roi.split(','))
    except (TypeError, ValueError) as exc:
        raise ValueError('--roi must be x,y,width,height') from exc
    if x < 0 or y < 0 or width < 12 or height < 12 or max(width, height) > 1024:
        raise ValueError('ROI must be a small on-screen crop (12..1024 pixels)')
    if not 1 <= count <= 500 or not 0.15 <= interval <= 30.0:
        raise ValueError('Capture 1..500 frames at intervals >=0.15s')
    # Import mss only on demand; no capture/network/input in tests or training.
    import mss
    dst = Path(root) / split / session / label
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError('Use a new session name to avoid overwriting a recording')
    dst.mkdir(parents=True, exist_ok=True)
    print('Developer-only cropped capture. Check ROI excludes codes, chat, personal info.')
    print('Starting capture in 3 seconds. Use Ctrl+C to stop.')
    time.sleep(3)
    with mss.mss() as scr:
        for idx in range(count):
            region = {'left': x, 'top': y, 'width': width, 'height': height}
            frame = np.asarray(scr.grab(region))[:, :, :3]
            result = cv2.imwrite(str(dst / f'{idx:05d}.png'), frame)
            if not result:
                raise OSError('Screenshot could not be written')
            time.sleep(interval)
    print(f'Captured {count} local CROPS at {dst} (not uploaded).')


def cli():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit', help='Validate labeled session splits without training')
    a.add_argument('--root', type=Path, default=Path('datasets/hybrid'))
    a.add_argument('--release', action='store_true', help='Check minimum data for public model')
    c = sub.add_parser('capture', help='Capture an explicitly selected rectangle; no input or network')
    c.add_argument('--root', type=Path, default=Path('datasets/hybrid'))
    c.add_argument('--split', choices=SPLITS, required=True)
    c.add_argument('--session', required=True)
    c.add_argument('--label', choices=sorted(allowed_labels()), required=True)
    c.add_argument('--roi', required=True)
    c.add_argument('--count', type=int, default=30)
    c.add_argument('--interval', type=float, default=0.4)
    args = p.parse_args()
    if args.command == 'capture':
        capture(args.root, args.split, args.session, args.label, args.roi, args.count, args.interval)
    else:
        result = audit(args.root, for_release=args.release)
        print('Real gameplay samples:', len(result['samples']))
        print('Labels:', ', '.join(result['labels']))
        for key, n in sorted(result['counts'].items()):
            print(f'  {key[0]:5} {key[1]:26} {n}')
        for issue in result['errors']:
            print('ERROR:', issue)
        if result['errors']:
            raise SystemExit(1)
        print('Dataset audit passed (not model validation).')


if __name__ == '__main__':
    cli()
