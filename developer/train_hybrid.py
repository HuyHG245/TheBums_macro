"""Train one shared, offline CPU Hybrid AI model from REAL labeled gameplay CROPS.

The developer executes this once, then bundles the validated ONNX result with
The Bums Macro. End users NEVER need Torch, model training, accounts or APIs.

Data folders: datasets/hybrid/{train,val,test}/{session_id}/{label}/*.png
A different recording/session is required in every split. Include many
__background__ near-miss crops and negatives. NEVER use a single PNG template
with synthetic noise as proof of real-world generalization.

Dependencies (developer only): pip install torch onnx onnxruntime numpy opencv-python
Usage: python developer/train_hybrid.py --root datasets/hybrid --out models
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np

from hybrid_dataset import audit, BACKGROUND

INPUT_HW = (96, 96)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
ACCEPT_CONFIDENCE = .98


def image_input(image, size=INPUT_HW):
    rgb = cv2.cvtColor(cv2.resize(image, (size[1], size[0]),
                                    interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    image = rgb.astype(np.float32) / 255.0
    return np.transpose((image - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32),
                        (2, 0, 1)).astype(np.float32)


def compute_ece(probs: np.ndarray, truth: np.ndarray, bins=10):
    confidence, prediction = probs.max(axis=1), probs.argmax(axis=1)
    result = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (confidence >= lo) & (confidence <= hi if i == bins - 1 else confidence < hi)
        if mask.any():
            result += float(mask.mean() * abs((prediction[mask] == truth[mask]).mean() - confidence[mask].mean()))
    return result


def evaluate(probs, truth, labels, threshold=ACCEPT_CONFIDENCE):
    """Selective acceptance: only max probability >=threshold allows borderline CV."""
    result = {'samples': len(truth), 'ece': compute_ece(probs, truth), 'classes': {}}
    predicts = probs.argmax(axis=1)
    best = probs.max(axis=1)
    neg = truth == 0
    # Any high-confidence non-background prediction on a negative is unsafe.
    false_accepts = int((neg & (predicts != 0) & (best >= threshold)).sum())
    result['hard_negative_false_accepts'] = false_accepts
    result['hard_negative_count'] = int(neg.sum())
    for i, label in enumerate(labels[1:], 1):
        accepted = (predicts == i) & (best >= threshold)
        tp = int((accepted & (truth == i)).sum())
        fp = int((accepted & (truth != i)).sum())
        positives = int((truth == i).sum())
        result['classes'][label] = {
            'tested_positives': positives, 'true_positive': tp,
            'false_positive': fp, 'precision': tp / (tp + fp) if tp + fp else 0.0,
            'recall': tp / positives if positives else 0.0}
    return result


def safe_to_release(report, *, min_precision=.99, min_recall=.70, max_ece=.05):
    """Conservative publish gate. Performance must be verified on held-out sessions."""
    if report['samples'] < 550 or report['hard_negative_count'] < 500:
        return False
    if report['hard_negative_false_accepts'] != 0 or report['ece'] > max_ece:
        return False
    return bool(report['classes']) and all(
        item['tested_positives'] >= 50 and item['true_positive'] >= 35 and
        item['precision'] >= min_precision and item['recall'] >= min_recall
        for item in report['classes'].values())


def train(dataset: Path, out: Path, epochs: int, seed: int):
    if (out / 'hybrid_manifest.json').exists():
        raise ValueError('Choose a fresh --out folder; never overwrite a previously approved model')
    # Heavy developer dependencies never enter the ordinary user application.
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    except ImportError as exc:
        raise SystemExit('Developer machine needs torch: pip install torch') from exc
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    inventory = audit(dataset, for_release=True)
    if inventory['errors']:
        raise ValueError('Dataset NOT ready for a shared model:\n- ' + '\n- '.join(inventory['errors']))
    labels = inventory['labels']
    indices = {label: i for i, label in enumerate(labels)}
    all_samples = inventory['samples']

    class Crops(Dataset):
        def __init__(self, split):
            self.items = [(path, indices[label]) for s, _, label, path in all_samples if s == split]
            self.training = split == 'train'
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            path, label = self.items[i]
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f'Unreadable crop: {path}')
            if self.training:
                # Small photometric changes only; do not manufacture fake
                # independent sessions or modify label/position/screen region.
                bgr = cv2.convertScaleAbs(bgr, alpha=random.uniform(.90, 1.10),
                                          beta=random.randint(-8, 8))
            return torch.from_numpy(image_input(bgr)), label

    sets = {split: Crops(split) for split in ('train', 'val', 'test')}
    counts = Counter(label for _, label in sets['train'].items)
    weights = [1. / counts[label] for _, label in sets['train'].items]
    sampler = WeightedRandomSampler(weights, len(sets['train']), replacement=True)
    loaders = {
        'train': DataLoader(sets['train'], batch_size=32, sampler=sampler),
        'val': DataLoader(sets['val'], batch_size=64, shuffle=False),
        'test': DataLoader(sets['test'], batch_size=64, shuffle=False)
    }

    class TinyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 20, 3, padding=1), nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(20, 40, 3, padding=1), nn.BatchNorm2d(40), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(40, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.classifier = nn.Linear(96, len(labels))
        def forward(self, x): return self.classifier(self.features(x))

    model = TinyClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    best_state, best_loss = None, float('inf')
    bad = 0
    for epoch in range(epochs):
        model.train()
        for inputs, truth in loaders['train']:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(inputs), truth)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            sums, total = 0, 0
            for inputs, truth in loaders['val']:
                sums += nn.functional.cross_entropy(model(inputs), truth, reduction='sum').item()
                total += len(truth)
            val_loss = sums / max(1, total)
        print(f'Epoch {epoch+1}/{epochs}: val_cross_entropy={val_loss:.4f}')
        if val_loss < best_loss - 1e-4:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_loss, bad = val_loss, 0
        else:
            bad += 1
            if bad >= 6:
                break
    model.load_state_dict(best_state)
    model.eval()

    def collect_logits(loader):
        logits, truths = [], []
        with torch.no_grad():
            for inputs, labels_batch in loader:
                logits.append(model(inputs))
                truths.append(labels_batch)
        return torch.cat(logits), torch.cat(truths)

    validation_logits, validation_truth = collect_logits(loaders['val'])
    # Calibrate on validation only. Never tune on the held-out TEST sessions.
    log_temperature = torch.tensor(0.0, requires_grad=True)
    optimizer_temp = torch.optim.LBFGS([log_temperature], lr=.1, max_iter=60)
    def objective():
        optimizer_temp.zero_grad()
        temp = torch.exp(log_temperature).clamp(.05, 20.0)
        value = nn.functional.cross_entropy(validation_logits / temp, validation_truth)
        value.backward()
        return value
    optimizer_temp.step(objective)
    temperature = float(torch.exp(log_temperature.detach()).clamp(.05, 20.0))

    test_logits, test_truth = collect_logits(loaders['test'])
    with torch.no_grad():
        probabilities = torch.softmax(test_logits / temperature, dim=1).numpy()
    report = evaluate(probabilities, test_truth.numpy(), labels)
    report.update({'source': 'real labeled gameplay crops', 'temperature': temperature,
                   'labels': labels, 'split_counts': {k: len(v) for k, v in sets.items()},
                   'independent_session_counts': {s: len(x) for s, x in inventory['sessions'].items()},
                   'threshold': ACCEPT_CONFIDENCE})
    approved = safe_to_release(report)
    report['validated'] = approved
    report['calibrated'] = approved

    # Always produce an inspection report; NEVER publish invalid model weights.
    out.mkdir(parents=True, exist_ok=True)
    (out / 'hybrid_validation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not approved:
        print('NOT APPROVED for release. Inspect hybrid_validation_report.json; Standard Mode stays active.')
        return False

    class Probabilities(nn.Module):
        def __init__(self, net, temperature):
            super().__init__()
            self.net = net
            self.register_buffer('temperature', torch.tensor(float(temperature)))
        def forward(self, image):
            return torch.softmax(self.net(image) / self.temperature, dim=1)

    package = Probabilities(model, temperature).eval()
    path = out / 'recognizer.onnx'
    torch.onnx.export(package, torch.zeros(1, 3, *INPUT_HW), str(path),
                      input_names=['image'], output_names=['probabilities'],
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    # Check that ONNX produces the same calibrated probabilities as PyTorch.
    sample_input, _ = sets['test'][0]
    ort_probs = np.asarray(sess.run(None, {'image': sample_input.numpy()[None]})[0])
    with torch.no_grad(): torch_probs = package(sample_input[None]).numpy()
    if not np.allclose(ort_probs, torch_probs, atol=2e-4):
        path.unlink(missing_ok=True)
        raise ValueError('ONNX/PyTorch outputs differ; model not packaged')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'schema_version': 1, 'validated': True, 'calibrated': True,
                'model': 'recognizer.onnx', 'model_sha256': sha,
                'labels': labels, 'input_hw': list(INPUT_HW),
                'mean': MEAN, 'std': STD,
                'validation_report': 'hybrid_validation_report.json'}
    (out / 'hybrid_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Validated shared CPU ONNX model created, with SHA-256 and independent test report:', path)
    return True


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('datasets/hybrid'))
    parser.add_argument('--out', type=Path, default=Path('models'))
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--seed', type=int, default=1729)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 200:
        parser.error('--epochs must be between 1 and 200')
    success = train(args.root, args.out, args.epochs, args.seed)
    raise SystemExit(0 if success else 2)


if __name__ == '__main__':
    cli()
