"""Safe synthetic unit tests for the *pipeline*, NOT gameplay model validation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'developer'))
from hybrid_dataset import audit, enumerate_samples
from train_hybrid import compute_ece, evaluate, image_input, safe_to_release
from check_hybrid_model import verify


def create_sample(root, split, session, label, name, value=70):
    path = root / split / session / label / f'{name}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 64, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def test_audit_requires_real_target_and_all_splits(tmp_path):
    x = create_sample(tmp_path, 'train', 'recording_a', '__background__', 'zero')
    result = audit(tmp_path, for_release=True)
    assert result['errors']
    assert any('target labels' in s for s in result['errors'])
    assert any('val' in s for s in result['errors'])


def test_audit_rejects_cross_split_leakage(tmp_path):
    for split in ('train', 'val', 'test'):
        create_sample(tmp_path, split, 'same_session', 'retry_text', split, value={'train': 12, 'val': 13, 'test': 14}[split])
        create_sample(tmp_path, split, f'session_{split}', '__background__', split, value={'train': 48, 'val': 58, 'test': 68}[split])
    result = audit(tmp_path)
    assert any('Session reused' in err for err in result['errors'])
    assert len(result['samples']) == 6


def test_audit_rejects_same_image_in_different_groups(tmp_path):
    a = create_sample(tmp_path, 'train', 'capture_a', 'retry_text', 'first', 66)
    b = create_sample(tmp_path, 'test', 'capture_b', 'retry_text', 'second', 66)
    create_sample(tmp_path, 'val', 'capture_c', 'retry_text', 'third', 67)
    create_sample(tmp_path, 'train', 'capture_a', '__background__', 'bg1', 20)
    create_sample(tmp_path, 'val', 'capture_c', '__background__', 'bg2', 30)
    create_sample(tmp_path, 'test', 'capture_b', '__background__', 'bg3', 40)
    assert hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest()
    assert any('Duplicate image across records' in err for err in audit(tmp_path)['errors'])


def test_release_requires_real_negative_validation(tmp_path):
    create_sample(tmp_path, 'train', 'recording_a', 'retry_text', 'a')
    create_sample(tmp_path, 'val', 'recording_b', 'retry_text', 'b', 73)
    create_sample(tmp_path, 'test', 'recording_c', 'retry_text', 'c', 75)
    for split, session in [('train', 'recording_a'), ('val', 'recording_b'), ('test', 'recording_c')]:
        create_sample(tmp_path, split, session, '__background__', 'bg', {'train':23, 'val':27, 'test':31}[split])
    errors = audit(tmp_path, for_release=True)['errors']
    assert any('Insufficient real' in s for s in errors)
    assert any('hard negatives' in s for s in errors)


def test_model_release_gate_rejects_false_positives_and_uncalibrated():
    good = {'samples': 600, 'hard_negative_count': 500, 'hard_negative_false_accepts': 0,
            'ece': .02, 'classes': {'retry_text': {
                'tested_positives': 100, 'true_positive': 87,
                'false_positive': 0, 'precision': 1.0, 'recall': .87}}}
    assert safe_to_release(good)
    assert not safe_to_release({**good, 'hard_negative_false_accepts': 1})
    assert not safe_to_release({**good, 'ece': .06})
    assert not safe_to_release({**good, 'hard_negative_count': 100})
    assert not safe_to_release({**good, 'classes': {'retry_text': {**good['classes']['retry_text'], 'precision': .8}}})


def test_preprocessing_and_ece_contract():
    im = np.zeros((60, 40, 3), dtype=np.uint8)
    x = image_input(im)
    assert x.shape == (3, 96, 96)
    assert x.dtype == np.float32
    result = evaluate(np.array([[.99, .01], [.001, .999]], np.float32),
                      np.array([0, 1]), ['__background__', 'retry_text'])
    assert result['hard_negative_false_accepts'] == 0
    assert result['classes']['retry_text']['true_positive'] == 1
    assert compute_ece(np.array([[.99, .01]], np.float32), np.array([0])) == pytest.approx(.01, abs=1e-6)


def test_build_rejects_missing_and_fake_manifest(tmp_path):
    assert not verify(tmp_path)[0]
    (tmp_path / 'recognizer.onnx').write_bytes(b'FAKE')
    (tmp_path / 'hybrid_manifest.json').write_text(json.dumps({
        'schema_version': 1, 'model': 'recognizer.onnx', 'validated': True,
        'calibrated': True, 'model_sha256': 'a' * 64,
        'validation_report': 'hybrid_validation_report.json'}), encoding='utf-8')
    assert not verify(tmp_path)[0]  # bad hash
    import hashlib
    data = json.loads((tmp_path / 'hybrid_manifest.json').read_text())
    data['model_sha256'] = hashlib.sha256(b'FAKE').hexdigest()
    (tmp_path / 'hybrid_manifest.json').write_text(json.dumps(data))
    assert not verify(tmp_path)[0]  # report missing
    (tmp_path / 'hybrid_validation_report.json').write_text('{}')
    assert not verify(tmp_path)[0]  # fake evaluation


def test_runtime_rejects_bad_model_checksum(tmp_path):
    from thebums_hybrid_ai import CpuOnnxBackend
    (tmp_path / 'recognizer.onnx').write_bytes(b'invalid model')
    (tmp_path / 'hybrid_manifest.json').write_text(json.dumps({
        'schema_version': 1, 'model': 'recognizer.onnx',
        'validated': True, 'calibrated': True,
        'model_sha256': 'a' * 64,
        'labels': ['__background__', 'retry_text']}))
    with pytest.raises(ValueError, match='checksum'):
        CpuOnnxBackend(tmp_path)
