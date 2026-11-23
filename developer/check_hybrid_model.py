"""Lightweight build-time gate for a public shared ONNX Hybrid AI model.

No machine-learning dependencies: models are accepted only when a validation
report exists, sha256 matches, and the independently held-out evaluation passed.
This is packaging validation, NOT a substitute for doing the evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(root: Path):
    manifest_path = Path(root) / 'hybrid_manifest.json'
    if not manifest_path.is_file():
        return False, 'No shared AI model included in this source version.'
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('schema_version') != 1 or not manifest.get('validated') or not manifest.get('calibrated'):
            raise ValueError('Model is not validated/calibrated')
        model_name = manifest.get('model')
        if (not isinstance(model_name, str) or Path(model_name).name != model_name
                or not model_name.endswith('.onnx')):
            raise ValueError('Model path invalid')
        path = Path(root) / model_name
        expected = manifest.get('model_sha256')
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError('Release model must have SHA-256 checksum')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('ONNX file missing or checksum mismatch')
        report_name = manifest.get('validation_report', '')
        if not isinstance(report_name, str) or Path(report_name).name != report_name:
            raise ValueError('Validation report path invalid')
        report_path = Path(root) / report_name
        if not report_path.is_file():
            raise ValueError('Validation report missing')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        from train_hybrid import safe_to_release
        if not report.get('validated') or not report.get('calibrated') or not safe_to_release(report):
            raise ValueError('Held-out test results do not qualify for release')
        if report.get('labels') != manifest.get('labels'):
            raise ValueError('Manifest labels differ from independent test report')
        return True, 'Shared model and held-out validation report accepted.'
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, f'Unsafe/unverified Hybrid AI model: {exc}'


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', type=Path, default=Path('models'))
    args = p.parse_args()
    ok, message = verify(args.models)
    print(message)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    cli()
