"""Read-only inventory of reproducibility prerequisites; never fabricates smoke success."""
import argparse
import importlib.metadata
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import atomic_json, code_hash, digest, load_config, read_json, resolve
from data_processing.datasets import load_benchmark, summarize


def main():
    import yaml
    from huggingface_hub import get_token
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/preflight.json')
    args = parser.parse_args()
    config = load_config()
    report = {'code_hash': code_hash(), 'config_hash': digest(config),
              'hf_login_present': bool(get_token()), 'models': {}, 'datasets': {}, 'blockers': [],
              'scope': 'asset readiness only; this check does not perform inference or submit jobs'}
    for role, model in config['models'].items():
        directory = resolve(model['path'])
        try:
            lock = read_json(directory / 'asset_lock.json')
            ready = lock['complete'] and lock['revision'] == model['revision']
            ready = ready and all((directory / name).is_file() and (directory / name).stat().st_size == item['bytes']
                                  for name, item in lock['files'].items())
        except (FileNotFoundError, KeyError):
            ready = False
        report['models'][role] = {'ready': ready, 'repo': model['repo'], 'revision': model['revision']}
        if not ready:
            report['blockers'].append(f'Model missing/incomplete: {role}')
    for dataset in ['pathmmu', 'bcnb', 'wsi_vqa']:
        cfg = yaml.safe_load(resolve(f'configs/datasets/{dataset}.yaml').read_text())
        record = {}
        try:
            rows = load_benchmark(cfg, require_images=False)
            record['annotations'] = summarize(rows)
            if dataset == 'wsi_vqa':
                from scripts.build_manifest import select_smoke
                smoke = select_smoke(rows, dataset)
                record['smoke_images_ready'] = True
                record['smoke'] = summarize(smoke)
            else:
                load_benchmark(cfg, require_images=True)
                record['full_images_ready'] = True
        except (OSError, ValueError, KeyError) as exc:
            record['error'] = str(exc)
            report['blockers'].append(f'{dataset}: {exc}')
        report['datasets'][dataset] = record
    report['environment'] = {'python': sys.version, 'packages': {}}
    for package in ['torch', 'torchvision', 'transformers', 'numpy', 'openslide-python', 'h5py', 'pycocoevalcap']:
        try:
            report['environment']['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['blockers'].append('Package missing: ' + package)
    report['assets_ready'] = not report['blockers']
    atomic_json(resolve(args.output), report)
    print('Asset preflight:', 'READY' if report['assets_ready'] else 'BLOCKED')
    for blocker in report['blockers']:
        print('- ' + blocker)
    print(resolve(args.output))
    if report['blockers']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
