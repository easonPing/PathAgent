"""Report exact PathMMU image coverage by source and split."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import atomic_json, resolve, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Open every existing image with Pillow')
    parser.add_argument('--output', default='runs/assets/pathmmu_sources/coverage.json')
    args = parser.parse_args()

    base = resolve('data/raw/pathmmu')
    annotation = base / 'data.json'
    data = json.loads(annotation.read_text())
    image_root = base / 'images'
    results = {'annotation_sha256': sha256_file(annotation), 'sources': {},
               'test_questions': 0, 'test_images': 0, 'test_images_present': 0,
               'all_images': 0, 'all_images_present': 0}
    checked = set()
    invalid = []
    if args.verify:
        from PIL import Image
    for source, splits in data.items():
        source_rows = [r for rows in splits.values() for r in rows]
        unique = {r['img']: r for r in source_rows}
        test_rows = [r for split, rows in splits.items() if split in ('test', 'test_tiny') for r in rows]
        test_unique = {r['img']: r for r in test_rows}
        missing = []
        for name, row in sorted(unique.items()):
            path = image_root / name
            if path.is_file() and path.stat().st_size > 0:
                if args.verify and name not in checked:
                    try:
                        with Image.open(path) as image:
                            image.verify()
                    except Exception as exc:
                        invalid.append({'source': source, 'img': name, 'error': type(exc).__name__})
                    checked.add(name)
            else:
                missing.append({key: row[key] for key in
                                ('img', 'source_img', 'tw_id', 'img_position', 'ref_web') if key in row})
        missing_names = {r['img'] for r in missing}
        all_present = len(unique) - len(missing_names)
        test_present = len(test_unique) - len(test_unique.keys() & missing_names)
        results['sources'][source] = {
            'questions_by_split': {split: len(rows) for split, rows in splits.items()},
            'all_images': len(unique), 'all_images_present': all_present,
            'test_questions': len(test_rows), 'test_images': len(test_unique),
            'test_images_present': test_present, 'missing': missing}
        results['all_images'] += len(unique)
        results['all_images_present'] += all_present
        results['test_questions'] += len(test_rows)
        results['test_images'] += len(test_unique)
        results['test_images_present'] += test_present
    results['invalid'] = invalid
    atomic_json(resolve(args.output), results)
    print(json.dumps({source: {'all': r['all_images'], 'present': r['all_images_present'],
                               'test': r['test_images'], 'test_present': r['test_images_present']}
                      for source, r in results['sources'].items()}, ensure_ascii=False))
    print(f"test {results['test_images_present']}/{results['test_images']} images, "
          f"all {results['all_images_present']}/{results['all_images']} images, "
          f"invalid {len(invalid)}, report {resolve(args.output)}")
    if invalid or results['all_images_present'] != results['all_images']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
