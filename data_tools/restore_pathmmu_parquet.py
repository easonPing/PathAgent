"""Restore mapped PathMMU originals from a Parquet file with filename and image columns."""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import atomic_json, resolve, sha256_file


def main():
    from PIL import Image
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', required=True, nargs='+')
    parser.add_argument('--filename-source', choices=['column', 'image_path'], default='column')
    parser.add_argument('--source-prefix', required=True)
    parser.add_argument('--source-url', required=True)
    args = parser.parse_args()

    base = resolve('data/raw/pathmmu')
    data = json.loads((base / 'data.json').read_text())
    mapping = {}
    for source in data.values():
        for rows in source.values():
            for row in rows:
                original = row.get('source_img', '')
                if not original.startswith(args.source_prefix + '/'):
                    continue
                name = Path(original).name
                if Path(row['img']).name != row['img']:
                    raise ValueError('Unsafe destination name')
                if name in mapping and mapping[name] != (row['img'], original):
                    raise ValueError(f'Ambiguous original filename: {name}')
                mapping[name] = (row['img'], original)
    if not mapping:
        raise ValueError('No images for this source prefix')

    report_path = resolve('runs/assets/pathmmu_sources') / (args.source_prefix + '_parquet_restored.json')
    report = {'source_prefix': args.source_prefix, 'source_url': args.source_url,
              'annotation_sha256': sha256_file(base / 'data.json'),
              'restored': {}, 'already_present': [], 'missing_members': []}
    found = set()
    for parquet_path in args.parquet:
        parquet = pq.ParquetFile(resolve(parquet_path))
        for group in range(parquet.metadata.num_row_groups):
            if args.filename_source == 'column':
                filenames = parquet.read_row_group(group, columns=['filename']).column('filename').to_pylist()
                if not any(name in mapping for name in filenames):
                    continue
                rows = parquet.read_row_group(group, columns=['filename', 'image']).to_pylist()
            else:
                rows = parquet.read_row_group(group, columns=['image']).to_pylist()
            for row in rows:
                name = (row['filename'] if args.filename_source == 'column'
                        else Path(row['image']['path'] or '').name)
                if name not in mapping:
                    continue
                if name in found:
                    raise ValueError(f'Duplicate original filename in Parquet: {name}')
                found.add(name)
                image_name, original = mapping[name]
                destination = base / 'images' / image_name
                if destination.is_file():
                    report['already_present'].append(image_name)
                    continue
                content = row['image']['bytes']
                if not content:
                    raise ValueError(f'No image bytes for {name}')
                with Image.open(io.BytesIO(content)) as image:
                    image.load()
                    dimensions = list(image.size)
                temporary = destination.with_suffix(destination.suffix + '.partial')
                temporary.write_bytes(content)
                temporary.replace(destination)
                report['restored'][image_name] = {'source_img': original,
                    'parquet_filename': name, 'output_sha256': hashlib.sha256(content).hexdigest(),
                    'dimensions': dimensions}
                if len(report['restored']) % 25 == 0:
                    atomic_json(report_path, report)
                    print(args.source_prefix, 'restored', len(report['restored']), flush=True)
    report['missing_members'] = [{'image': mapping[name][0], 'source_img': mapping[name][1]}
                                 for name in sorted(mapping.keys() - found)]
    atomic_json(report_path, report)
    print({'prefix': args.source_prefix, 'restored': len(report['restored']),
           'already_present': len(report['already_present']),
           'missing_members': len(report['missing_members']), 'report': str(report_path)}, flush=True)
    if report['missing_members']:
        raise SystemExit('Some mapped originals are missing from this Parquet file; see report')


if __name__ == '__main__':
    main()
