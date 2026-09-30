"""Freeze 1,000 BCNB questions, proportional across task × semantic answer strata.

Preparation only: never loads a model, reads image pixels or submits jobs.
Existing output is immutable; reruns check its pinned identity and reuse it.
"""
import csv
import io
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import digest, resolve, sha256_file, write_manifest
from data_processing.datasets import load_bcnb, summarize

SOURCE = 'data/raw/bcnb/SlideBench-VQA-BCNB.csv'
SOURCE_SHA256 = '10b83c6177bd13bee9471fb0bd54b7d500e7eccdb7e52b462424ed50cb259e0b'
OUTPUT = 'data/raw/bcnb_1000_sample'
SEED = 128
SIZE = 1000


def stratified_sample(rows, size=SIZE, seed=SEED):
    groups = defaultdict(list)
    for row in rows:
        # Option letters are shuffled between questions; stratify by option text.
        groups[(row['Task'], row[row['Answer'].strip()].strip())].append(row)
    if not len(groups) <= size <= len(rows):
        raise ValueError('Sample size cannot cover every stratum or exceeds source')
    if len({r['ID'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source question IDs')
    keys = sorted(groups)
    # Proportional quotas with a lower bound of one; deterministic largest deficit.
    quotas = {k: max(1, size * len(groups[k]) // len(rows)) for k in keys}
    while sum(quotas.values()) < size:
        k = max((k for k in keys if quotas[k] < len(groups[k])),
                key=lambda k: (size * len(groups[k]) - quotas[k] * len(rows), k))
        quotas[k] += 1
    while sum(quotas.values()) > size:
        k = max((k for k in keys if quotas[k] > 1),
                key=lambda k: (quotas[k] * len(rows) - size * len(groups[k]), k))
        quotas[k] -= 1
    selected = [r for k in keys for r in sorted(groups[k], key=lambda r: (digest([seed, r['ID']]), r['ID']))[:quotas[k]]]
    selected.sort(key=lambda r: int(r['ID']))
    strata = [{'task': k[0], 'answer_text': k[1], 'source_count': len(groups[k]),
               'sample_count': quotas[k]} for k in keys]
    return selected, strata


def main():
    source, output = resolve(SOURCE), resolve(OUTPUT)
    if sha256_file(source) != SOURCE_SHA256:
        raise ValueError('BCNB source differs from the pinned release')
    with source.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    selected, strata = stratified_sample(rows)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(selected)
    csv_bytes = buffer.getvalue().encode('utf-8')
    import hashlib
    lock = {'schema_version': 1, 'source': SOURCE, 'source_sha256': SOURCE_SHA256,
            'seed': SEED, 'questions': SIZE,
            'method': 'proportional task × semantic answer; minimum one; largest deficit; SHA256(seed, ID) ranking',
            'sample_ids': ['bcnb:' + r['ID'] for r in selected], 'strata': strata,
            'annotation_sha256': hashlib.sha256(csv_bytes).hexdigest()}
    if output.exists():
        existing = json.loads((output / 'sampling.json').read_text())
        if existing != lock or (output / 'SlideBench-VQA-BCNB.csv').read_bytes() != csv_bytes:
            raise ValueError('Frozen sample differs; refusing to overwrite')
        manifest = json.loads((output / 'manifest.json').read_text())
        if manifest['samples_hash'] != digest(manifest['samples']):
            raise ValueError('Frozen manifest changed')
        print(f'Reusing frozen sample: {output}')
        return
    # Resolve original paths, without decoding or modifying image pixels.
    original_samples = load_bcnb(source, 'data/raw/bcnb/images', assumed_mpp=0.5)
    by_id = {r['sample_id']: r for r in original_samples}
    samples = [by_id[sid] for sid in lock['sample_ids']]
    output.mkdir(parents=True)
    images = output / 'images'
    images.mkdir()
    import os
    for sample in samples:
        original = Path(sample['image_path'])
        target = images / original.name
        if not target.exists():
            target.symlink_to(os.path.relpath(original, images))
        sample['image_path'] = str(target)
    (output / 'SlideBench-VQA-BCNB.csv').write_bytes(csv_bytes)
    (output / 'sampling.json').write_text(json.dumps(lock, ensure_ascii=False, indent=2) + '\n')
    write_manifest(output / 'manifest.json', samples, dataset='bcnb', purpose='fixed_stratified_sample',
                   annotation_sha256=lock['annotation_sha256'], source_annotation_sha256=SOURCE_SHA256,
                   seed=SEED, summary=summarize(samples), images_verified=False,
                   note='Original image paths exist; pixel/scale validation and inference have not been run.')
    print(json.dumps({'output': str(output), **summarize(samples), 'strata': len(strata)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
