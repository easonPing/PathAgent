"""Restore PathMMU SocialPath images from their original public X post pages.

The official mapping gives a post ID and a 1-based image position for each
image. X's public post HTML currently exposes ordered photo media URLs. This
tool only accepts media attached to the exact post ID and records unavailable
posts instead of substituting another image.
"""

import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
import re
import sys
import threading
from pathlib import Path

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import atomic_json, resolve, sha256_file


_thread_state = threading.local()


def session():
    if not hasattr(_thread_state, 'session'):
        client = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                      respect_retry_after_header=True)
        client.mount('https://', HTTPAdapter(max_retries=retry))
        client.headers['User-Agent'] = 'Mozilla/5.0 (compatible; PathMMU-research-restore/1.0)'
        _thread_state.session = client
    return _thread_state.session


def media_urls(html, post_id):
    marker = base64.b64encode(f'Tweet:{post_id}'.encode()).decode()
    expression = (r'__id:"client:' + re.escape(marker)
                  + r':media_entities2:(\d+)",__typename:"ApiMediaEntity",'
                  + r'media_url_https:"(https://pbs\.twimg\.com/media/[^"]+)",type:"photo"')
    found = re.findall(expression, html)
    photos = {}
    for index, url in found:
        index = int(index)
        if index in photos and photos[index] != url:
            raise ValueError(f'Conflicting photo URL for position {index}')
        photos[index] = url
    return photos


def restore_one(item, image_root):
    name, post_id, position = item
    destination = image_root / name
    if destination.exists():
        return name, 'already_present', None
    try:
        page_url = f'https://x.com/i/status/{post_id}'
        response = session().get(page_url, timeout=30)
        response.raise_for_status()
        photos = media_urls(response.text, post_id)
        url = photos.get(position - 1)
        if not url:
            return name, 'missing', {'post_id': post_id, 'position': position,
                                     'reason': f'Post page exposed {len(photos)} photos'}
        image_url = url + ('&' if '?' in url else '?') + 'name=orig'
        response = session().get(image_url, timeout=60)
        response.raise_for_status()
        content = response.content
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            image_format = image.format
            dimensions = list(image.size)
        temporary = destination.with_suffix(destination.suffix + '.partial')
        temporary.write_bytes(content)
        temporary.replace(destination)
        return name, 'restored', {'post_id': post_id, 'position': position,
                                  'post_url': page_url,
                                  'media_url': image_url,
                                  'sha256': hashlib.sha256(content).hexdigest(),
                                  'format': image_format, 'dimensions': dimensions}
    except (requests.RequestException, OSError, ValueError) as error:
        return name, 'missing', {'post_id': post_id, 'position': position,
                                 'reason': f'{type(error).__name__}: {error}'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--limit', type=int, help='Restore only this many images for a trial')
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error('--workers must be between 1 and 12')
    base = resolve('data/raw/pathmmu')
    mapping_path = base / 'socialpath_mapping.json'
    mapping = json.loads(mapping_path.read_text())
    items = {}
    for rows in mapping.values():
        for row in rows:
            name = row['img']
            post_id = str(row['tw_id'])
            position = int(row['img_position'])
            if Path(name).name != name or not post_id.isdigit() or not 1 <= position <= 4:
                raise ValueError(f'Invalid SocialPath mapping: {name}')
            value = (name, post_id, position)
            if name in items and items[name] != value:
                raise ValueError(f'Conflicting SocialPath mapping: {name}')
            items[name] = value
    annotation = json.loads((base / 'data.json').read_text())
    expected = {row['img'] for rows in annotation['SocialPath'].values() for row in rows}
    if set(items) != expected:
        raise ValueError('SocialPath mapping does not match data.json')
    report_path = resolve('runs/assets/pathmmu_sources/SocialPath_restored.json')
    report = {'source': 'Original public X post pages',
              'mapping_sha256': sha256_file(mapping_path),
              'restored': {}, 'already_present': [], 'missing': {}}
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get('mapping_sha256') == report['mapping_sha256']:
            report['restored'].update(previous.get('restored', {}))
    image_root = base / 'images'
    selected = list(items.values())[:args.limit]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(restore_one, item, image_root) for item in selected]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name, status, detail = future.result()
            if status == 'restored':
                report['restored'][name] = detail
            elif status == 'already_present':
                if name not in report['restored']:
                    report['already_present'].append(name)
            else:
                report['missing'][name] = detail
            if index % 50 == 0:
                atomic_json(report_path, report)
                print(f'processed {index}/{len(selected)}; restored {len(report["restored"])}; '
                      f'missing {len(report["missing"])}', flush=True)
    atomic_json(report_path, report)
    print({'processed': len(selected), 'restored': len(report['restored']),
           'missing': len(report['missing']), 'report': str(report_path)}, flush=True)


if __name__ == '__main__':
    main()
