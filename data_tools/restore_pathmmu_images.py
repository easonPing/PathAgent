"""Restore missing PathMMU images from archives using source_img mappings.

HTTP ZIPs may be read by byte range. Only mapped members are read; archive paths
are never extracted. TIFF conversion follows the released construct_pathcls.py.
"""
import argparse
import contextlib
import hashlib
import io
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import atomic_json, resolve, sha256_file


class HTTPRangeReader(io.RawIOBase):
    """Seekable, bounded reader for public ZIPs served without Content-Length."""

    def __init__(self, url, block_size=2 * 1024 * 1024):
        self.url = url
        self.block_size = block_size
        self.session = requests.Session()
        retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retry))
        response = self.session.get(url, headers={'Range': 'bytes=0-0'}, timeout=120)
        response.raise_for_status()
        match = re.fullmatch(r'bytes 0-0/(\d+)', response.headers.get('Content-Range', ''))
        if response.status_code != 206 or not match or len(response.content) != 1:
            self.session.close()
            raise ValueError('Source does not support verified HTTP byte ranges')
        self.size = int(match.group(1))
        self.position = 0
        self.block_start = -1
        self.block = b''

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_END:
            offset += self.size
        elif whence == io.SEEK_CUR:
            offset += self.position
        elif whence != io.SEEK_SET:
            raise ValueError('Invalid seek mode')
        if offset < 0:
            raise ValueError('Negative seek')
        self.position = offset
        return offset

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, max(0, self.size - self.position))
        result = bytearray()
        while size:
            if not self.block_start <= self.position < self.block_start + len(self.block):
                start = self.position // self.block_size * self.block_size
                end = min(start + self.block_size, self.size) - 1
                response = self.session.get(self.url, headers={'Range': f'bytes={start}-{end}'}, timeout=120)
                response.raise_for_status()
                expected = f'bytes {start}-{end}/{self.size}'
                if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                    raise IOError(f'Unexpected HTTP range response at byte {start}')
                self.block = response.content
                self.block_start = start
                if len(self.block) != end - start + 1:
                    raise IOError(f'Short HTTP range response at byte {start}')
            offset = self.position - self.block_start
            count = min(size, len(self.block) - offset)
            result.extend(self.block[offset:offset + count])
            self.position += count
            size -= count
        return bytes(result)

    def close(self):
        self.session.close()
        super().close()


def main():
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--source-prefix', required=True)
    parser.add_argument('--source-url', help='Public provenance URL for a locally downloaded archive')
    parser.add_argument('--block-size-mib', type=int, default=2,
                        help='HTTP byte-range cache block size (use larger blocks for large images)')
    args = parser.parse_args()
    if not 1 <= args.block_size_mib <= 64:
        parser.error('--block-size-mib must be between 1 and 64')
    base = resolve('data/raw/pathmmu')
    annotation = json.loads((base / 'data.json').read_text())
    mapping = {}
    for source in annotation.values():
        for split in ['val', 'test', 'test_tiny']:
            for row in source.get(split, []):
                original = row.get('source_img', '')
                if original.startswith(args.source_prefix + '/'):
                    if Path(row['img']).name != row['img']:
                        raise ValueError('Unsafe destination name')
                    previous = mapping.setdefault(row['img'], original)
                    if previous != original:
                        raise ValueError('Conflicting image mapping')
    if not mapping:
        raise ValueError('No images for this source prefix')
    report_path = resolve('runs/assets/pathmmu_sources') / (args.source_prefix + '_restored.json')
    report = {'source_prefix': args.source_prefix,
              'source_url': args.source_url or args.archive,
              'annotation_sha256': sha256_file(base / 'data.json'),
              'conversion': 'TIFF -> RGB JPEG with Pillow defaults, as in official construct_pathcls.py',
              'restored': {}, 'already_present': [], 'missing_members': []}
    if not args.archive.startswith(('https://', 'http://')):
        report['archive_sha256'] = sha256_file(resolve(args.archive))
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if (previous['annotation_sha256'] == report['annotation_sha256']
                and previous['source_url'] == report['source_url']):
            report['restored'].update(previous['restored'])
    def restore_content(name, original, member, content):
        destination = base / 'images' / name
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            dimensions = list(image.size)
            if member.lower().endswith(('.tif', '.tiff')):
                buffer = io.BytesIO()
                image.convert('RGB').save(buffer, 'JPEG')
                output = buffer.getvalue()
            else:
                output = content
        temporary = destination.with_suffix(destination.suffix + '.partial')
        temporary.write_bytes(output)
        temporary.replace(destination)
        report['restored'][name] = {'source_img': original, 'archive_member': member,
                                   'source_sha256': hashlib.sha256(content).hexdigest(),
                                   'output_sha256': hashlib.sha256(output).hexdigest(),
                                   'dimensions': dimensions}
        if len(report['restored']) % 25 == 0:
            atomic_json(report_path, report)
            print(args.source_prefix, 'restored', len(report['restored']), flush=True)

    with contextlib.ExitStack() as stack:
        if args.archive.startswith(('https://', 'http://')):
            stream = stack.enter_context(HTTPRangeReader(args.archive,
                                        block_size=args.block_size_mib * 1024 * 1024))
        else:
            stream = stack.enter_context(resolve(args.archive).open('rb'))
        if args.archive.endswith(('.tar.gz', '.tgz')):
            found = {}
            archive = stack.enter_context(tarfile.open(fileobj=stream, mode='r|gz'))
            for member in archive:
                if not member.isfile():
                    continue
                matches = [(name, original) for name, original in mapping.items()
                           if member.name == original or member.name == original.split('/', 1)[1]
                           or member.name.endswith('/' + original)
                           or member.name.endswith('/' + original.split('/', 1)[1])]
                if len(matches) > 1:
                    raise ValueError(f'Ambiguous archive member: {member.name}')
                if matches:
                    name, original = matches[0]
                    if name in found:
                        raise ValueError(f'Duplicate archive member for {name}')
                    member_stream = archive.extractfile(member)
                    if member_stream is None:
                        raise ValueError(f'Unreadable TAR member: {member.name}')
                    with member_stream:
                        found[name] = (member.name, member_stream.read())
            for name, original in sorted(mapping.items()):
                if (base / 'images' / name).exists():
                    if name not in report['restored']:
                        report['already_present'].append(name)
                elif name in found:
                    member, content = found[name]
                    restore_content(name, original, member, content)
                else:
                    report['missing_members'].append({'image': name, 'source_img': original,
                                                      'candidates': []})
        elif args.archive.endswith('.tar'):
            archive = stack.enter_context(tarfile.open(fileobj=stream, mode='r:'))
            members = [member.name for member in archive.getmembers() if member.isfile()]
            def read_member(name):
                member_stream = archive.extractfile(name)
                if member_stream is None:
                    raise ValueError(f'Unreadable TAR member: {name}')
                with member_stream:
                    return member_stream.read()
        elif not args.archive.endswith(('.tar.gz', '.tgz')):
            archive = stack.enter_context(zipfile.ZipFile(stream))
            members = [name for name in archive.namelist() if not name.endswith('/')]
            read_member = archive.read
        if not args.archive.endswith(('.tar.gz', '.tgz')):
            for name, original in sorted(mapping.items()):
                destination = base / 'images' / name
                if destination.exists():
                    if name not in report['restored']:
                        report['already_present'].append(name)
                    continue
                relative = original.split('/', 1)[1]
                candidates = [m for m in members if m == original or m == relative
                              or m.endswith('/' + original) or m.endswith('/' + relative)]
                if len(candidates) != 1:
                    report['missing_members'].append({'image': name, 'source_img': original,
                                                      'candidates': candidates})
                    continue
                member = candidates[0]
                content = read_member(member)  # ZipFile validates ZIP members' CRC.
                restore_content(name, original, member, content)
    atomic_json(report_path, report)
    print({'prefix': args.source_prefix, 'restored': len(report['restored']),
           'already_present': len(report['already_present']),
           'missing_members': len(report['missing_members']), 'report': str(report_path)}, flush=True)
    if report['missing_members']:
        raise SystemExit('Some mapped originals are missing from this archive; see report')


if __name__ == '__main__':
    main()
