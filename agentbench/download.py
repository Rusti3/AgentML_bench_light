from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import time
import threading
from collections import defaultdict

import httpx

from .common import ROOT, read_json, sha256, write_json

_LOCKS = defaultdict(threading.RLock)


class Downloader:
    """Resumable public HTTP downloads with an immutable local receipt.

    A resumed response must match both its offset and its previous entity validator.
    Remote LFS SHA-256 / ARFF MD5 is verified when supplied; otherwise a locally
    recorded SHA-256 pins exactly the bytes received on the first successful fetch.
    """
    def __init__(self, attempts=5, timeout=120):
        self.attempts = attempts
        self.pinned_assets = read_json(ROOT/'data'/'v1'/'catalog'/'assets.lock.json', {}).get('files', {})
        self.client = httpx.Client(follow_redirects=True, timeout=httpx.Timeout(timeout, connect=30),
                                   headers={'User-Agent': 'AgentBenchmark/1.0',
                                            'Accept-Encoding': 'identity'})

    def close(self):
        self.client.close()

    def json(self, url, path):
        self.get(url, path)
        return read_json(path)

    def get(self, url, path, expected_sha=None, expected_md5=None, expected_size=None):
        try:
            key = Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            key = None
        pinned = self.pinned_assets.get(key, {})
        if pinned:
            if expected_sha and expected_sha != pinned['sha256']:
                raise RuntimeError('Upstream checksum conflicts with the published asset lock')
            expected_sha = pinned['sha256']
            expected_size = pinned['bytes']
        with _LOCKS[str(Path(path).resolve())]:
            return self._get(url, path, expected_sha, expected_md5, expected_size)

    def _get(self, url, path, expected_sha=None, expected_md5=None, expected_size=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path = path.with_name(path.name + '.receipt.json')
        receipt = read_json(receipt_path, {})
        if path.exists() and receipt.get('url') == url:
            actual = sha256(path)
            if actual == receipt.get('sha256') and (not expected_sha or actual == expected_sha):
                if expected_size is None or path.stat().st_size == expected_size:
                    return receipt
        part = path.with_name(path.name + '.part')
        partial_meta = part.with_name(part.name + '.json')
        last_error = None
        for attempt in range(self.attempts):
            try:
                if shutil.disk_usage(path.parent).free < 2 * 1024**3:
                    raise OSError('Less than 2 GiB free on the destination drive')
                meta = read_json(partial_meta, {})
                offset = part.stat().st_size if part.exists() and meta.get('url') == url else 0
                headers = {}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                    validator = meta.get('etag') or meta.get('last_modified')
                    if validator:
                        headers['If-Range'] = validator
                with self.client.stream('GET', url, headers=headers) as response:
                    if response.status_code == 416:
                        part.unlink(missing_ok=True)
                        partial_meta.unlink(missing_ok=True)
                        raise RuntimeError('Remote entity changed or invalid resume offset')
                    response.raise_for_status()
                    resumed = response.status_code == 206 and offset > 0
                    if response.status_code == 206:
                        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)', response.headers.get('content-range', ''))
                        if not match or int(match[1]) != offset:
                            part.unlink(missing_ok=True)
                            raise RuntimeError('Invalid Content-Range')
                        total = int(match[3]) if match[3] != '*' else None
                        old_validator = meta.get('etag')
                        if resumed and old_validator and response.headers.get('etag') != old_validator:
                            part.unlink(missing_ok=True)
                            raise RuntimeError('Changed ETag during resume')
                    else:
                        total = int(response.headers['content-length']) if 'content-length' in response.headers else None
                    write_json(partial_meta, {'url': url, 'etag': response.headers.get('etag'),
                                             'last_modified': response.headers.get('last-modified')})
                    with part.open('ab' if resumed else 'wb') as out:
                        for block in response.iter_bytes(1024 * 1024):
                            out.write(block)
                    if total is not None and part.stat().st_size != total:
                        raise RuntimeError('Incomplete HTTP body')
                    if expected_size is not None and part.stat().st_size != expected_size:
                        raise RuntimeError('Remote size does not match pinned manifest')
                actual_sha = sha256(part)
                if expected_sha and actual_sha != expected_sha:
                    part.unlink(missing_ok=True)
                    raise RuntimeError('SHA-256 does not match upstream manifest')
                if expected_md5:
                    h = hashlib.md5()
                    with part.open('rb') as stream:
                        for block in iter(lambda: stream.read(1024*1024), b''):
                            h.update(block)
                    if h.hexdigest() != expected_md5:
                        part.unlink(missing_ok=True)
                        raise RuntimeError('MD5 does not match upstream manifest')
                os.replace(part, path)
                receipt = {'url': url, 'sha256': actual_sha, 'bytes': path.stat().st_size,
                           'upstream_sha256': expected_sha, 'upstream_md5': expected_md5,
                           'etag': read_json(partial_meta, {}).get('etag')}
                write_json(receipt_path, receipt)
                partial_meta.unlink(missing_ok=True)
                return receipt
            except (httpx.HTTPError, OSError, RuntimeError) as exc:
                last_error = exc
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in [401, 403, 404]:
                    break
                if attempt + 1 < self.attempts:
                    time.sleep(min(2**attempt, 15))
        raise RuntimeError(f'Download failed: {url}: {last_error}') from last_error
