"""Bounded, cancellable release download; never installs or executes the asset."""
import hashlib
import http.client
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

MAX_SIZE = 500_000_000
HOSTS = {'github.com', 'objects.githubusercontent.com', 'release-assets.githubusercontent.com'}
BACKOFF = (2, 4, 8, 16, 32)

class DownloadCancelled(Exception):
    pass

class DownloadIntegrityError(Exception):
    pass

class DownloadNetworkError(Exception):
    pass


class DownloadDeadlineExceeded(DownloadNetworkError):
    pass


def _trusted(url):
    p = urllib.parse.urlsplit(url)
    if (p.scheme != 'https' or p.hostname not in HOSTS or p.username or p.password
            or p.port not in (None, 443)):
        raise DownloadIntegrityError('Unexpected update download URL')


class _TrustedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _trusted(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _retryable(exc):
    if isinstance(exc, DownloadDeadlineExceeded):
        return False
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (408, 429, 500, 502, 503, 504)
    if isinstance(exc, urllib.error.URLError):
        return not isinstance(exc.reason, ssl.SSLError)
    return isinstance(exc, (ConnectionError, TimeoutError, http.client.IncompleteRead,
                            http.client.RemoteDisconnected, DownloadNetworkError))


def download_verified(info, destination, *, repo, cancel, progress=lambda *a: None,
                      status=lambda *a: None, timeout=15, backoff=BACKOFF,
                      opener=None, clock=time.monotonic, max_duration=600):
    """Resume within this download only. Stage atomically after complete verification.

    Progress callback: downloaded bytes, total bytes, bytes/sec, ETA seconds.
    Status callback: human-readable retry message. No credentials or signed URLs logged.
    """
    url = info['url']; _trusted(url)
    if not url.startswith('https://github.com/'+repo+'/releases/download/'):
        raise DownloadIntegrityError('Unexpected release asset path')
    digest = info.get('digest', '')
    if not isinstance(digest, str) or not re.fullmatch(r'sha256:[0-9a-fA-F]{64}', digest):
        raise DownloadIntegrityError('Missing or invalid SHA-256 digest')
    total = info.get('size')
    if not isinstance(total, int) or isinstance(total, bool) or not 1_000_000 <= total <= MAX_SIZE:
        raise DownloadIntegrityError('Unexpected release file size')
    if opener is None:
        opener = urllib.request.build_opener(_TrustedRedirect()).open
    part = destination+'.part'
    deadline = clock() + max_duration
    def check_cancel():
        if cancel.is_set():
            raise DownloadCancelled('Download cancelled')
        if clock() >= deadline:
            raise DownloadDeadlineExceeded('Total update download time limit reached')
    def discard():
        try: os.remove(part)
        except FileNotFoundError: pass
    discard()  # do not trust a partial file from an earlier process or release
    try:
        for attempt in range(len(backoff)+1):
            check_cancel()
            offset = os.path.getsize(part) if os.path.exists(part) else 0
            try:
                headers = {'User-Agent':'thebums-macro-updater', 'Accept-Encoding':'identity'}
                if offset: headers['Range'] = f'bytes={offset}-'
                # Always request the original URL to obtain a fresh signed CDN redirect.
                req = urllib.request.Request(url, headers=headers)
                with opener(req, timeout=min(timeout, max(.001, deadline-clock()))) as response:
                    _trusted(response.geturl()); check_cancel()
                    code = response.getcode()
                    encoding = response.headers.get('Content-Encoding','identity').lower()
                    if encoding != 'identity':
                        raise DownloadIntegrityError('Unexpected content encoding')
                    if code == 206 and offset:
                        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range',''))
                        if not match or tuple(map(int, match.groups())) != (offset, total-1, total):
                            raise DownloadIntegrityError('Invalid Content-Range; refusing to append')
                        mode = 'ab'
                    elif code == 200:
                        if offset: status('Server did not support resume; restarting download')
                        offset = 0; mode = 'wb'
                    else:
                        raise DownloadIntegrityError(f'Unexpected download response HTTP {code}')
                    length = response.headers.get('Content-Length')
                    if length is not None and (not length.isdecimal() or int(length) != total-offset):
                        raise DownloadIntegrityError('Unexpected response length')
                    received = offset; started = last = clock()
                    read = getattr(response, "read1", response.read)
                    progress(received, total, 0., None)
                    with open(part,mode) as out:
                        while received < total:
                            check_cancel()
                            chunk = read(min(131072,total-received+1))
                            check_cancel()
                            if not chunk: break
                            if received+len(chunk)>total:
                                raise DownloadIntegrityError('Update exceeds release size')
                            out.write(chunk); received += len(chunk)
                            now = clock()
                            if now-last >= 1 or received == total:
                                speed = (received-offset)/max(now-started,.001)
                                progress(received,total,speed,(total-received)/speed if speed else None)
                                last = now
                        if received != total:
                            raise DownloadNetworkError('Connection ended before full asset arrived')
                        check_cancel()
                        if read(1):
                            raise DownloadIntegrityError('Update exceeds release size')
                break
            except Exception as exc:
                if not _retryable(exc): raise
                if attempt == len(backoff):
                    raise DownloadNetworkError(f'Download failed after {attempt+1} attempts ({type(exc).__name__})') from exc
                # If the last read failed after all bytes arrived, restart rather than request EOF.
                if os.path.exists(part) and os.path.getsize(part) >= total: discard()
                check_cancel()
                wait = min(backoff[attempt], max(0., deadline-clock()))
                status(f'Network interrupted ({type(exc).__name__}); retry {attempt+1}/{len(backoff)} in {wait}s')
                if cancel.wait(wait): raise DownloadCancelled('Download cancelled')
        check_cancel()
        sha = hashlib.sha256()
        with open(part,'rb') as stream:
            if stream.read(2) != b'MZ': raise DownloadIntegrityError('Downloaded asset is not a Windows EXE')
            stream.seek(0)
            for chunk in iter(lambda:stream.read(131072),b''):
                check_cancel(); sha.update(chunk)
        if os.path.getsize(part) != total or sha.hexdigest().lower() != digest[7:].lower():
            raise DownloadIntegrityError('SHA-256 or size mismatch')
        check_cancel()
        os.replace(part,destination)
    finally:
        discard()
