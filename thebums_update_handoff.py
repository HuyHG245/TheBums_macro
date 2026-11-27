"""Local updater process handoff; no network and no executable replacement here."""
import os
import ntpath
import sys
import subprocess
import time
from pathlib import Path


class HandoffError(RuntimeError):
    pass


def clean_child_environment(environ, bundle=None):
    """A restarted one-file app must not reuse the exiting bootloader's state."""
    env = {k: v for k, v in environ.items()
           if not k.upper().startswith('_PYI_') and
           k.upper() not in ('_MEIPASS2', 'PYINSTALLER_RESET_ENVIRONMENT',
                             'THEBUMS_UPDATE_HEALTH')}
    if bundle:
        root = ntpath.normcase(ntpath.normpath(str(bundle)))
        for key in env:
            if key.upper() != 'PATH':
                continue
            # Some runtime hooks prepend bundled DLL directories to PATH.
            def external(entry):
                try:
                    candidate = ntpath.normcase(ntpath.normpath(entry.strip('"')))
                    return ntpath.commonpath([root, candidate]) != root
                except ValueError:
                    return True
            env[key] = ';'.join(p for p in env[key].split(';') if external(p))
    return env


def start_windows_helper(argv, log_path, *, cwd):
    """Launch the system PowerShell outside PyInstaller's modified DLL search path."""
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetSystemDirectoryW.argtypes = [wintypes.LPWSTR, wintypes.UINT]
    kernel.GetSystemDirectoryW.restype = wintypes.UINT
    kernel.GetDllDirectoryW.argtypes = [wintypes.DWORD, wintypes.LPWSTR]
    kernel.GetDllDirectoryW.restype = wintypes.DWORD
    kernel.SetDllDirectoryW.argtypes = [wintypes.LPCWSTR]
    kernel.SetDllDirectoryW.restype = wintypes.BOOL
    system = ctypes.create_unicode_buffer(32768)
    size = kernel.GetSystemDirectoryW(system, len(system))
    if not size or size >= len(system):
        raise ctypes.WinError(ctypes.get_last_error())
    powershell = Path(system.value) / 'WindowsPowerShell' / 'v1.0' / 'powershell.exe'
    if not powershell.is_file():
        raise HandoffError('Windows PowerShell was not found in the system directory')
    previous = ctypes.create_unicode_buffer(32768)
    size = kernel.GetDllDirectoryW(len(previous), previous)
    if size >= len(previous):
        raise HandoffError('DLL search path too long')
    if not kernel.SetDllDirectoryW(None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        with open(log_path, 'ab', buffering=0) as output:
            return subprocess.Popen([str(powershell), '-NoProfile', '-NonInteractive',
                                     '-ExecutionPolicy', 'Bypass', *argv],
                                    cwd=cwd, env=clean_child_environment(os.environ, getattr(sys, '_MEIPASS', None)),
                                    stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                    creationflags=subprocess.CREATE_NO_WINDOW, close_fds=True)
    finally:
        kernel.SetDllDirectoryW(previous.value if size else None)


def wait_until_ready(process, ready_file, *, timeout=12.0, clock=time.monotonic,
                     sleep=time.sleep):
    """Only a fresh, per-transaction marker from the actual helper can authorize exit."""
    deadline = clock() + timeout
    marker = Path(ready_file)
    while clock() < deadline:
        code = process.poll()
        if code is not None:
            raise HandoffError(f'Updater exited before ready (exit code {code})')
        try:
            if marker.read_text(encoding='utf-8-sig').strip() == str(process.pid):
                if process.poll() is None:
                    return
                raise HandoffError('Updater exited immediately after ready')
        except (OSError, UnicodeError):
            pass
        sleep(.05)
    raise HandoffError('Updater did not acknowledge readiness before timeout')


def authorize_install(process, commit_file):
    """The helper will never replace the executable unless this marker is published."""
    if process.poll() is not None:
        raise HandoffError('Updater exited before install authorization')
    marker = Path(commit_file)
    with marker.open('x', encoding='ascii') as f:
        f.write(str(process.pid))
    if process.poll() is not None:
        raise HandoffError('Updater exited during install authorization')


def cancel_uncommitted(process, commit_file):
    """Revoke consent then stop only the helper we created; never kill another process."""
    try:
        Path(commit_file).unlink(missing_ok=True)
    except OSError:
        pass
    if process is None:
        return True
    try:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        return process.poll() is not None
    except (OSError, subprocess.TimeoutExpired):
        return False
