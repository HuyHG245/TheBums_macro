"""Offline integration and failure-path tests. Real PowerShell/EXE replacement is Windows-only."""
import ast
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import shutil
import tempfile
import uuid
import subprocess
import pytest
from thebums_update_handoff import (clean_child_environment, wait_until_ready,
    authorize_install, cancel_uncommitted, HandoffError)

ROOT=Path(__file__).resolve().parents[1]
TREE=ast.parse((ROOT/'main.py').read_text(encoding='utf-8'))

def test_environment_isolated():
    source={'_PYI_APPLICATION_HOME_DIR':'old','_PYI_PARENT_PROCESS_LEVEL':'1',
        '_pyi_archive_file':'old.exe','_MEIPASS2':'old','THEBUMS_UPDATE_HEALTH':'old',
        'PYINSTALLER_RESET_ENVIRONMENT':'0','SystemRoot':'C:\\Windows','PATH':'path'}
    assert clean_child_environment(source)=={'SystemRoot':'C:\\Windows','PATH':'path'}
    assert len(source)==8

class FakeProcess:
    pid=12345
    def __init__(self): self.returncode=None; self.terminated=False
    def poll(self):return self.returncode
    def terminate(self):self.terminated=True;self.returncode=1
    def wait(self,timeout):return self.returncode

def fake_clock():
    now=[0.0]
    return lambda:now[0],lambda n:now.__setitem__(0,now[0]+n)

def test_acknowledge_then_authorize(tmp_path):
    proc=FakeProcess(); ready=tmp_path/'ready'; commit=tmp_path/'commit'
    ready.write_text(str(proc.pid));wait_until_ready(proc,ready)
    assert not commit.exists();authorize_install(proc,commit)
    assert commit.read_text()==str(proc.pid)

@pytest.mark.parametrize('marker',[None,'9999','', '123'])
def test_missing_or_wrong_pid_times_out(tmp_path,marker):
    proc=FakeProcess();ready=tmp_path/'ready'
    if marker is not None:ready.write_text(marker)
    clock,sleep=fake_clock()
    with pytest.raises(HandoffError,match='timeout'):
        wait_until_ready(proc,ready,timeout=.2,clock=clock,sleep=sleep)

def test_exit_before_ready(tmp_path):
    proc=FakeProcess();proc.returncode=2
    with pytest.raises(HandoffError,match='exit code 2'):
        wait_until_ready(proc,tmp_path/'ready')

def test_exit_even_with_ready(tmp_path):
    proc=FakeProcess();proc.returncode=0;ready=tmp_path/'ready';ready.write_text(str(proc.pid))
    with pytest.raises(HandoffError):wait_until_ready(proc,ready)

def test_exit_before_authorize(tmp_path):
    proc=FakeProcess();proc.returncode=1;commit=tmp_path/'commit'
    with pytest.raises(HandoffError):authorize_install(proc,commit)
    assert not commit.exists()

def test_authorize_refuses_existing_marker(tmp_path):
    commit=tmp_path/'commit';commit.write_text('old')
    with pytest.raises(FileExistsError):authorize_install(FakeProcess(),commit)
    assert commit.read_text()=='old'

def test_cancel_removes_consent_and_stops_helper(tmp_path):
    proc=FakeProcess();commit=tmp_path/'commit';commit.write_text(str(proc.pid))
    assert cancel_uncommitted(proc,commit) and proc.terminated and not commit.exists()

def test_cancel_unconfirmed_helper(tmp_path):
    proc=FakeProcess();proc.terminate=Mock(side_effect=PermissionError())
    assert not cancel_uncommitted(proc,tmp_path/'commit')

@pytest.fixture
def integration(tmp_path,monkeypatch):
    bundle=tmp_path/'bundle';bundle.mkdir();(bundle/'updater.ps1').write_text('helper')
    exe=tmp_path/'macro.exe';exe.write_bytes(b'old');staged=tmp_path/'macro.exe.new';staged.write_bytes(b'new')
    lock=tmp_path/'macro.exe.updating';lock.write_text(str(os.getpid()))
    monkeypatch.setattr(tempfile,'gettempdir',lambda:str(tmp_path))
    proc=FakeProcess(); events=[]; order=[]
    update={'staged':str(staged),'release':{'tag':'v1.4.3','digest':'sha256:'+'a'*64},'lockfile':str(lock),'prompted':True}
    def launch(argv,log_path,*,cwd):
        order.append('launch'); assert Path(argv[1]).is_file(); assert cwd==str(tmp_path)
        Path(argv[argv.index('-ReadyFile')+1]).write_text(str(proc.pid));return proc
    def ready(p,path):order.append('ready');wait_until_ready(p,path)
    def consent(p,path):order.append('authorize');authorize_install(p,path)
    ns=dict(os=os,sys=SimpleNamespace(executable=str(exe),_MEIPASS=str(bundle)),
        app_dir=lambda:str(tmp_path),data_dir=lambda:str(tmp_path),uuid=uuid,tempfile=tempfile,shutil=shutil,
        _update=update,_update_install_guard=threading.Lock(),can_install_update=lambda:True,
        supports_exe_update=lambda:True,start_windows_helper=launch,wait_until_ready=ready,
        authorize_install=consent,cancel_uncommitted=cancel_uncommitted,log=Mock(),ev=SimpleNamespace(put=events.append))
    node=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='launch_staged_update')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'main.py','exec'),ns)
    return ns,proc,events,order,exe,staged,lock

def test_main_closes_only_after_ack_and_consent(integration):
    ns,proc,events,order,exe,staged,lock=integration
    assert ns['launch_staged_update']()
    assert order==['launch','ready','authorize'] and events==[('update_ready',)]
    assert ns['_update']['install_launched'] and exe.read_bytes()==b'old' and staged.exists() and lock.exists()

@pytest.mark.parametrize('stage',['launch','ready','authorize'])
def test_main_keeps_download_on_handoff_failure(integration,stage):
    ns,proc,events,order,exe,staged,lock=integration
    key={'launch':'start_windows_helper','ready':'wait_until_ready','authorize':'authorize_install'}[stage]
    ns[key]=Mock(side_effect=HandoffError('failure'))
    assert not ns['launch_staged_update']()
    assert not ns['_update']['install_launched'] and not ns['_update']['prompted']
    assert events[0][0]=='update_handoff_failed' and ('update_ready',) not in events
    assert exe.read_bytes()==b'old' and staged.read_bytes()==b'new' and lock.exists()
    assert proc.terminated==(stage!='launch')

def test_main_rechecks_stop_before_authorize(integration):
    ns,proc,events,order,exe,staged,lock=integration
    ns['can_install_update']=Mock(side_effect=[True,True,False])
    assert not ns['launch_staged_update']()
    assert 'authorize' not in order and proc.terminated and lock.exists()

def test_main_refuses_second_install_if_cancel_uncertain(integration):
    ns,proc,events,order,exe,staged,lock=integration
    ns['wait_until_ready']=Mock(side_effect=HandoffError('timeout'))
    proc.terminate=Mock(side_effect=PermissionError())
    assert not ns['launch_staged_update']()
    assert ns['_update']['handoff_uncertain'] and events[0][-1] is True
    assert not ns['launch_staged_update']() and order==['launch'] and lock.exists()

def test_helper_orders_authorization_parent_exit_digest_replacement():
    ps=(ROOT/'updater.ps1').read_text()
    indices=[ps.index(x) for x in ('WriteAllText($ReadyFile', '$authorized = $true; break',
        '$parentProcess.WaitForExit', 'Get-FileHash', 'Copy-Item -LiteralPath $CurrentExe',
        'Move-Item -LiteralPath $StagedExe', '$started = Start-Macro $true')]
    assert indices==sorted(indices)
    assert 'if (-not $authorized) { exit 1 }' in ps
    assert 'if ($authorized -and $parentExited)' in ps
    assert "$env:PYINSTALLER_RESET_ENVIRONMENT = '1'" in ps


def test_bundle_path_entries_removed():
    bundle=r'C:\Users\user\AppData\Local\Temp\_MEI123'
    env=clean_child_environment({'Path':bundle+';'+bundle+r'\cv2;C:\Windows;D:\Tools'},bundle)
    assert env['Path']==r'C:\Windows;D:\Tools'


@pytest.mark.skipif(os.name!='nt',reason='Real Windows PowerShell handoff required')
def test_windows_real_helper_ready_and_cancel(tmp_path):
    """Runs the actual bundled script, without ever authorizing EXE replacement."""
    from thebums_update_handoff import start_windows_helper
    exe=tmp_path/'macro.exe';exe.write_bytes(b'untouched old')
    staged=tmp_path/'macro.exe.new';staged.write_bytes(b'uninstalled new')
    lock=tmp_path/'macro.exe.updating';lock.write_text(str(os.getpid()))
    health=tmp_path/('update_health_'+uuid.uuid4().hex+'.ok')
    ready=Path(str(health)+'.ready');commit=Path(str(health)+'.commit')
    log=tmp_path/'helper.log'
    proc=start_windows_helper(['-File',str(ROOT/'updater.ps1'),'-CurrentExe',str(exe),
        '-StagedExe',str(staged),'-HealthFile',str(health),'-StatusFile',str(tmp_path/'status.json'),
        '-ReleaseTag','v1.4.3-test','-Digest','sha256:'+'0'*64,'-ParentPid',str(os.getpid()),
        '-ReadyFile',str(ready),'-CommitFile',str(commit)],log,cwd=str(tmp_path))
    try:
        wait_until_ready(proc,ready)
        assert not commit.exists()
    finally:
        assert cancel_uncommitted(proc,commit)
    assert exe.read_bytes()==b'untouched old' and staged.exists() and lock.exists()
