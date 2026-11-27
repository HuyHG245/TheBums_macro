import hashlib
import io
import threading
import urllib.error
import pytest
from thebums_update_download import download_verified, DownloadCancelled, DownloadIntegrityError, DownloadNetworkError

REPO='HuyHG245/TheBums_macro'
URL='https://github.com/'+REPO+'/releases/download/v1.4.1/TheBumsMacro.exe'
CDN='https://release-assets.githubusercontent.com/file'
DATA=b'MZ'+b'x'*1_200_000
INFO=dict(url=URL,size=len(DATA),digest='sha256:'+hashlib.sha256(DATA).hexdigest())

class Response:
    def __init__(self,data=DATA,code=200,headers=None,fail=False,url=CDN):
        self.data=io.BytesIO(data); self.code=code; self.headers=headers or {}; self.fail=fail;self.url=url;self.calls=0
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def geturl(self):return self.url
    def getcode(self):return self.code
    def read(self,n):
        self.calls+=1
        if self.fail and self.calls==2:raise ConnectionResetError(10054,'remote reset')
        return self.data.read(n)

@pytest.fixture
def env(tmp_path):
    dest=str(tmp_path/'macro.exe.new'); cancel=threading.Event()
    def run(open_,info=INFO,**kw):
        return download_verified(info,dest,repo=REPO,cancel=cancel,opener=open_,backoff=(0,0),**kw)
    return tmp_path,cancel,run

def test_success(env):
    p,c,run=env; events=[];run(lambda *a,**k:Response(),progress=lambda *a:events.append(a))
    assert (p/'macro.exe.new').read_bytes()==DATA and events[-1][:2]==(len(DATA),len(DATA))

def test_winerror_before_response(env):
    p,c,run=env; calls=[]
    def op(req,**kw):
        calls.append(req)
        if len(calls)==1:raise urllib.error.URLError(ConnectionResetError(10054,'reset'))
        return Response()
    run(op);assert len(calls)==2

@pytest.mark.parametrize('resume',[True,False])
def test_interruption(env,resume):
    p,c,run=env; calls=[]; offset=131072
    def op(req,**kw):
        calls.append(req)
        if len(calls)==1:return Response(fail=True)
        assert req.get_header('Range')==f'bytes={offset}-'
        if resume:return Response(DATA[offset:],206,{'Content-Range':f'bytes {offset}-{len(DATA)-1}/{len(DATA)}'})
        return Response()
    run(op);assert (p/'macro.exe.new').read_bytes()==DATA

@pytest.mark.parametrize('code',[408,429,500,502,503,504])
def test_retry_http(env,code):
    p,c,run=env; calls=[]
    def op(req,**kw):
        calls.append(req)
        if len(calls)==1:raise urllib.error.HTTPError(URL,code,'error',{},None)
        return Response()
    run(op);assert len(calls)==2

def test_retry_limit(env):
    p,c,run=env; calls=[]
    def op(*a,**kw):calls.append(1);raise ConnectionResetError()
    with pytest.raises(DownloadNetworkError):run(op)
    assert len(calls)==3 and not list(p.iterdir())

def test_wrong_range(env):
    p,c,run=env; calls=[]
    def op(*a,**kw):
        calls.append(1)
        return Response(fail=True) if len(calls)==1 else Response(DATA,206,{'Content-Range':'bytes 0-1/2'})
    with pytest.raises(DownloadIntegrityError):run(op)
    assert not list(p.iterdir())

@pytest.mark.parametrize('data',[b'MZ'+b'z'*(len(DATA)-2),b'NO'+DATA[2:],DATA+b'x'])
def test_bad_data_never_staged(env,data):
    p,c,run=env
    with pytest.raises(DownloadIntegrityError):run(lambda *a,**k:Response(data))
    assert not list(p.iterdir())

def test_cancel_during_download(env):
    p,c,run=env
    def progress(*a):c.set()
    with pytest.raises(DownloadCancelled):run(lambda *a,**k:Response(),progress=progress)
    assert not list(p.iterdir())

def test_cancel_during_backoff(env):
    p,c,run=env
    def op(*a,**kw):raise ConnectionResetError()
    with pytest.raises(DownloadCancelled):run(op,status=lambda *a:c.set())
    assert not list(p.iterdir())

@pytest.mark.parametrize('url',['http://github.com/a','https://evil.test/file','https://github.com:444/a'])
def test_untrusted_redirect(env,url):
    p,c,run=env
    with pytest.raises(DownloadIntegrityError):run(lambda *a,**k:Response(url=url))
    assert not list(p.iterdir())

def test_permanent_http_not_retried(env):
    p,c,run=env;calls=[]
    def op(*a,**k):calls.append(1);raise urllib.error.HTTPError(URL,404,'not found',{},None)
    with pytest.raises(urllib.error.HTTPError):run(op)
    assert len(calls)==1

def test_truncated_eof_resume(env):
    p,c,run=env;calls=[];n=500000
    def op(req,**kw):
        calls.append(req)
        return Response(DATA[:n]) if len(calls)==1 else Response(DATA[n:],206,{'Content-Range':f'bytes {n}-{len(DATA)-1}/{len(DATA)}'})
    run(op);assert calls[1].get_header('Range')==f'bytes={n}-'

# Exercise the real main.py integration without importing GUI/Roblox dependencies.
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import os
TREE=ast.parse((Path(__file__).resolve().parents[1]/'main.py').read_text())
def extract(name,ns,method=False):
    body=next(n for n in TREE.body if isinstance(n,ast.ClassDef) and n.name=='App').body if method else TREE.body
    node=next(n for n in body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'main.py','exec'),ns)
    return ns[name]

def test_staging_only_after_verification(tmp_path):
    new=str(tmp_path/'macro.exe.new');events=[]
    state={'cancel':threading.Event(),'busy':True,'downloading':True,'staged':None}
    def download(info,dest,**kw):
        assert state['staged'] is None
        Path(dest).write_bytes(DATA)
    ns=dict(sys=SimpleNamespace(executable=new[:-4],frozen=True),IS_WIN=True,_update=state,
            os=os,UPDATE_REPO=REPO,UPDATE_TIMEOUT=15,download_verified=download,
            DownloadCancelled=DownloadCancelled,DownloadNetworkError=DownloadNetworkError,
            update_history=lambda:SimpleNamespace(blocked=lambda *a:False,record=Mock()),
            log=Mock(),ev=SimpleNamespace(put=events.append),release_update_lock=Mock())
    extract('_install_update_worker',ns)(dict(INFO,tag='v1.4.1'))
    assert state['staged']==new and not state['busy'] and not state['downloading']
    assert events==[('update_download_done',),('update_staged',)]
    ns['release_update_lock'].assert_not_called()

@pytest.mark.parametrize('exc',[DownloadCancelled('cancel'),DownloadNetworkError('failed')])
def test_worker_cleanup(tmp_path,exc):
    events=[];state={'cancel':threading.Event(),'busy':True,'downloading':True,'staged':None}
    ns=dict(sys=SimpleNamespace(executable=str(tmp_path/'macro.exe'),frozen=True),IS_WIN=True,_update=state,
            os=os,UPDATE_REPO=REPO,UPDATE_TIMEOUT=15,download_verified=Mock(side_effect=exc),
            DownloadCancelled=DownloadCancelled,DownloadNetworkError=DownloadNetworkError,
            update_history=lambda:SimpleNamespace(blocked=lambda *a:False,record=Mock()),
            log=Mock(),ev=SimpleNamespace(put=events.append),release_update_lock=Mock())
    extract('_install_update_worker',ns)(dict(INFO,tag='v1.4.1',manual=True))
    assert state['staged'] is None and not state['busy'] and not state['downloading']
    assert ('update_staged',) not in events
    ns['release_update_lock'].assert_called_once()
    assert (('update_download_failed',) in events)==isinstance(exc,DownloadNetworkError)

@pytest.mark.parametrize('started,stopped,worker_alive,flags,expected',[
    (False,False,False,{},True),  # pristine idle: no Stop needed
    (False,True,False,{},True),
    (True,True,False,{},True),  # fully stopped
    (True,False,False,{},False),  # unexpected exit
    (True,False,True,{},False),  # active / waiting for mission
    (True,True,True,{},False),  # stopping, worker not yet exited
    (True,True,False,{'paused':True},False),
    (True,True,False,{'reconnecting':True},False),
    (True,True,False,{'rescuing':True},False),
    (True,True,False,{'mission_t0':123.},False),
    (False,False,False,{'reconnecting':True},False),
    (False,False,False,{'paused':True},False),
])
def test_stop_gate(started,stopped,worker_alive,flags,expected):
    ns=dict(running=lambda:worker_alive,state=dict(session_active=started,user_stopped=stopped,**flags))
    assert extract('can_install_update',ns)() is expected


def test_pristine_idle_manual_ready_without_stop():
    ns=dict(_update={'staged':'macro.exe.new','prompted':True},running=lambda:False,
            state={'session_active':False,'user_stopped':False},start_update_check=Mock())
    extract('can_install_update',ns)
    app=SimpleNamespace(root=None,maybe_install_update=Mock())
    extract('on_update_click',ns,True)(app)
    app.maybe_install_update.assert_called_once()
    assert not ns['_update']['prompted'] and not ns['state']['user_stopped']


@pytest.mark.parametrize('mode',['automatic','ask'])
@pytest.mark.parametrize('started,stopped,alive,expected',[
    (False,False,False,True),(True,False,False,False),
    (True,True,True,False),(True,True,False,True)])
def test_idle_gate_install_preference(mode,started,stopped,alive,expected):
    prompt=Mock(return_value=True);launch=Mock(return_value=True)
    ns=dict(_update={'staged':'macro.exe.new','prompted':False},running=lambda:alive,
            state={'session_active':started,'user_stopped':stopped},
            settings={'update_prefs':{'install_mode':mode}},messagebox=SimpleNamespace(askyesno=prompt),
            launch_staged_update=launch)
    extract('can_install_update',ns)
    extract('maybe_install_update',ns,True)(SimpleNamespace(root=None))
    assert launch.call_count==int(expected)
    assert prompt.call_count==int(expected and mode=='ask')


def test_start_stop_and_unexpected_exit_lifecycle():
    import time
    guard=threading.Lock();events=[];workers=[]
    class FakeWorker:
        def __init__(self,**kwargs):self.alive=False;workers.append(self)
        def is_alive(self):return self.alive
        def start(self):
            assert guard.locked(), 'worker must start under the same guard as install'
            self.alive=True
    ns=dict(worker=None,state={'session_active':False,'user_stopped':False},
            _update={},_update_install_guard=guard,_health_lock=threading.Lock(),
            _shared_scan_lock=threading.Lock(),_shared_scan_counts={},
            settings={'family':'Universal'},time=time,threading=SimpleNamespace(Thread=FakeWorker),
            ev=SimpleNamespace(put=events.append),log=Mock(),_reset_shared_scan=Mock(),
            session_history=lambda:SimpleNamespace(begin=Mock()),macro_main=Mock(),start_reporter=Mock())
    for name in ('running','can_install_update','start_macro','request_stop'):extract(name,ns)
    assert ns['can_install_update']()
    ns['start_macro']();assert ns['state']['session_active'] and not ns['can_install_update']()
    # Simulate a worker exit without user Stop: sticky Start latch must survive.
    workers[-1].alive=False
    assert not ns['can_install_update']()
    ns['request_stop']();assert ns['can_install_update']()
    ns['start_macro']();assert not ns['state']['user_stopped'] and not ns['can_install_update']()
    ns['request_stop']();assert ns['state']['quit'] and not ns['can_install_update']()
    workers[-1].alive=False;assert ns['can_install_update']()
    ns['_update']['install_launched']=True
    ns['start_macro']();assert len(workers)==2  # updater owns the guard, no new Start


def test_cancel_click():
    c=threading.Event();ns=dict(_update={'downloading':True,'cancel':c},messagebox=SimpleNamespace(askyesno=lambda *a,**k:True),start_update_check=Mock())
    extract('on_update_click',ns,True)(SimpleNamespace(root=None))
    assert c.is_set();ns['start_update_check'].assert_not_called()

@pytest.mark.parametrize('mode',['response','trickle'])
def test_total_deadline(env,mode):
    from thebums_update_download import DownloadDeadlineExceeded
    p,c,run=env; tick=[0.]; calls=[]
    class Slow(Response):
        def read1(self,n):
            tick[0]+=3
            return self.data.read(min(n,4096))
    def op(*a,**k):
        calls.append(1)
        if mode=='response':tick[0]=11
        return Slow()
    with pytest.raises(DownloadDeadlineExceeded):run(op,clock=lambda:tick[0],max_duration=10)
    assert len(calls)==1 and not list(p.iterdir())

def test_old_lock_not_deleted(tmp_path):
    import time
    exe=str(tmp_path/'macro.exe'); lock=Path(exe+'.updating');lock.write_text('123')
    os.utime(lock,(time.time()-100000,time.time()-100000))
    ns=dict(sys=SimpleNamespace(executable=exe),os=os,time=time,log=Mock(),_update={})
    assert not extract('acquire_update_lock',ns)()
    assert lock.read_text()=='123'
    ns['log'].assert_called_once()

def test_staged_click_waits_for_explicit_stop():
    ns=dict(_update={'staged':'macro.exe.new','prompted':True},can_install_update=lambda:False,
            messagebox=SimpleNamespace(showinfo=Mock()),start_update_check=Mock())
    self=SimpleNamespace(root=None,maybe_install_update=Mock())
    extract('on_update_click',ns,True)(self)
    ns['messagebox'].showinfo.assert_called_once();self.maybe_install_update.assert_not_called()

def test_staged_click_reopens_declined_prompt():
    ns=dict(_update={'staged':'macro.exe.new','prompted':True},can_install_update=lambda:True,start_update_check=Mock())
    self=SimpleNamespace(root=None,maybe_install_update=Mock())
    extract('on_update_click',ns,True)(self)
    assert not ns['_update']['prompted'];self.maybe_install_update.assert_called_once()

@pytest.mark.parametrize('frozen,relative,expected',[(True,False,True),(True,True,False),(False,False,False)])
def test_onefile_only(tmp_path,frozen,relative,expected):
    app=tmp_path/'installed'; bundle=app/'_internal' if relative else tmp_path/'temp'/'_MEI123'
    ns=dict(IS_WIN=True,sys=SimpleNamespace(executable=str(app/'macro.exe'),frozen=frozen,_MEIPASS=str(bundle)),os=os)
    assert extract('supports_exe_update',ns)() is expected


def test_commit_api_error_does_not_hide_release():
    def api(path):
        if path=='releases/latest':return {'tag_name':'v1.4.2','assets':[{'name':'TheBumsMacro.exe','browser_download_url':URL,'size':len(DATA),'digest':INFO['digest']}]}
        raise urllib.error.URLError('temporary failure')
    ns=dict(_github_api=api,UPDATE_REPO=REPO,GITHUB_URL='https://github.com/'+REPO,urllib=__import__('urllib'),local_build_revision=lambda:None)
    info=extract('fetch_repo_updates',ns)()
    assert info['release']['tag']=='v1.4.2' and info['commit'] is None


def test_launch_failure_keeps_owned_stage_and_lock(tmp_path):
    import shutil, uuid, tempfile
    from thebums_update_handoff import cancel_uncommitted
    (tmp_path/'updater.ps1').write_text('test')
    exe=tmp_path/'macro.exe';staged=tmp_path/'macro.exe.new';staged.write_bytes(DATA)
    lock=tmp_path/'macro.exe.updating';lock.write_text('123')
    state=dict(release=dict(tag='v1.4.2',digest=INFO['digest']),staged=str(staged),lockfile=str(lock),prompted=True)
    ns=dict(can_install_update=lambda:True,supports_exe_update=lambda:True,_update=state,os=os,
        sys=SimpleNamespace(executable=str(exe),_MEIPASS=str(tmp_path)),app_dir=lambda:str(tmp_path),data_dir=lambda:str(tmp_path),
        _update_install_guard=threading.Lock(),uuid=uuid,tempfile=SimpleNamespace(gettempdir=lambda:str(tmp_path)),shutil=shutil,
        start_windows_helper=Mock(side_effect=OSError('launch failed')),cancel_uncommitted=cancel_uncommitted,
        ev=SimpleNamespace(put=Mock()),log=Mock(),release_update_lock=Mock())
    assert extract('launch_staged_update',ns)() is False
    assert lock.exists() and staged.exists() and not state['install_launched'] and not state['prompted']
    ns['release_update_lock'].assert_not_called()


def test_startup_transaction_marker(tmp_path,monkeypatch):
    import re,json
    name='update_health_'+'a'*32+'.ok';monkeypatch.setenv('THEBUMS_UPDATE_HEALTH',name)
    callbacks=[];root=SimpleNamespace(winfo_exists=lambda:True,after=lambda delay,callback:callbacks.append((delay,callback)))
    ns=dict(os=os,re=re,json=json,data_dir=lambda:str(tmp_path),log=Mock())
    extract('update_startup_check',ns)(root)
    next(cb for delay,cb in callbacks if delay==3000)()
    assert (tmp_path/name).read_text()=='ready' and not (tmp_path/'update_health.ok').exists()


def test_startup_legacy_marker(tmp_path,monkeypatch):
    import re,json
    monkeypatch.delenv('THEBUMS_UPDATE_HEALTH',raising=False)
    callbacks=[];root=SimpleNamespace(winfo_exists=lambda:True,after=lambda delay,callback:callbacks.append((delay,callback)))
    ns=dict(os=os,re=re,json=json,data_dir=lambda:str(tmp_path),log=Mock())
    extract('update_startup_check',ns)(root)
    next(cb for delay,cb in callbacks if delay==3000)()
    assert (tmp_path/'update_health.ok').read_text()=='ready'


def test_helper_contract_static():
    # Structural checks only; actual replacement/rollback needs Windows PowerShell.
    src=(Path(__file__).resolve().parents[1]/'updater.ps1').read_text()
    assert src.index('Get-FileHash')<src.index('Move-Item -LiteralPath $StagedExe')
    assert 'if ($replacementAttempted -and $backupCreated)' in src
    assert 'THEBUMS_UPDATE_HEALTH' in src
