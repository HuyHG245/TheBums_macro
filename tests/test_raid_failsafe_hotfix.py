"""Exercise actual recovery functions with isolated time/screen/input mocks.
No Windows input, network or gameplay is used by these tests.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call
import cv2
import numpy as np
import pytest
import thebums_extra_failsafe as extra_fs

ROOT=Path(__file__).resolve().parents[1]
TREE=ast.parse((ROOT/'main.py').read_text(encoding='utf-8'))
RAIDS=('Female Raid','Attack Titan Raid Fritz','Attack Titan Raid Blood Awakening')
class Retry(Exception): pass
class Chest(Exception): pass
class Disconnect(Exception): pass
class Stop(Exception): pass

def load(names,e):
    nodes=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'main.py','exec'),e)
    return e

def clock(now=10.):
    t=SimpleNamespace(now=now)
    t.time=lambda:t.now
    t.monotonic=lambda:t.now
    t.sleep=lambda dt:setattr(t,'now',t.now+dt)
    return t

def env(family='Female Raid', elapsed=181):
    t=clock(1000.)
    st=dict(quit=False,paused=False,mission_t0=t.now-elapsed,rescuing=False,debug_rescue=False)
    e=dict(time=t,state=st,extra_fs=extra_fs,raid_counting=lambda:family in RAIDS,settings={'family':family,'reconnect':{}},_raid={'seq_done_t':None,'active':False},
        RAID_FAILSAFE_MAX_TIME=180.,RAID_FAILSAFE_AFTER_SEQUENCE=180.,
        RETRY_CONFIRM_DELAY=.08,COLOSSAL_RAIDS=RAIDS,ATTACK_FAMILIES=RAIDS[1:],
        grab_screen=Mock(return_value='fresh'),passive_scan_screen=Mock(return_value='cached'),
        raid_terminal_heading=Mock(return_value=None),wait_raid_terminal_retry=Mock(side_effect=Retry),
        scan_for_retry=Mock(),check_colossal_chest=Mock(),raid_trigger_failsafe=Mock(side_effect=Stop),
        check_macro_health=Mock(),_disconnect_template=None,cached_retry_hit=Mock(return_value=None),
        RetryFound=Retry,Disconnected=Disconnect,attack_titan_auto_skip=Mock(),
        mission_timeout=lambda:120.,rescue_clicks=Mock(side_effect=Stop),raid_skip_poll=Mock(),
        tv=lambda f,k:False,FAST_SCAN_INTERVAL=.1,SCAN_INTERVAL=.5,_last_scan=t.now)
    return load(['raid_failsafe_reason','raid_failsafe_time_check','raid_runtime_recovery_check','extra_failsafe_check','tick'],e)

@pytest.mark.parametrize('fam',RAIDS)
def test_all_native_raids_use_180_not_mission_timer(fam):
    e=env(fam,180.)
    e['tick']()
    e['raid_trigger_failsafe'].assert_not_called()
    e['rescue_clicks'].assert_not_called()
    e['time'].now+=.01
    with pytest.raises(Stop): e['tick']()
    e['raid_trigger_failsafe'].assert_called_once_with('raid longer than 3 min')
    e['scan_for_retry'].assert_called_once()
    e['check_colossal_chest'].assert_called_once()

@pytest.mark.parametrize('fam',['Universal','Blood Awakening','Shiki','Outskirts Fritz','custom:raid'])
def test_non_native_raid_still_uses_mission_settings(fam):
    e=env(fam,120.)
    e['tick']()
    e['rescue_clicks'].assert_not_called()
    e['time'].now+=.01
    with pytest.raises(Stop): e['tick']()
    e['rescue_clicks'].assert_called_once_with()
    e['raid_trigger_failsafe'].assert_not_called()

@pytest.mark.parametrize('flag',['rescuing','raid_chest_active','retry_processing','reconnecting','raid_terminal_wait'])
def test_no_duplicate_or_wrong_phase_recovery(flag):
    e=env(); e['state'][flag]=True
    e['raid_runtime_recovery_check']()
    e['raid_trigger_failsafe'].assert_not_called()

@pytest.mark.parametrize('kind',['failed','completed'])
def test_terminal_heading_confirmed_before_timeout_no_eject(kind):
    e=env(elapsed=100.)
    e['raid_terminal_heading'].side_effect=[kind,kind]
    with pytest.raises(Retry): e['tick']()
    e['wait_raid_terminal_retry'].assert_called_once_with(kind)
    e['raid_trigger_failsafe'].assert_not_called()
    e['attack_titan_auto_skip'].assert_not_called()
    e['grab_screen'].assert_called_once()

def test_single_ambiguous_heading_defers_due_rescue():
    e=env(); e['raid_terminal_heading'].side_effect=['failed',None]
    e['tick']()
    e['raid_trigger_failsafe'].assert_not_called()
    e['scan_for_retry'].assert_called_once()

def test_verified_retry_preempts_eject_even_with_obscured_heading():
    e=env(); e['scan_for_retry'].side_effect=Retry
    with pytest.raises(Retry): e['tick']()
    e['check_colossal_chest'].assert_not_called()
    e['raid_trigger_failsafe'].assert_not_called()

def test_cached_retry_preempts_every_recovery_action():
    e=env(); e['cached_retry_hit'].return_value=('match','normal','frame',0.)
    with pytest.raises(Retry): e['tick']()
    e['raid_trigger_failsafe'].assert_not_called()
    e['grab_screen'].assert_not_called()

def test_disconnect_preempts_recovery():
    e=env(); e['_disconnect_template']={'normal':object()}; e['reconnect_watchdog_hit']=lambda:True
    with pytest.raises(Disconnect): e['tick']()
    e['raid_trigger_failsafe'].assert_not_called()
    e['grab_screen'].assert_not_called()

def test_chest_preempts_due_rescue():
    e=env(); e['check_colossal_chest'].side_effect=Chest
    with pytest.raises(Chest): e['tick']()
    e['raid_trigger_failsafe'].assert_not_called()

def test_chest_during_rescue_is_routed_without_second_rescue():
    e=env(); e['state']['rescuing']=True; e['check_colossal_chest'].side_effect=Chest
    with pytest.raises(Chest): e['raid_runtime_recovery_check'](force=True,allow_rescue=False)
    e['raid_trigger_failsafe'].assert_not_called()

def test_normal_checks_rate_limited_but_deadline_not_delayed():
    e=env(elapsed=179.9)
    e['tick'](); e['tick']()
    assert e['passive_scan_screen'].call_count==1
    e['time'].now+=.2
    with pytest.raises(Stop): e['tick']()

@pytest.mark.parametrize('interrupt',[Retry,Disconnect,SystemExit])
def test_terminal_wait_releases_keys_and_never_sends_unverified_inputs(interrupt):
    e=env(); e.update(tick=Mock(side_effect=interrupt),_skip_spam={'until':5.},
        _custom_held={'w'},pydirectinput=SimpleNamespace(keyUp=Mock()),release_all_keys=Mock(),log=Mock(),show=Mock())
    load(['wait_raid_terminal_retry'],e)
    with pytest.raises(interrupt): e['wait_raid_terminal_retry']('failed')
    e['release_all_keys'].assert_called_once()
    e['pydirectinput'].keyUp.assert_called_once_with('w')
    assert e['_skip_spam']['until']==0 and not e['_custom_held'] and not e['state']['raid_terminal_wait']
    e['scan_for_retry'].assert_not_called()

def test_terminal_wait_uses_existing_two_frame_scanner():
    e=env(); e.update(tick=Mock(),_skip_spam={'until':0.},_custom_held=set(),
        pydirectinput=SimpleNamespace(keyUp=Mock()),release_all_keys=Mock(),log=Mock(),show=Mock())
    e['scan_for_retry'].side_effect=Retry
    load(['wait_raid_terminal_retry'],e)
    with pytest.raises(Retry): e['wait_raid_terminal_retry']('failed')
    e['scan_for_retry'].assert_called_once()
    e['tick'].assert_called_once_with(scan_retry=False)

def vision_env():
    e=dict(cv2=cv2)
    load(['gold_mask','raid_terminal_heading'],e)
    e['_result_templates']={k:e['gold_mask'](cv2.imread(str(ROOT/'image'/f'result_{k}.png'))) for k in ('failed','completed')}
    return e

def test_real_dim_raid_failed_report_heading():
    e=vision_env(); f=np.zeros((1080,1920,3),np.uint8)
    c=cv2.imread(str(ROOT/'tests/fixtures/raid_failed_dim_heading.png'))
    assert c is not None
    y,x=int(.14*1080),int(.40*1920); f[y:y+c.shape[0],x:x+c.shape[1]]=c
    assert e['raid_terminal_heading'](f)=='failed'

@pytest.mark.parametrize('kind',['failed','completed'])
@pytest.mark.parametrize('scale',[.75,1.,1.3])
def test_heading_templates_at_supported_sizes(kind,scale):
    e=vision_env(); f=np.zeros((1080,1920,3),np.uint8)
    c=cv2.resize(cv2.imread(str(ROOT/'image'/f'result_{kind}.png')),None,fx=scale,fy=scale)
    h,w=c.shape[:2]; f[200:200+h,900:900+w]=c
    assert e['raid_terminal_heading'](f)==kind

@pytest.mark.parametrize('fixture',['raid_attack_menu.png','raid_menu.png','raid_combat_no_skip.png','idle_278_dialog.png','error_256_dialog.png'])
def test_non_terminal_reference_images(fixture):
    e=vision_env(); f=np.zeros((1080,1920,3),np.uint8)
    c=cv2.imread(str(ROOT/'tests/fixtures'/fixture)); assert c is not None
    h,w=c.shape[:2]; y=max(0,(1080-h)//2); x=max(0,(1920-w)//2)
    f[y:y+min(h,1080),x:x+min(w,1920)]=c[:1080,:1920]
    assert e['raid_terminal_heading'](f) is None

@pytest.mark.parametrize('family,reason,look',[('Universal',None,False),('Female Raid','raid longer than 3 min',True),('Attack Titan Raid Fritz','raid longer than 3 min',False),('Attack Titan Raid Blood Awakening','raid longer than 3 min',True)])
def test_shared_rescue_exact_mission_sequence(family,reason,look):
    e=env(family); t=e['time']; events=[]; start=t.now
    def spam(key,seconds,interval,label):
        events.append(('spam',key,seconds,t.now-start)); t.sleep(seconds)
    def look_down(*a,**kw): events.append(('look',round(t.now-start,4),kw))
    def click(x,y): events.append(('click',round(t.now-start,4),x,y))
    e.update(_custom_held={'w'},pydirectinput=SimpleNamespace(keyUp=Mock()),_skip_spam={'until':12},
        settings={'family':family,'failsafe':{'look_down':look,'look_time':2.,'look_pixels':40,'look_step':.01}},
        raid_runtime_recovery_check=Mock(return_value=True),release_all_keys=Mock(),
        notify_rescue_async=Mock(),counters=lambda:({'missions':2},None),failsafe_default_point=lambda:(960,864),
        failsafe_click_point=lambda x,y:(x,y),show=Mock(),log=Mock(),_raid_templates={},
        tick=Mock(),scan_for_retry=Mock(side_effect=[None,Retry()]),spam_key_for=spam,wait=t.sleep,
        look_down=look_down,click_middle=click,KEY_ABILITY_1ST='2',STILL_SHIFTED_SPAM_TIME=5.,
        STILL_SHIFTED_WAIT_START=2.,STILL_SHIFTED_WAIT=15.,STILL_SHIFTED_SPAM_INTERVAL=.1,RESCUE_CLICK_DELAY=.5)
    load(['rescue_clicks','raid_trigger_failsafe'],e)
    with pytest.raises(Retry):
        if reason: e['raid_trigger_failsafe'](reason)
        else: e['rescue_clicks']()
    assert events[0]==('spam','2',5.,0.)
    assert events[-1]==('click',17.,960,864)
    if look: assert events[1]==('look',17.,dict(seconds=2.,pixels=40,step=.01))
    assert not e['state']['rescuing'] and e['state']['rescue_t0'] is None
    assert e['state']['rescued'] and not e['_custom_held']
    e['release_all_keys'].assert_called_once()
    e['pydirectinput'].keyUp.assert_called_once_with('w')
    if reason:
        assert e['raid_runtime_recovery_check'].call_args_list==[call(force=True,allow_rescue=False),call(force=True,allow_rescue=False,retry_checked=True)]


def test_force_ambiguous_heading_refuses_preflight():
    e=env(); e['raid_terminal_heading'].side_effect=['failed',None]
    assert e['raid_runtime_recovery_check'](force=True,allow_rescue=False) is False
    e['scan_for_retry'].assert_called_once()
    e['check_colossal_chest'].assert_not_called()


def test_rescue_preflight_failure_does_not_eject():
    e=env(); e['raid_runtime_recovery_check']=Mock(return_value=False); e['release_all_keys']=Mock()
    load(['rescue_clicks'],e)
    e['rescue_clicks'](reason='raid longer than 3 min')
    assert not e['state']['rescuing']
    e['release_all_keys'].assert_not_called()


@pytest.mark.parametrize('family,timeout',[('Universal',120.),('Female Raid',180.),('Attack Titan Raid Fritz',180.)])
def test_level2_preempts_active_level1_with_scan_disabled(family, timeout):
    """Exercise real tick + deadline check, not only a standalone timer helper."""
    e=env(family, elapsed=2*timeout-.01)
    class Extra(Disconnect): pass
    e['ExtraFailsafe']=Extra
    e['state'].update(fs2_started=e['state']['mission_t0'], fs2_timeout=timeout, rescuing=True)
    e['tick'](scan_retry=False)
    e['grab_screen'].assert_not_called()
    e['time'].now += .02
    with pytest.raises(Extra):
        e['tick'](scan_retry=False)
    e['scan_for_retry'].assert_called_once_with(check_grab=False)
    e['rescue_clicks'].assert_not_called()
    e['raid_trigger_failsafe'].assert_not_called()
