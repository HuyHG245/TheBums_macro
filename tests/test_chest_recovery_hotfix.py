"""Offline simulated-clock tests for the actual chest functions (no GUI/input)."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

TREE=ast.parse((Path(__file__).resolve().parents[1]/'main.py').read_text(encoding='utf-8'))
NAMES={'chest_poll','chest_recovery_failed','chest_verified_click','chest_finish_and_retry','female_chests','chest_coordinate_burst','chest_wait_panel','attack_titan_chests','raid_finish','macro_openchest'}
CONSTANTS={'CHEST_OPEN_TIMEOUT','CHEST_MAX_ATTEMPTS','CHEST_RETRY_INTERVAL','CHEST_SETTLE','CHEST_FINISH_TIMEOUT'}
class Retry(Exception): pass
class Disconnect(Exception): pass
M=(400,700,.99,80,35)
L=(250,600,.99,80,35)
R=(650,600,.99,80,35)

def env(family='Female Raid', paid=False, reconnect=False):
    t=SimpleNamespace(now=1000.)
    t.monotonic=t.time=lambda:t.now
    def sleep(dt):
        t.now+=dt
        assert t.now<1660., 'unbounded loop reproduced'
    t.sleep=sleep
    clicks=[]
    mouse=SimpleNamespace(point=None)
    def move(x,y): mouse.point=(x,y)
    def click(): clicks.append((t.now,mouse.point))
    config={'open_chest2':paid,'fast_mode':True,'finish_wait':6.,'chest1':(250,600),'chest2':(650,600),'chest_click_time':.8,'chest_click_interval':.12,'finish_click_time':.8,'finish_click_interval':.12}
    e=dict(time=t,state={},settings={'family':family,'reconnect':{'enabled':reconnect}},
           _disconnect_template=object() if reconnect else None,_last_scan=0.,FAST_SCAN_INTERVAL=.1,SCAN_INTERVAL=.5,
           chest_focus_game=Mock(return_value=True),chest_pointer_safe=Mock(return_value=True),tv=lambda f,k:config[k],tick=Mock(),scan_for_retry=Mock(),log=Mock(),show=Mock(),raid_phase=Mock(),release_all_keys=Mock(),
           capture_error_snapshot=Mock(),Disconnected=Disconnect,frame_to_screen=lambda x,y:(x,y),
           pydirectinput=SimpleNamespace(moveTo=Mock(side_effect=move),moveRel=Mock(),click=Mock(side_effect=click)),
           grab_screen=lambda:'frame',female_finish_button=lambda f:M,female_open_button=lambda f,s:L if s=='left' else R,
           attack_titan_picture=lambda k,f:M,raid_find=lambda k,f:M,_raid_templates={'finish':1},clicks=clicks,config=config)
    nodes=[n for n in TREE.body if (isinstance(n,ast.FunctionDef) and n.name in NAMES) or (isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in CONSTANTS for x in n.targets))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'main.py','exec'),e)
    return e

def test_original_infinite_left_open_bounded_then_finish():
    e=env()
    with pytest.raises(SystemExit): e['female_chests']()
    left=[x for x in e['clicks'] if x[1]==L[:2]]; finish=[x for x in e['clicks'] if x[1]==M[:2]]
    assert len(left)==5 and len(finish)==5
    assert all(b[0]-a[0]>=2.5-1e-9 for a,b in zip(left,left[1:]))
    assert e['time'].now<1061
    e['capture_error_snapshot'].assert_any_call('chest_open_unconfirmed_left')
    assert not e['state']['raid_chest_active']

@pytest.mark.parametrize('reconnect',[False,True])
def test_panel_never_seen_has_no_click_and_exits(reconnect):
    e=env(reconnect=reconnect); e['female_finish_button']=lambda f:None
    with pytest.raises(Disconnect if reconnect else SystemExit): e['female_chests']()
    assert not e['clicks'] and e['time'].now<1061
    assert not e['state']['raid_chest_active']
    e['capture_error_snapshot'].assert_any_call('chest_recovery_finish_or_retry_timeout')

@pytest.mark.parametrize('paid',[False,True])
def test_open_clears_and_verified_retry_interrupts_finish(paid):
    e=env(paid=paid)
    e['female_open_button']=lambda f,s: None if any(x[1]==(L if s=='left' else R)[:2] for x in e['clicks']) else (L if s=='left' else R)
    def scan(**kwargs):
        if any(x[1]==M[:2] for x in e['clicks']): raise Retry
    e['scan_for_retry'].side_effect=scan
    with pytest.raises(Retry): e['female_chests']()
    assert [x[1] for x in e['clicks']]==[L[:2]]+([R[:2]] if paid else [])+[M[:2]]
    e['capture_error_snapshot'].assert_not_called()
    assert not e['state']['raid_chest_active']

def test_paid_open_stuck_never_repeats():
    e=env(paid=True)
    with pytest.raises(SystemExit): e['female_chests']()
    assert sum(x[1]==R[:2] for x in e['clicks'])==1
    assert sum(x[1]==L[:2] for x in e['clicks'])==5

def test_optional_paid_chest_missing_is_skipped():
    e=env(paid=True); e['female_open_button']=lambda f,s:None if s=='right' or e['clicks'] else L
    with pytest.raises(SystemExit): e['female_chests']()
    assert not any(x[1]==R[:2] for x in e['clicks'])
    assert any(x[1]==M[:2] for x in e['clicks'])

@pytest.mark.parametrize('fresh',[None,(430,700,.99,80,35)])
def test_revalidation_disallows_disappeared_or_moved_control(fresh):
    e=env(); assert not e['chest_verified_click'](lambda f:fresh,M,1030.)
    assert not e['clicks']; assert e['time'].now>=1000.15

@pytest.mark.parametrize('exc',[Retry,Disconnect,SystemExit])
def test_retry_disconnect_stop_during_settle_preempt_input(exc):
    e=env(); e['tick'].side_effect=exc
    with pytest.raises(exc): e['chest_verified_click'](lambda f:M,M,1030.)
    assert not e['clicks']

@pytest.mark.parametrize('family',['Attack Titan Raid Fritz','Attack Titan Raid Blood Awakening'])
@pytest.mark.parametrize('paid',[False,True])
def test_attack_preserves_coordinate_route_and_bounds_finish(family,paid):
    e=env(family,paid)
    with pytest.raises(SystemExit): e['attack_titan_chests']()
    assert any(x[1]==L[:2] for x in e['clicks'])
    assert sum(x[1]==R[:2] for x in e['clicks'])==int(paid)
    assert sum(x[1]==M[:2] for x in e['clicks'])==5
    assert e['time'].now<1040 and not e['state']['raid_chest_active']

def test_attack_does_not_click_coordinates_without_panel():
    e=env('Attack Titan Raid Fritz',paid=True); e['attack_titan_picture']=lambda k,f:None
    with pytest.raises(SystemExit): e['attack_titan_chests']()
    assert not e['clicks'] and e['time'].now<1044

def test_coordinate_burst_stops_when_panel_disappears():
    e=env(); e['chest_coordinate_burst'](L[:2],lambda f:None if e['clicks'] else M,2.,.1,'free')
    assert len(e['clicks'])==1

def test_custom_openchest_timeout_is_bounded():
    e=env('custom:raid'); e['raid_find']=lambda k,f:None
    with pytest.raises(SystemExit): e['macro_openchest']()
    assert not e['clicks'] and e['time'].now<1031 and not e['state']['raid_chest_active']

def test_custom_openchest_returns_after_verified_finish():
    e=env('custom:raid');e['config']['chest1']=None
    e['raid_find']=lambda k,f:None if e['clicks'] else M
    e['macro_openchest']()
    assert [x[1] for x in e['clicks']]==[M[:2]] and not e['state']['raid_chest_active']

def test_recovery_retry_preempts_rejoin():
    e=env(reconnect=True);e['scan_for_retry'].side_effect=Retry
    with pytest.raises(Retry):e['chest_recovery_failed']('test')
    e['capture_error_snapshot'].assert_not_called()


def test_finish_near_deadline_gets_separate_retry_window():
    e=env(); t=e['time']
    def locate(f): return M if t.now>=1028. else None
    def scan(**kwargs):
        if e['clicks'] and t.now>=1040.: raise Retry
    e['scan_for_retry'].side_effect=scan
    with pytest.raises(Retry): e['chest_finish_and_retry'](locate)
    assert t.now>=1040. and t.now<1041.
    assert e['clicks'][0][0]>=1028.


def test_reconnect_enabled_but_missing_templates_stops_safely():
    e=env(reconnect=True); e['_disconnect_template']=None
    with pytest.raises(SystemExit): e['chest_recovery_failed']('test')


def test_custom_finish_zero_setting_preserved():
    e=env('custom:raid'); e['config'].update(chest1=None,finish_click_time=0)
    e['macro_openchest']()
    assert not e['clicks']
