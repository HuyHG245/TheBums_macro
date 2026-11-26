"""Offline regressions for chest focus, cursor verification, and real relaunch."""
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path
import ast
import cv2
import numpy as np
import pytest
from test_chest_recovery_hotfix import env, M, Retry, Disconnect
from raid_vision import locate

ROOT=Path(__file__).resolve().parents[1]
TREE=ast.parse((ROOT/'main.py').read_text(encoding='utf-8'))
def load(name,e):
    n=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[n],type_ignores=[]),'main.py','exec'),e)
    return e[name]

def test_no_click_when_focus_denied():
    e=env();e['chest_focus_game'].return_value=False
    assert not e['chest_verified_click'](lambda f:M,M,1030.)
    assert not e['clicks']
    e['pydirectinput'].moveTo.assert_not_called()

def test_no_click_if_actual_cursor_or_window_is_wrong():
    e=env();e['chest_pointer_safe'].return_value=False
    assert not e['chest_verified_click'](lambda f:M,M,1030.)
    assert not e['clicks']
    e['chest_pointer_safe'].assert_called_once_with(401,700)

def test_verified_click_repositions_and_nudges_after_capture():
    e=env();assert e['chest_verified_click'](lambda f:M,M,1030.)
    assert e['pydirectinput'].moveTo.call_count==2
    assert e['pydirectinput'].moveRel.call_count==2
    assert len(e['clicks'])==1

def test_coordinate_chest_respects_pointer_guard():
    e=env();e['chest_pointer_safe'].return_value=False
    e['chest_coordinate_burst']((250,600),lambda f:M,2.,.1,'free')
    assert not e['clicks']

def test_resume_existing_chests_finishes_without_paid_click():
    e=env(paid=True);e.update(COLOSSAL_RAIDS=('Female Raid',),locate_raid_control=Mock(return_value=M))
    e['chest_finish_and_retry']=Mock(side_effect=Retry)
    with pytest.raises(Retry):load('resume_visible_raid_chest',e)()
    e['chest_finish_and_retry'].assert_called_once_with(e['female_finish_button'])
    assert not e['state']['raid_chest_active'] and not e['clicks']

def test_resume_needs_two_confirmed_headers():
    e=env();e.update(COLOSSAL_RAIDS=('Female Raid',),locate_raid_control=Mock(side_effect=[M,None]))
    e['chest_finish_and_retry']=Mock();load('resume_visible_raid_chest',e)()
    e['chest_finish_and_retry'].assert_not_called()

@pytest.mark.parametrize('name',['raid_menu.png','raid_failed_dim_heading.png','raid_combat_no_skip.png','error_256_dialog.png'])
def test_header_does_not_match_unrelated_frames(name):
    tpl=cv2.imread(str(ROOT/'image/chest_header.png'))
    f=cv2.imread(str(ROOT/'tests/fixtures'/name))
    assert locate(f,{'normal':tpl},(.30,.025,.70,.20),.88) is None

def test_chest_header_geometry():
    tpl=cv2.imread(str(ROOT/'image/chest_header.png'));f=np.zeros((1080,1920,3),np.uint8)
    h,w=tpl.shape[:2];f[78:78+h,858:858+w]=tpl
    assert locate(f,{'normal':tpl},(.30,.025,.70,.20),.88)[:2]==(960,102)

@pytest.mark.parametrize('reason,slot,expected',[('chest recovery: finish_or_retry_timeout',False,True),('chest recovery: finish_or_retry_timeout',True,False),('Reconnect button detected by image watchdog',True,False)])
def test_image_reconnect_only_relaunches_for_chest_recovery_when_needed(reason,slot,expected):
    order=[]
    matcher=SimpleNamespace(find=lambda f,k:None,slot_a_select=lambda f:slot)
    rc=SimpleNamespace(ImageMatcher=Mock(return_value=matcher),find_leave_disconnect=lambda f:None,
        find_roblox_home_game=lambda f:None,find_roblox_game_play=lambda f:None,
        ImageReconnectFlow=Mock(return_value=SimpleNamespace(run=lambda:order.append('route'))),
        NewDisconnect=type('NewDisconnect',(Exception,),{}),ReconnectError=type('ReconnectError',(Exception,),{}))
    import sys,os
    e=dict(sys=sys,os=os,app_dir=lambda:'.',get_hybrid_advisor=lambda:None,image_rc=rc,state={},
        release_all_keys=Mock(),grab_screen=lambda:'frame',tick=Mock(),settings={'family':'Female Raid','reconnect':{}},
        ATTACK_FAMILIES=(),_attack_skip={},RECONNECT_MAX_ATTEMPTS=3,log=Mock(),reconnect_launch=lambda reason:order.append('launch'),
        _image_reconnect_progress=Mock(),session_history=lambda:SimpleNamespace(record=Mock()),ev=Mock(),
        time=SimpleNamespace(time=lambda:0.),RECONNECT_COOLDOWN=10)
    assert load('image_reconnect_rejoin',e)(reason) is True
    assert order==(['launch','route'] if expected else ['route'])
