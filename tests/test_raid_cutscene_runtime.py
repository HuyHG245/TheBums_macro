"""Cutscene gating and real main.py functions with deterministic clock/input mocks."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import cv2
import numpy as np
import pytest
from raid_vision import locate_skip

ROOT=Path(__file__).resolve().parents[1]
TREE=ast.parse((ROOT/'main.py').read_text(encoding='utf-8'))

def load(names, env):
    nodes=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'main.py','exec'),env)
    return env


def setup(enabled=True):
    clock=SimpleNamespace(now=10.)
    clock.monotonic=lambda:clock.now
    templates={k:{'normal':cv2.imread(str(ROOT/'image'/name))} for k,name in
               [('skip','skip.png'),('skip_text','raid_skip_text.png')]}
    e=dict(time=clock,settings={'family':'Female Raid'},_raid_templates=templates,
           _attack_skip=dict(next_scan=0.,last_press=-100.,last_seen=-100.,streak=0,busy=False),
           tv=lambda f,k:enabled if k=='auto_skip_cutscene' else .9,
           locate_skip=locate_skip, tap=Mock(),log=Mock(), KEY_RAID_SKIP='e',
           grab_screen=Mock(side_effect=AssertionError('reuse supplied frame')),
           find_retry=Mock(return_value=None),attack_titan_start_icon_visible=Mock(return_value=True))
    load(['attack_titan_auto_skip','raid_skip_text','legacy_raid_skip','attack_titan_skip_visible',
          'raid_intro_clear','raid_start_ready'],e)
    return e,clock


def image(e,scale=1.):
    f=np.zeros((1080,1920,3),np.uint8)
    t=e['_raid_templates']['skip_text']['normal']
    t=cv2.resize(t,None,fx=scale,fy=scale)
    h,w=t.shape[:2]; f[920:920+h,1500:1500+w]=t
    return f


@pytest.mark.parametrize('scale',[.55,.70,.85,1.,1.15,1.30,1.50])
def test_skip_detected_in_expanded_region_at_scale(scale):
    e,c=setup(); f=image(e,scale)
    assert e['attack_titan_auto_skip'](f)
    e['tap'].assert_called_once_with('e')
    assert not e['raid_intro_clear'](f)


def test_reconnect_skips_before_hud_and_waits_for_clear():
    e,c=setup(); f=image(e)
    assert not e['raid_start_ready'](f,reconnect=True)
    e['tap'].assert_called_once_with('e')
    e['attack_titan_start_icon_visible'].assert_not_called()
    c.now+=.1
    blank=np.zeros_like(f)
    assert not e['raid_start_ready'](blank,reconnect=True)
    c.now+=.3
    assert e['raid_start_ready'](blank,reconnect=True)
    e['tap'].assert_called_once_with('e')


def test_normal_gate_and_disabled_skip_do_not_send_inputs():
    e,c=setup(enabled=False); f=image(e)
    assert not e['raid_start_ready'](f,reconnect=True)
    e['tap'].assert_not_called()
    assert not e['raid_start_ready'](f)
    c.now+=.4
    assert e['raid_start_ready'](np.zeros_like(f))


def test_retry_overlay_blocks_skip_and_hud():
    e,c=setup(); f=image(e)
    e['find_retry'].return_value=((960,800,.99,100,30),'normal')
    assert not e['raid_start_ready'](f,reconnect=True)
    e['tap'].assert_not_called()
    e['attack_titan_start_icon_visible'].assert_not_called()


def test_skip_repeats_only_at_configured_interval():
    e,c=setup(); f=image(e)
    for offset in (0,.1,.5,.8):
        c.now=10+offset; e['attack_titan_auto_skip'](f)
    e['tap'].assert_called_once_with('e')
    c.now=11.; e['attack_titan_auto_skip'](f)
    assert e['tap'].call_count==2


def test_actual_female_combat_screenshot_has_no_skip():
    e,c=setup()
    crop=cv2.imread(str(ROOT/'tests/fixtures/raid_combat_no_skip.png'))
    assert crop is not None
    f=np.zeros((1080,1922,3),np.uint8)
    y,x=int(f.shape[0]*.68),int(f.shape[1]*.65)
    f[y:y+crop.shape[0],x:x+crop.shape[1]]=crop
    assert not e['attack_titan_auto_skip'](f)
    assert e['raid_intro_clear'](f)
    e['tap'].assert_not_called()
