"""Offline Raid reconnect: real supplied menu crops, simulated Female/START/HUD transitions."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import cv2
import numpy as np
import pytest
import thebums_reconnect as rc

ROOT=Path(__file__).resolve().parents[1]


def frame(name):
    x,y = {'raid_menu':(325,325),'raid_attack_menu':(465,282)}[name]
    crop=cv2.imread(str(ROOT/'tests/fixtures'/f'{name}.png'))
    f=np.zeros((1079,1919,3),np.uint8)
    f[y:y+crop.shape[0],x:x+crop.shape[1]]=crop
    return f


def female_simulated():
    f=frame('raid_attack_menu')
    # Only the unselected Female card is supplied. Synthesize the same
    # selected gold outline as the real Attack card and remove Attack heading.
    f[305:361,608:993]=0
    for y in (607,608,609,725,726,727):
        f[y,1180:1429]=(30,200,255)
    return f


@pytest.fixture(scope='module')
def matcher():
    return rc.ImageMatcher(str(ROOT/'image'))


@pytest.mark.parametrize('capture',['monitor','client'])
def test_supplied_raid_menu_and_selection(matcher,capture):
    menu,attack=frame('raid_menu'),frame('raid_attack_menu')
    if capture=='client':
        def transform(f):
            f=f[23:1032]
            return cv2.resize(f,None,fx=1080/f.shape[0],fy=1080/f.shape[0],interpolation=cv2.INTER_CUBIC)
        menu,attack=transform(menu),transform(attack)
    assert matcher.find(menu,'raids')
    assert matcher.find(menu,'raid_create_attack') is None
    assert matcher.find(menu,'raid_create_female') is None
    assert matcher.find(attack,'attack_selected')
    assert matcher.find(attack,'female_raid')
    assert matcher.find(attack,'raid_create_attack')
    assert matcher.find(attack,'female_selected') is None
    assert matcher.find(attack,'raid_create_female') is None
    assert matcher.find(attack,'raid_start') is None


def test_female_selection_requires_both_edges_and_no_attack_header(matcher):
    f=female_simulated()
    assert matcher.find(f,'female_selected')
    assert matcher.find(f,'raid_create_female')
    assert matcher.find(f,'raid_create_attack') is None
    attack=frame('raid_attack_menu')
    f[305:361,608:993]=attack[305:361,608:993]
    assert matcher.find(f,'raid_create_female') is None
    f=female_simulated(); f[718:735,1178:1430]=0
    assert matcher.find(f,'raid_create_female') is None


def test_raid_start_reuses_original_button_in_roi(matcher):
    f=np.zeros((1080,1920,3),np.uint8)
    t=matcher.templates['start']; h,w=t.shape[:2]
    f[800:800+h,1000:1000+w]=t
    assert matcher.find(f,'raid_start')
    f[:]=0;f[100:100+h,100:100+w]=t
    assert matcher.find(f,'raid_start') is None


@pytest.mark.parametrize('destination,tail',[
    ('mission',['play_large','missions','outskirts','create','start']),
    ('attack',['play_large','raids','raid_create_attack','raid_start']),
    ('female',['play_large','raids','female_raid','raid_create_female','raid_start']),
])
@pytest.mark.parametrize('mode,prefix',[
    ('public',['slot_select','play_small','lobby']),
    ('ps_code',['slot_select','ps_input','generate','lobby']),
    ('gamepass',['slot_select','generate']),
])
def test_full_state_machine(mode,prefix,destination,tail):
    # Every click changes screen; the next marker must be found on the new screen.
    steps=prefix+tail
    gates={'slot_select':'play_small','play_small':'lobby','lobby':'play_large',
           'generate':'lobby' if mode=='ps_code' else 'play_large',
           'missions':'outskirts','outskirts':'outskirts_selected','create':'start',
           'raids':'attack_selected','female_raid':'female_selected',
           'raid_create_attack':'raid_start','raid_create_female':'raid_start'}
    gates['play_large']='missions' if destination=='mission' else 'raids'
    index=0; clicked=[]; ready=Mock(return_value=True)
    def capture():return index
    def match(frame,name):
        if frame>=len(steps):return None
        visible={steps[frame]}
        if frame:
            visible.add(gates.get(steps[frame-1]))
        if mode=='ps_code' and steps[frame]=='ps_input': visible.add('generate')
        if mode=='gamepass' and steps[frame]=='generate':visible.add('generate')
        return rc.Match(name,100,100,1.,20,20) if name in visible else None
    def click(hit):
        nonlocal index
        assert hit.name==steps[index]
        clicked.append(hit.name);index+=1
    m=SimpleNamespace(find=match,slot_a_select=lambda f:match(f,'slot_select'))
    code=Mock();flow=rc.ImageReconnectFlow(m,capture,click,lambda:None,lambda f:False,
                    lambda *a:None,code,ready,server_mode=mode,ps_code='Test123' if mode=='ps_code' else '',destination=destination)
    # State machine still performs its real stable-marker checks, but no real delays.
    flow._sleep=lambda seconds:None
    assert flow.run()
    assert clicked==steps
    ready.assert_called_once_with(len(steps))
    if mode=='ps_code': code.assert_called_once_with('Test123')
    else:code.assert_not_called()


@pytest.mark.parametrize('family', ['Attack Titan Raid','Attack Titan Raid Blood Awakening','Female Raid'])
def test_settings_accept_raid_without_coordinates(family):
    tree=ast.parse((ROOT/'main.py').read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reconnect_missing')
    import os
    env=dict(os=os,settings={'family':family,'reconnect':{'method':'image','server_mode':'public'}},
             OUTSKIRTS_FRITZ_KEY='Outskirts Fritz',COLOSSAL_RAIDS=('Attack Titan Raid','Attack Titan Raid Blood Awakening','Female Raid'),
             image_rc=rc,asset_path=lambda n:str(ROOT/'image'/n))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'main.py','exec'),env)
    assert env['reconnect_missing']()==[]
    env['settings']['reconnect']['server_mode']='ps_code'
    assert env['reconnect_missing']()==['Private Server Code']
