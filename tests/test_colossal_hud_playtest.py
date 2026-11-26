"""Regression fixtures cropped from the supplied Windows recording (not synthetic labels)."""
from pathlib import Path
import cv2
import numpy as np
import pytest
from raid_vision import ColossalHUD
from thebums_skill_scheduler import BloodSkillScheduler

ROOT=Path(__file__).resolve().parents[1]
NAMES={'form2':'colossal_form2','form3':'colossal_form3',
       'heat_ready':'colossal_heat_ready','seismic_local':'colossal_seismic_ready'}

def detector():
    return ColossalHUD({k:{'normal':cv2.imread(str(ROOT/'image'/f'{v}.png'))} for k,v in NAMES.items()})

def frame(name, scale=1., offset_y=0):
    patch=cv2.imread(str(ROOT/'tests'/'fixtures'/f'blood_hud_{name}.png'))
    a=np.zeros((1078,1918,3),np.uint8)
    patch=cv2.resize(patch,None,fx=scale,fy=scale)
    h,w=patch.shape[:2]
    x=957-w//2; y=950-h//2+offset_y
    a[y:y+h,x:x+w]=patch
    return a

@pytest.mark.parametrize('name,expected',[
    ('human',(False,None,None)),('human_late',(False,None,None)),
    ('shift_ready',(True,True,True)),('ready',(True,True,True)),
    ('cooldown',(True,False,True)),('cooldown_late',(True,False,True)),
    ('ready_again',(True,True,True))])
def test_actual_video_hud(name,expected):
    assert detector().observe(frame(name))==expected

@pytest.mark.parametrize('scale',[.45,.55,.65,.75,.85,.95,1.05,1.15,1.25,1.35,1.5])
def test_hud_at_other_ui_scales(scale):
    d=detector()
    assert d.observe(frame('shift_ready',scale))==(True,True,True)
    assert d.observe(frame('cooldown',scale))==(True,False,True)


def test_disappearance_is_not_cooldown():
    d=detector()
    assert d.observe(frame('ready'))==(True,True,True)
    assert d.observe(np.zeros((1078,1918,3),np.uint8))==(False,None,None)


def test_real_cooldown_observations_drive_scheduler():
    d=detector(); s=BloodSkillScheduler({'1':3,'3':6,'5':8},0,stable_for=.3,min_frames=2)
    def obs(t,name):
        hud,heat,seismic=d.observe(frame(name))
        return s.observe(t,hud=hud,heatwave=heat,seismic=seismic)
    assert obs(0,'ready')=='4'; s.sent('4',0)
    obs(.1,'cooldown'); action=obs(.45,'cooldown_late')
    assert s.confirmed_activations==1
    assert action=='5'
    s.sent('5',.45)
    obs(1,'ready_again'); obs(1.35,'ready_again')
    assert s.completed_cycles==1 and s.stage=='second_hidden'
