"""Exercise actual main.py functions without importing Windows input/GUI APIs."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from thebums_skill_scheduler import BloodSkillScheduler

ROOT=Path(__file__).resolve().parents[1]
TREE=ast.parse((ROOT/'main.py').read_text(encoding='utf-8'))

def function(name,env):
    node=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'main.py','exec'),env)
    return env[name]

class Stop(Exception): pass

def test_shift_uses_two_distinct_form_frames_and_stops_repeating_skill5():
    frames=iter(['human','colossal1','colossal2'])
    clock=SimpleNamespace(monotonic=lambda:0.,sleep=Mock())
    detector=Mock(); detector.observe.side_effect=[(False,None,None),(True,None,True),(True,False,True)]
    tap=Mock(); chest=Mock(); tick=Mock()
    f=function('blood_shift_until_confirmed',dict(time=clock, settings={'family':'Female Raid'},
        tv=lambda f,k:.5,tick=tick,grab_screen=lambda:next(frames),
        attack_titan_skip_visible=lambda f:False,check_colossal_chest=chest,
        tap=tap,log=Mock(),KEY_ABILITY_LAST='5'))
    assert f(detector)=='colossal2'
    tap.assert_called_once_with('5')
    assert tick.call_count==3 and detector.observe.call_count==3
    chest.assert_called_once_with('human')


def test_combat_consumes_confirmed_frame_without_redundant_wait_and_checks_chest():
    detector=Mock(); detector.templates={'seismic_local':object()}
    detector.observe.return_value=(True,True,True)
    chest=Mock(); tap=Mock(); tick=Mock(side_effect=Stop)
    f=function('blood_colossal_priority',dict(settings={'family':'Female Raid'},
        tv=lambda f,k:.35 if k=='heatwave_confirm' else 5.,
        time=SimpleNamespace(monotonic=lambda:0.,sleep=Mock()),
        BloodSkillScheduler=BloodSkillScheduler,KEY_ABILITY_3RD='4',KEY_ABILITY_LAST='5',
        KEY_ABILITY_2ND='3',KEY_GRAB='1',log=Mock(),tick=tick,
        attack_titan_skip_visible=lambda f:False,_attack_skip={},
        check_colossal_chest=chest,tap=tap))
    with pytest.raises(Stop): f(detector,'confirmed')
    detector.observe.assert_called_once_with('confirmed')
    chest.assert_called_once_with('confirmed')
    tap.assert_called_once_with('4')


def test_chest_guard_stops_combat_input_even_with_form_hud():
    detector=Mock(); detector.templates={'seismic_local':object()}
    detector.observe.return_value=(True,True,True)
    chest=Mock(side_effect=Stop); tap=Mock()
    f=function('blood_colossal_priority',dict(settings={'family':'Female Raid'},
        tv=lambda f,k:.35 if k=='heatwave_confirm' else 5.,
        time=SimpleNamespace(monotonic=lambda:0.,sleep=Mock()),
        BloodSkillScheduler=BloodSkillScheduler,KEY_ABILITY_3RD='4',KEY_ABILITY_LAST='5',
        KEY_ABILITY_2ND='3',KEY_GRAB='1',log=Mock(),tick=Mock(),
        attack_titan_skip_visible=lambda f:False,_attack_skip={},
        check_colossal_chest=chest,tap=tap))
    with pytest.raises(Stop): f(detector,'chest')
    tap.assert_not_called()

@pytest.mark.parametrize('family,opening,delay', [
    ('Female Raid','dash',2.), ('Attack Titan Raid Blood Awakening','2',1.5)])
def test_full_native_blood_opening_reaches_shift_and_combat(family,opening,delay):
    """Exercise the real caller: testing only the Shift callee missed undefined t."""
    from unittest.mock import call
    order=Mock(); detector=object(); order.detector.return_value=detector
    order.shift.return_value='confirmed_frame'
    def config(f,k):
        assert (f,k)==(family,'heatwave_threshold')
        return .8
    f=function('attack_titan_blood_routine', dict(settings={'family':family},
        FAMILY_LABELS={family:family},_raid={},tv=config,log=Mock(),raid_phase=Mock(),
        double_tap=order.dash,tap=order.tap,KEY_DASH='dash',KEY_ABILITY_1ST='2',
        attack_titan_wait=order.wait,ColossalHUD=order.detector,_raid_templates={},
        blood_shift_until_confirmed=order.shift,blood_colossal_priority=order.combat,
        AttackTitanChestFound=Stop,female_chests=Mock(),attack_titan_chests=Mock()))
    f()
    first=call.dash(opening) if family=='Female Raid' else call.tap(opening)
    assert order.mock_calls==[first,call.wait(delay),call.detector({},.8),
                              call.shift(detector),call.combat(detector,'confirmed_frame')]


def test_fritz_uses_shared_heatwave_scheduler_and_timer_seismic():
    combat=Mock()
    f=function('attack_titan_colossal',dict(settings={'family':'Attack Titan Raid Fritz'},
        ATTACK_BLOOD_KEY='Attack Titan Raid Blood Awakening',show=Mock(),
        blood_colossal_priority=combat))
    f()
    combat.assert_called_once_with(visual_seismic=False)
