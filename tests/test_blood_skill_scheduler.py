from thebums_skill_scheduler import BloodSkillScheduler


def make():
    return BloodSkillScheduler({'1': 3, '3': 6, '5': 8}, 0,
                               stable_for=.30, min_frames=3)


def obs(s, t, heat=True, hud=True):
    return s.observe(t, hud=hud, heatwave=heat)


def stable(s, start, heat):
    for dt in (0, .16, .32):
        result = obs(s, start + dt, heat)
    return result


def activate(s):
    assert obs(s, 0) == '4'
    s.sent('4', 0)
    stable(s, .1, False)
    assert s.stage == 'first_visible'
    assert s.confirmed_activations == 1


def test_failed_initial_attempt_retries_without_lower_skills():
    s = make()
    assert obs(s, 0) == '4'
    s.sent('4', 0)
    assert obs(s, .2) is None
    assert obs(s, .5) == '4'
    assert s.confirmed_activations == 0
    s.sent('4', .5)
    stable(s, .6, False)
    assert s.confirmed_activations == 1


def test_two_cycles_and_priority_then_retry_without_reset():
    s = make()
    activate(s)
    assert stable(s, 1, True) == '5'
    assert s.completed_cycles == 1
    s.sent('5', 1.32)
    stable(s, 2, False)
    assert s.stage == 'true_ready'
    assert stable(s, 10, True) == '4'
    assert s.completed_cycles == 2
    s.sent('4', 10.32)
    assert obs(s, 10.6) is None
    assert obs(s, 11) == '4'
    assert s.completed_cycles == 2
    s.sent('4', 11)
    stable(s, 11.1, False)
    assert s.completed_cycles == 0
    assert s.confirmed_activations == 2


def test_one_action_and_independent_timers():
    s = make()
    activate(s)
    assert obs(s, .5, False) == '5'
    s.sent('5', .5)
    assert obs(s, .51, False) is None
    assert obs(s, .7, False) == '3'
    s.sent('3', .7)
    assert obs(s, .9, False) == '1'
    s.sent('1', .9)
    assert s.ready_at == {'5': 8.5, '3': 6.7, '1': 3.9}
    assert obs(s, 4, False) == '1'


def test_cutscene_cannot_complete_activation_or_cycle():
    s = make()
    assert obs(s, 0) == '4'
    s.sent('4', 0)
    for t in (.1, .2, 1, 3):
        assert obs(s, t, False, False) is None
    assert s.confirmed_activations == 0
    assert obs(s, 4, True) == '5'
    s.sent('5', 4)
    assert stable(s, 4.2, True) != '4'
    assert s.interrupted_heat_phase == 0
    assert stable(s, 5, False) != '4'
    assert stable(s, 6, True) != '4'
    assert stable(s, 7, False) != '4'
    assert stable(s, 8, True) == '4'
    s.sent('4', 8.32)
    stable(s, 8.5, False)
    assert s.confirmed_activations == 1


def test_hidden_noise_and_interrupted_debounce_do_not_confirm():
    s = make()
    assert obs(s, 0) == '4'
    s.sent('4', 0)
    obs(s, .1, False)
    obs(s, .2, False)
    obs(s, .3, False, False)
    obs(s, 5, False)
    assert s.confirmed_activations == 0
    obs(s, 5.4, False)
    assert s.confirmed_activations == 0
    obs(s, 5.5, True)
    assert s.confirmed_activations == 0


def test_skill5_timer_not_reset_by_heatwave_retry():
    s = make()
    activate(s)
    s.sent('5', .5)
    old = s.ready_at.copy()
    stable(s, 1, True)
    stable(s, 2, False)
    assert stable(s, 3, True) == '4'
    s.sent('4', 3.32)
    assert s.ready_at == old


def visual(s, t, seismic, heat=False, hud=True):
    return s.observe(t, hud=hud, heatwave=heat, seismic=seismic)


def visual_stable(s, start, seismic, heat=False):
    for dt in (0, .16, .32):
        result = visual(s, start+dt, seismic, heat)
    return result


def test_seismic_readiness_preempts_legacy_timer_and_lower_skills():
    s = make()
    activate(s)
    s.ready_at['5'] = 100
    assert visual(s, 1., True) is None
    assert visual(s, 1.16, True) is None
    assert visual(s, 1.32, True) == '5'
    s.sent('5', 1.32)
    assert visual(s, 1.6, True) is None
    assert visual(s, 1.83, True) == '5'
    s.sent('5', 1.83)
    assert visual(s, 2., False) is None
    assert visual(s, 2.16, False) is None
    assert visual(s, 2.32, False) == '3'
    assert not s.seismic_pending
    assert 'unverified' in s.last_event
    s.sent('3', 2.32)
    assert visual_stable(s, 3., True) == '5'


def test_no_ready_evidence_does_not_fire_timer_or_block_lower_skills():
    s = make()
    activate(s)
    assert visual(s, 1, False) == '3'
    s.sent('3', 1)
    assert visual(s, 2, False) == '1'
    s.sent('1', 2)
    assert visual(s, 3, False) is None


def test_seismic_cutscene_requires_fresh_ready_evidence():
    s = make()
    activate(s)
    assert visual_stable(s, 1, True) == '5'
    s.sent('5', 1.32)
    for t in (2, 3, 4):
        assert visual(s, t, False, hud=False) is None
    assert s.seismic_pending
    assert visual(s, 5, True) is None
    assert visual(s, 5.16, True) is None
    assert visual(s, 5.32, True) == '5'


def test_heatwave_priority_when_both_ready_after_second_cycle():
    s = make()
    activate(s)
    visual_stable(s, 1, False, heat=True)
    visual_stable(s, 2, False, heat=False)
    assert s.stage == 'true_ready'
    assert visual(s, 3, True, heat=True) is None
    assert visual(s, 3.16, True, heat=True) is None
    assert visual(s, 3.32, True, heat=True) == '4'
    s.sent('4', 3.32)
    assert visual(s, 3.6, True, heat=True) is None
    assert visual(s, 3.9, True, heat=True) == '4'


def test_seismic_priority_when_heatwave_has_only_one_cycle():
    s = make()
    activate(s)
    assert visual_stable(s, 1, True, heat=True) == '5'
    assert s.completed_cycles == 1
    s.sent('5', 1.32)
    assert visual(s, 1.6, True, heat=True) is None
    assert visual(s, 1.9, True, heat=True) == '5'
    assert s.completed_cycles == 1


def test_interrupted_heatwave_attempt_not_confirmed_by_later_absence():
    s = make()
    assert obs(s, 0) == '4'
    s.sent('4', 0)
    assert obs(s, .1, False, False) is None
    assert stable(s, 1, False) == '5'
    assert s.confirmed_activations == 0
    assert s.stage == 'activation_pending'
    s.sent('5', 1.32)
    assert stable(s, 2, True) != '4'
    assert s.interrupted_heat_phase == 2
    assert stable(s, 3, False) != '4'
    assert stable(s, 4, True) == '4'
    s.sent('4', 4.32)
    stable(s, 4.42, False)
    assert s.confirmed_activations == 1


def test_long_pause_does_not_confirm_old_keypress():
    s = make()
    assert obs(s, 0) == '4'
    s.sent('4', 0)
    obs(s, .1, False)
    stable(s, 10, False)
    assert s.stage == 'activation_pending'
    assert s.confirmed_activations == 0


def test_interrupted_second_activation_preserves_completed_cycles():
    s = make()
    activate(s)
    stable(s, 1, True)
    stable(s, 2, False)
    assert stable(s, 3, True) == '4'
    s.sent('4', 3.32)
    obs(s, 3.4, False, False)
    stable(s, 4, False)
    assert s.completed_cycles == 2
    assert s.confirmed_activations == 1
    assert stable(s, 5, True) != '4'
    assert stable(s, 6, False) != '4'
    assert stable(s, 7, True) == '4'
    s.sent('4', 7.32)
    stable(s, 7.42, False)
    assert s.completed_cycles == 0
    assert s.confirmed_activations == 2


def test_long_randomized_state_invariants():
    # Synthetic stress: not a simulation of actual skill cooldown durations.
    import random
    rng = random.Random(261126)
    s = make()
    now = 0.
    last_sent = -100.
    for _ in range(10000):
        now += rng.choice([.08,.12,.18,.3,2.2])
        hud = rng.random() > .13
        old_cycles = s.completed_cycles
        old_confirmed = s.confirmed_activations
        old_timers = s.ready_at.copy()
        action = s.observe(now,hud=hud,heatwave=rng.random()>.4,seismic=rng.random()>.6)
        assert s.completed_cycles in (0,1,2)
        assert s.confirmed_activations>=old_confirmed
        if not hud:
            assert action is None
            assert s.completed_cycles==old_cycles
            assert s.confirmed_activations==old_confirmed
        if s.completed_cycles < old_cycles:
            assert s.confirmed_activations == old_confirmed + 1
        if action:
            assert action in s.priority and now-last_sent >= s.action_gap-1e-9
            s.sent(action,now)
            last_sent=now
            if action=='4':
                assert s.ready_at==old_timers
            else:
                assert all(s.ready_at[k]==old_timers[k] for k in old_timers if k!=action)


def test_failed_heatwave_retries_do_not_starve_other_skills():
    s=make(); assert obs(s,0)=='4'; s.sent('4',0)
    assert obs(s,.5)=='4'; s.sent('4',.5)
    for t,expected in ((.7,'5'),(.9,'3'),(1.1,'1'),(2.,'4')):
        assert obs(s,t)==expected
        s.sent(expected,t)
    assert s.confirmed_activations==0 and s.completed_cycles==0


def test_unknown_heat_observation_is_not_cooldown_and_allows_other_skills():
    s=make(); assert obs(s,0)=='4'; s.sent('4',0)
    assert s.observe(.2,hud=True,heatwave=None)=='5'; s.sent('5',.2)
    assert s.observe(.4,hud=True,heatwave=None)=='3'; s.sent('3',.4)
    assert s.observe(.6,hud=True,heatwave=None)=='1'; s.sent('1',.6)
    assert s.confirmed_activations==0


def test_failed_seismic_retries_do_not_starve_other_skills():
    s=make(); activate(s)
    assert visual_stable(s,1,True)=='5'; s.sent('5',1.32)
    assert visual(s,1.9,True)=='5'; s.sent('5',1.9)
    assert visual(s,2.1,True)=='3'; s.sent('3',2.1)
    assert visual(s,2.3,True)=='1'; s.sent('1',2.3)


def test_cutscene_no_further_edges_allows_other_skills():
    s=make(); s.sent('4',0); obs(s,.1,hud=False)
    actions=[]
    for i in range(1,200):
        t=1+i*.2
        a=obs(s,t,True)
        if a:
            actions.append(a); s.sent(a,t)
    assert set(actions)=={'1','3','5'}


def test_second_cutscene_breaks_recovery_sequence():
    s=make(); s.sent('4',0); obs(s,.1,hud=False)
    stable(s,1,False); stable(s,2,True)
    assert s.interrupted_heat_phase==2
    obs(s,2.5,hud=False)
    assert s.interrupted_heat_phase==0
    assert stable(s,3,False)!='4'
    assert stable(s,4,True)!='4'


def test_verified_second_cooldown_survives_cutscene():
    s=make(); activate(s); stable(s,1,True); stable(s,2,False)
    assert s.stage=='true_ready'
    obs(s,3,hud=False)
    assert obs(s,4,True)!='4'
    assert obs(s,4.16,True)!='4'
    assert obs(s,4.32,True)=='4'


def test_missing_first_ready_never_claims_two_cycles():
    s=make(); activate(s); obs(s,1,hud=False)
    assert stable(s,2,True)!='4'
    assert s.completed_cycles==1
    assert stable(s,3,True)!='4'


def test_unknown_during_recovery_requires_new_sequence():
    s=make(); s.sent('4',0); obs(s,.1,hud=False)
    stable(s,1,False); stable(s,2,True)
    s.observe(2.5,hud=True,heatwave=None)
    assert s.interrupted_heat_phase==0
    assert stable(s,3,False)!='4'
    assert stable(s,4,True)!='4'
