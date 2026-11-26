from grid114.video_v114 import finish_event

def rows():
    return [dict(event='a',slot='0:1',label='gold',stream='clip',frame=i,t=1+i*.25,hash=str(i),value=999504,flags=[],sharpness=i+1,icon_conf=.9,span=20,roi_quality=1.) for i in range(3)]

def test_observed_terminal_slot_can_be_consistent():
    r=finish_event(rows(),terminal_frame=2)[0]
    assert r['candidate']==999504 and r['status']=='consistent_review'
    assert r['ledger_allowed'] is False

def test_slot_missing_from_last_confirmed_panel_cannot_credit_old_value():
    r=finish_event(rows(),terminal_frame=3)[0]
    assert r['candidate'] is None
    assert r['terminal_reason']=='slot_absent_on_last_frame'

def test_conflict_not_erased_by_terminal_rejection():
    rs=rows();rs[-1]['value']=99504
    r=finish_event(rs,terminal_frame=3)[0]
    assert r['candidate'] is None and r['consensus_status']=='conflicting_candidates'
