"""v11.5 scope: reject unclear currency cheaply; never block macro input."""
import ast
from pathlib import Path
from threading import Event
import numpy as np
import pytest
from grid114 import video_v114 as video
from grid114.live import LiveReview
from grid114.reward_summary import MissionCurrencyEvidence, RewardBook, reward_field


def geo(flags=()):
    return dict(flags=list(flags), span=20, gap=0, sharpness=1.)


@pytest.mark.parametrize('flags', [('low_quality',), ('invalid_crop',), ('no_text',),
                                    ('separated_text_or_overlay',), ('low_quality','text_near_crop_edge')])
def test_rejected_currency_never_invokes_hybrid_reader(monkeypatch, flags):
    reader=video.Reader.__new__(video.Reader)
    monkeypatch.setattr(video, 'geometry', lambda _: geo(flags))
    monkeypatch.setattr(reader, 'read', lambda *args: pytest.fail('unreadable crop decoded'))
    _,g,r=reader.quantity(np.zeros((100,100,3),np.uint8),(10,10,60,60),'gold')
    assert g['flags']==list(flags)
    assert r['candidate'] is None and r['method']=='quality_guard' and not r['ledger_allowed']


def test_clear_currency_retains_original_reader(monkeypatch):
    reader=video.Reader.__new__(video.Reader)
    monkeypatch.setattr(video,'geometry',lambda _:geo())
    result=dict(candidate=157,method='ai_fallback')
    monkeypatch.setattr(reader,'read',lambda *args:result)
    _,g,r=reader.quantity(np.zeros((100,100,3),np.uint8),(10,10,60,60),'gem')
    assert not g['flags'] and r is result


@pytest.mark.parametrize('expanded_value,expected',[(123456,123456),(23456,None)])
def test_edge_expansion_requires_agreement(monkeypatch,expanded_value,expected):
    reader=video.Reader.__new__(video.Reader)
    geometries=iter([geo(['text_near_crop_edge']),geo()])
    values=iter([123456,expanded_value])
    monkeypatch.setattr(video,'geometry',lambda _:next(geometries))
    monkeypatch.setattr(reader,'read',lambda *args:dict(candidate=next(values),method='ai_fallback'))
    _,g,r=reader.quantity(np.zeros((100,100,3),np.uint8),(10,10,60,60),'gold')
    assert r['candidate']==expected
    assert bool(g['flags'])==(expected is None)


def test_session_skips_alternate_for_unclear_gold_but_reads_gem_and_perks(monkeypatch):
    class StubReader:
        panel={'tpl':None,'slope':1,'bias':0}
        banks=[None]
        calls=[]
        def classify(self,tile):
            return {1:'gold',2:'gem',3:'other'}[int(tile[0,0,0])],.99,.99
        def quantity(self,frame,box,label):
            crop=np.full((20,40,3),int(frame[0,0,0]),np.uint8)
            return crop,geo(['low_quality'] if label=='gold' else []),dict(candidate=None if label=='gold' else 157,method='quality_guard' if label=='gold' else 'test')
        def read(self,crop,label):
            self.calls.append(label)
            return dict(candidate=157)
    reader=StubReader()
    monkeypatch.setattr(video,'match',lambda *a:1.)
    monkeypatch.setattr(video,'grid',lambda _: [[(0,5,20,20),(25,5,20,20),(50,5,20,20)]])
    monkeypatch.setattr(video,'read_perks',lambda *a:{'epic_perk':((6,1,2),)})
    monkeypatch.setattr(video,'preprocess',lambda *a:None)
    session=video.Session(reader,'test')
    for i in range(3):
        frame=np.zeros((100,100,3),np.uint8)
        frame[0,0]=i+10
        for j,x in enumerate((0,25,50),1): frame[5:25,x:x+20]=j
        rows,_=session.observe(frame,i,1+i*.25)
        assert rows[0]['value'] is None and rows[1]['value']==157
        assert session.perk_slots=={'epic_perk':((6,1,2),)}
    assert reader.calls==['gem']*3
    session.flush()
    evidence,book=MissionCurrencyEvidence(),RewardBook()
    evidence.add_events(session.events)
    assert evidence.finalize(book,1)=={'gem':157}
    assert book.credit(1,'epic_perk',2)
    assert book.snapshot(1)==({'gem':157,'epic_perk':2},{'gem':157,'epic_perk':2})
    assert '?' in reward_field(book.snapshot(1))['value']


def test_full_queue_drops_frames_without_waiting_for_reader(monkeypatch,tmp_path):
    entered,release=Event(),Event()
    def slow_run(self):
        entered.set()
        release.wait(5)
    monkeypatch.setattr(LiveReview,'_run',slow_run)
    review=LiveReview(lambda _:None,tmp_path)
    try:
        assert entered.wait(1)
        # Worker is deliberately blocked; submits must return without waiting for it.
        for i in range(13): review.submit(i,1)
        assert review.queue.qsize()==12 and review.dropped==1
        assert not release.is_set()
    finally:
        release.set();review.close()


def test_webhook_summary_wait_is_inside_background_worker():
    tree=ast.parse((Path(__file__).resolve().parents[1]/'main.py').read_text(encoding='utf-8'))
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='notify_mission_async')
    work=next(x for x in fn.body if isinstance(x,ast.FunctionDef) and x.name=='work')
    calls=[x for x in ast.walk(fn) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='webhook_snapshot']
    assert len(calls)==1 and calls[0] in list(ast.walk(work))
    assert any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='start' for x in ast.walk(fn))
