import json
from grid114 import video_v114
from grid114.live import LiveReview


def test_worker_delays_credit_until_summary_and_handles_mission_rollover(monkeypatch, tmp_path):
    class StubSession:
        def __init__(self, reader, source):
            self.events = []
            self.perk_slots = {}
        def observe(self, frame, index, timestamp):
            if frame is not None:
                self.events.append(dict(event=str(index), slot='0:1', label='gold',
                                        candidate=frame, status='consistent_review',
                                        review_only=True, ledger_allowed=False))
            return [], []
        def flush(self):
            pass
    monkeypatch.setattr(video_v114, 'Reader', lambda: object())
    monkeypatch.setattr(video_v114, 'Session', StubSession)
    logs=[]
    review=LiveReview(logs.append,tmp_path)
    try:
        review.submit(999504,1)
        review.submit(999504,1)
        review.submit(99504,1)
        assert review.webhook_snapshot(1)==({}, {})
        review.submit(1019568,2)
        review.submit(1019568,2)
        assert review.webhook_snapshot(2)==({'gold':1019568},{'gold':1019568})
        review.submit(1019568,2)
        assert review.webhook_snapshot(2)==({'gold':1019568},{'gold':1019568})
        review.submit(1,3)
        review.submit(None,4)  # rollover finalizes mission 3 before the request
        assert review.webhook_snapshot(3)==({'gold':1},{'gold':1019569})
    finally:
        review.close()
    records=[json.loads(line) for line in review.file.read_text().splitlines()]
    assert [(r['mission'],r['value']) for r in records if r['kind']=='reward_credit']==[(2,1019568),(3,1)]
    assert any(r['kind']=='late_reward_evidence' for r in records)
    assert not any('stopped:' in line for line in logs)
