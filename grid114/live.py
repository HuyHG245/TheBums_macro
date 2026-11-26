"""Process immutable watchdog frames on a bounded queue; write local review results."""
from datetime import datetime
import json
from collections import deque
from .reward_summary import RewardBook, SummaryRequest, MissionCurrencyEvidence
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
import time
from uuid import uuid4


class LiveReview:
    def __init__(self, log, output):
        self.log = log
        self.directory = Path(output)
        self.session_id = datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid4().hex[:8]
        self.file = self.directory / (self.session_id + '.jsonl')
        self.queue = Queue(maxsize=12)
        self.stopping = Event()
        self.dropped = 0
        self.thread = Thread(target=self._run, name='Grid114Review', daemon=True)
        self.thread.start()

    def submit(self, frame, mission):
        if self.stopping.is_set():
            return
        try:
            self.queue.put_nowait((time.monotonic(), frame, mission))
        except Full:
            self.dropped += 1

    def webhook_snapshot(self, mission):
        request = SummaryRequest(mission)
        if self.stopping.is_set():
            return None
        try:
            self.queue.put(request, timeout=10)
        except Full:
            return None
        if request.ready.wait(45):
            return request.snapshot
        return None

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=1)

    def _run(self):
        try:
            from .video_v114 import Reader, Session
            self.directory.mkdir(parents=True, exist_ok=True)
            with self.file.open('x', encoding='utf-8', buffering=1) as out:
                def record(kind, **data):
                    out.write(json.dumps(dict(kind=kind, **data)) + '\n')
                record('start', session=self.session_id)
                session = Session(Reader(), self.session_id)
                self.log('Grid v11.4 ready: ' + str(self.file))
                count = 0
                mission = 0
                book = RewardBook()
                evidence = MissionCurrencyEvidence()
                recent_perks = deque(maxlen=3)
                perk_values = {}
                def finish_perks():
                    for label, values in perk_values.items():
                        if len(values) == 1:
                            value = next(iter(values))
                            if book.credit(mission, label, value):
                                record('perk', mission=mission, label=label, value=value)
                    perk_values.clear()
                    recent_perks.clear()
                def report_events():
                    if session.events and mission:
                        if not evidence.add_events(session.events):
                            record('late_reward_evidence', mission=mission, count=len(session.events))
                    for event in session.events:
                        record('event', mission=mission, session_total=book.snapshot(mission)[1].get(event['label']), **event)
                        self.log(f"Grid review: {event['label']} {event['candidate']} ({event['status']})")
                    session.events.clear()
                def finish_currency():
                    for label, value in evidence.finalize(book, mission).items():
                        total = book.snapshot(mission)[1][label]
                        record('reward_credit', mission=mission, label=label, value=value, session_total=total)
                        self.log(f"Mission {mission}: +{value:,} {label} (session {total:,})")
                while not self.stopping.is_set() or not self.queue.empty():
                    try:
                        item = self.queue.get(timeout=.2)
                    except Empty:
                        continue
                    if isinstance(item, SummaryRequest):
                        if item.mission == mission:
                            session.flush()
                            report_events()
                            finish_currency()
                            finish_perks()
                        item.snapshot = book.snapshot(item.mission)
                        item.ready.set()
                        continue
                    timestamp, frame, captured_mission = item
                    if mission != captured_mission:
                        finish_perks()
                        session.flush()
                        report_events()
                        finish_currency()
                        evidence = MissionCurrencyEvidence()
                        mission = captured_mission
                    observations, _ = session.observe(frame, count, timestamp)
                    recent_perks.append((timestamp, session.perk_slots))
                    if len(recent_perks) == 3 and .39 <= timestamp - recent_perks[0][0] < 4:
                        for label, slots in session.perk_slots.items():
                            if all(reads.get(label) == slots for _, reads in recent_perks):
                                perk_values.setdefault(label, set()).add(sum(s[2] for s in slots))
                    for row in observations:
                        record('observation', **row)
                    report_events()
                    count += 1
                session.flush()
                report_events()
                finish_currency()
                finish_perks()
                record('stop', sampled=count, dropped=self.dropped)
        except Exception as exc:
            self.log(f'Grid v11.4 stopped: {type(exc).__name__}: {exc}')
        finally:
            self.stopping.set()
