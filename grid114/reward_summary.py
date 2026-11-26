from threading import Event

LABELS = {'gold': 'Gold', 'gem': 'Gem', 'epic_perk': 'Epic Perk',
          'legendary_perk': 'Legendary Perk', 'mythic_perk': 'Mythic Perk'}


class SummaryRequest:
    def __init__(self, mission):
        self.mission = mission
        self.ready = Event()
        self.snapshot = None


class RewardBook:
    def __init__(self):
        self.missions = {}

    def credit(self, mission, label, amount):
        if type(mission) is not int or mission <= 0 or label not in LABELS or type(amount) is not int or amount < 0:
            return False
        amounts = self.missions.setdefault(mission, {})
        if label in amounts:
            return False
        amounts[label] = amount
        return True

    def snapshot(self, mission):
        # No observed reward is unknown, not a confirmed zero.
        totals = {}
        for key, amounts in self.missions.items():
            if key <= mission:
                for label, amount in amounts.items():
                    totals[label] = totals.get(label, 0) + amount
        return self.missions.get(mission, {}).copy(), totals


def reward_field(snapshot):
    current, totals = snapshot or ({}, {})
    lines = ['Reward          This run      Session']
    for key, name in LABELS.items():
        amount = f"+{current[key]:,}" if key in current else '?'
        total = f"{totals[key]:,}" if key in totals else '?'
        lines.append(f"{name:<15} {amount:>8} {total:>12}")
    return {'name': 'REWARDS', 'inline': False,
            'value': '```\n' + '\n'.join(lines) + '\n```\n? = unconfirmed · Session = confirmed total'}


class MissionCurrencyEvidence:
    """Stage review results until the caller explicitly ends a mission.

    Different event segments are not different raids. Repeated values are
    deduplicated; conflicting values or ambiguous same-event slots are withheld.
    """
    def __init__(self):
        self.values = {}
        self.conflicts = set()
        self.finalized = False

    def add_events(self, events):
        if self.finalized:
            return False
        groups = {}
        for e in events:
            if e['label'] in ('gold', 'gem'):
                groups.setdefault((e['event'], e['label']), []).append(e)
        for (_, label), seq in groups.items():
            if len(seq) != 1:
                self.conflicts.add(label)
                continue
            event = seq[0]
            if event['status'] == 'conflicting_candidates' or event.get('consensus_status') == 'conflicting_candidates':
                self.conflicts.add(label)
            value = event.get('candidate')
            if event['status'] == 'consistent_review' and type(value) is int and value >= 0:
                self.values.setdefault(label, set()).add(value)
        return True

    def finalize(self, book, mission):
        if self.finalized:
            return {}
        self.finalized = True
        credits = {}
        for label, values in self.values.items():
            if label not in self.conflicts and len(values) == 1:
                value = next(iter(values))
                if book.credit(mission, label, value):
                    credits[label] = value
        return credits
