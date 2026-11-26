from grid114.reward_summary import RewardBook, reward_field

def test_unknown_totals_remain_unknown():
    book = RewardBook()
    assert book.snapshot(1) == ({}, {})
    text = reward_field(book.snapshot(1))['value']
    assert 'Gold' in text and '0' not in text
    assert book.credit(1, 'gold', 999504)
    assert book.snapshot(1) == ({'gold': 999504}, {'gold': 999504})
    assert '?' in reward_field(book.snapshot(1))['value']

def test_zero_only_when_observed_and_duplicate_rejected():
    book = RewardBook()
    assert book.credit(1, 'gem', 0)
    assert not book.credit(1, 'gem', 157)
    assert book.credit(2, 'gem', 157)
    assert book.snapshot(1)[1] == {'gem': 0}
    assert book.snapshot(2)[1] == {'gem': 157}

def test_gold_gem_epic_session_regression():
    book = RewardBook()
    for label, value in [('gold',999504), ('gem',157), ('epic_perk',4)]:
        assert book.credit(1,label,value)
        assert not book.credit(1,label,value)
    assert book.snapshot(2) == ({}, {'gold':999504, 'gem':157, 'epic_perk':4})
    for value in (None, -1, '157', True, 1.2):
        assert not book.credit(2,'gem',value)
    assert not book.credit(2,'unknown',1)
    assert not book.credit(0,'gold',1)

from grid114.reward_summary import MissionCurrencyEvidence

def event(key, label, value, status='consistent_review', **extra):
    return dict(event=key, slot='1', label=label, candidate=value, status=status, **extra)

def test_currency_consensus_deduplicates_across_event_segments():
    book, ev = RewardBook(), MissionCurrencyEvidence()
    ev.add_events([event('a','gold',999504),event('a','gem',157)])
    ev.add_events([event('b','gold',999504),event('b','gem',157)])
    assert book.snapshot(1) == ({}, {})
    assert ev.finalize(book,1) == {'gold':999504,'gem':157}
    assert not ev.add_events([event('late','gold',1)])
    assert ev.finalize(book,1) == {}
    assert book.snapshot(1)[1] == {'gold':999504,'gem':157}

def test_conflicting_segments_withheld_without_affecting_other_label():
    book, ev = RewardBook(), MissionCurrencyEvidence()
    ev.add_events([event('a','gold',999504),event('a','gem',157)])
    ev.add_events([event('b','gold',99504)])
    assert ev.finalize(book,1) == {'gem':157}

def test_multiple_same_label_slots_or_conflicting_temporal_reads_withheld():
    book, ev = RewardBook(), MissionCurrencyEvidence()
    ev.add_events([event('a','gold',999504),event('a','gold',999504)])
    ev.add_events([event('b','gem',None,'uncertain_terminal_panel',consensus_status='conflicting_candidates')])
    ev.add_events([event('c','gold',999504),event('c','gem',157)])
    assert ev.finalize(book,1) == {}
