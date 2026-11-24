"""Headless tests exercising real v1.3 recognition code with synthetic frames.
No mouse, keyboard, network, GUI, or Roblox is touched.
"""
import hashlib
import importlib
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSET_BASELINE = ROOT / 'developer' / 'recognition_assets.sha256'
sys.path.insert(0, str(ROOT))
from thebums_hybrid_ai import HybridAdvisor, CpuOnnxBackend, HybridDecision
import thebums_reconnect as rc


class CounterBackend:
    def __init__(self, value=0.999):
        self.value = value
        self.calls = []

    def score(self, crop, name):
        self.calls.append((name, crop.shape))
        return self.value


@pytest.fixture(scope='module')
def macro():
    # Linux CI cannot import Windows input drivers or use a real display.
    sys.modules.setdefault('mss', types.SimpleNamespace(mss=lambda: None))
    sys.modules.setdefault('pyautogui', types.SimpleNamespace(
        moveTo=lambda *a, **k: None,
        moveRel=lambda *a, **k: None,
        click=lambda *a, **k: None,
        keyUp=lambda *a, **k: None))
    return importlib.import_module('main')


@pytest.fixture
def macro_restore(macro, monkeypatch):
    monkeypatch.setitem(macro.settings, 'recognition_mode', 'standard')
    monkeypatch.setitem(macro.settings, 'fix_loot', False)
    monkeypatch.setattr(macro, '_scale', 1.0)
    monkeypatch.setattr(macro, '_hybrid_init_done', True)
    monkeypatch.setattr(macro, '_hybrid_advisor', None)
    return macro


def template(name):
    img = cv2.imread(str(ROOT / 'image' / (name + '.png')), cv2.IMREAD_COLOR)
    assert img is not None
    return img


def degrade(image, lower, upper):
    noise = np.random.default_rng(20261122).integers(0, 256, image.shape, dtype=np.uint8)
    for alpha in np.linspace(0.0, .6, 151):
        candidate = cv2.addWeighted(image, 1 - alpha, noise, alpha, 0)
        score = float(cv2.matchTemplate(candidate, image, cv2.TM_CCOEFF_NORMED)[0, 0])
        if lower <= score < upper:
            return candidate
    raise AssertionError(f'No candidate in [{lower}, {upper})')


def canvas(tpl, x=450, y=340, width=900, height=500):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    th, tw = tpl.shape[:2]
    frame[y:y+th, x:x+tw] = tpl
    return frame


def test_old_pngs_identical():
    # Portable even when the original ZIP is not beside the source folder.
    lines = [v.strip().split('  ', 1) for v in ASSET_BASELINE.read_text(encoding='utf-8').splitlines()
             if v.strip() and not v.startswith('#')]
    assert len(lines) == 56
    for original_sha, relative in lines:
        target = ROOT / relative
        assert target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == original_sha


def test_standard_retry_original_unchanged(macro_restore, monkeypatch):
    m = macro_restore
    tpl = template('retry_text')
    monkeypatch.setattr(m, '_retry_template', {'normal': tpl})
    monkeypatch.setattr(m, '_alt_retry_templates', [])
    backend = CounterBackend()
    monkeypatch.setattr(m, '_hybrid_advisor', HybridAdvisor(backend))
    clean = canvas(tpl)
    hit = m.find_retry(clean)
    assert hit and hit[0][2] >= .92
    assert not backend.calls  # Standard and high-confidence CV both skip AI
    uncertain = canvas(degrade(tpl, .882, .917))
    assert m.find_retry(uncertain) is None
    assert not backend.calls


def test_retry_beta_needs_roi_and_new_frame(macro_restore, monkeypatch):
    m = macro_restore
    tpl = template('retry_text')
    monkeypatch.setattr(m, '_retry_template', {'normal': tpl})
    monkeypatch.setattr(m, '_alt_retry_templates', [])
    backend = CounterBackend()
    monkeypatch.setattr(m, '_hybrid_advisor', HybridAdvisor(backend))
    monkeypatch.setitem(m.settings, 'recognition_mode', 'hybrid_beta')
    altered = degrade(tpl, .882, .917)
    screen = canvas(altered)
    res = m.find_retry(screen)
    assert res and .88 <= res[0][2] < .92 and res[1] == 'normal'
    assert backend.calls[0][0] == 'retry_text'
    assert m.find_retry(canvas(altered, x=100, y=340)) is None  # wrong screen area
    assert m.find_retry(canvas(degrade(tpl, .5, .7))) is None  # weak CV cannot be rescued

    # Foreground scan sees a borderline result, but the next capture lost it.
    captures = iter([screen, np.zeros_like(screen)])
    monkeypatch.setattr(m, 'grab_screen', lambda: next(captures))
    monkeypatch.setattr(m, 'check_grabbed', lambda frame: None)
    monkeypatch.setattr(m, 'RETRY_CONFIRM_DELAY', 0)
    m.scan_for_retry()  # must NOT raise RetryFound

    captures = iter([screen, screen])
    monkeypatch.setattr(m, 'grab_screen', lambda: next(captures))
    with pytest.raises(m.RetryFound):
        m.scan_for_retry()


def test_advisor_strict_gates_and_backend_failure():
    backend = CounterBackend()
    a = HybridAdvisor(backend)
    img = np.zeros((160, 250, 3), np.uint8)
    candidate = (100, 80, .90, 40, 20)
    assert a.verify(img, 'retry_text', candidate, .92, (0, 0, 250, 160)).accepted
    assert not a.verify(img, 'retry_text', (100, 80, .50, 40, 20), .92).accepted
    assert not a.verify(img, 'retry_text', (100, 80, .93, 40, 20), .92).accepted
    assert not a.verify(img, 'retry_text', candidate, .92, (0, 0, 90, 160)).accepted
    assert not a.verify(img, 'unknown', candidate, .92).accepted
    assert len(backend.calls) == 1
    backend.value = float('nan')
    assert not a.verify(img, 'retry_text', candidate, .92).accepted
    backend.value = 0.97
    assert not a.verify(img, 'retry_text', candidate, .92).accepted  # 0.98 cutoff


def test_reconnect_original_and_beta():
    tpl = template('rc_reconnect')
    backend = CounterBackend()
    matcher = rc.ImageMatcher(str(ROOT / 'image'), advisor=HybridAdvisor(backend))
    exact = matcher.find(canvas(tpl), 'reconnect')
    assert exact and exact.source == 'opencv' and not backend.calls
    altered = degrade(tpl, .87, .898)
    beta = matcher.find(canvas(altered), 'reconnect')
    assert beta and beta.source == 'hybrid_ai'
    assert beta.name == 'reconnect' and beta.confidence < .90
    assert backend.calls[-1][0] == 'rc_reconnect'
    standard = rc.ImageMatcher(str(ROOT / 'image'))
    assert standard.find(canvas(altered), 'reconnect') is None
    assert standard.find(canvas(tpl), 'reconnect') is not None


def test_reconnect_roi_cannot_be_bypassed():
    tpl = template('rc_start')
    backend = CounterBackend()
    matcher = rc.ImageMatcher(str(ROOT / 'image'), advisor=HybridAdvisor(backend))
    # rc_start is restricted to central-lower button, not an arbitrary START.
    outside = canvas(degrade(tpl, .85, .878), x=50, y=50)
    assert matcher.find(outside, 'start') is None
    assert not backend.calls


def test_loot_beta_multi_frame_and_single_frame_rejection(macro_restore, monkeypatch):
    m = macro_restore
    tpl = template('loot_scroll')
    altered = degrade(tpl, .782, .818)
    screen = canvas(altered, x=460, y=240)
    backend = CounterBackend()
    monkeypatch.setattr(m, '_hybrid_advisor', HybridAdvisor(backend))
    monkeypatch.setitem(m.settings, 'recognition_mode', 'hybrid_beta')
    monkeypatch.setattr(m, '_loot_templates', {'normal': {'scroll': tpl}})
    monkeypatch.setattr(m, '_loot_extra_templates', {'normal': {}})
    monkeypatch.setattr(m, '_result_templates', {})
    monkeypatch.setattr(m, 'active_loot_kinds', lambda: ['scroll'])
    # Fast mode provides a single cached frame and cannot use Hybrid AI.
    single, _ = m.scan_end_screen(quality='normal', want_result=False, frame=screen)
    assert single == {}
    assert not backend.calls
    # Real multiple captures verify the same location.
    counts = [0]
    def grab():
        counts[0] += 1
        return screen
    monkeypatch.setattr(m, 'grab_screen', grab)
    monkeypatch.setattr(m, 'wait', lambda *args, **kwargs: None)
    found, _ = m.scan_end_screen(quality='normal', want_result=False, scan_time=.06)
    assert counts[0] >= 2
    assert 'scroll' in found
    assert any(c[0] == 'loot_scroll' for c in backend.calls)


def test_no_validated_model_keeps_standard(macro_restore, monkeypatch):
    m = macro_restore
    monkeypatch.setitem(m.settings, 'recognition_mode', 'hybrid_beta')
    monkeypatch.setattr(m, '_hybrid_init_done', False)
    assert m.get_hybrid_advisor() is None
    assert m.settings['recognition_mode'] == 'standard'


def test_missing_manifest_and_no_internet(tmp_path):
    with pytest.raises(FileNotFoundError):
        CpuOnnxBackend(tmp_path)
    (tmp_path / 'hybrid_manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='schema_version'):
        CpuOnnxBackend(tmp_path)


def test_cpu_onnx_model_contract_with_fake_runtime(monkeypatch, tmp_path):
    # A fake ONNX Runtime isolates the preprocessing/provider contract. It is
    # not a substitute for running a real trained model on Windows.
    import json
    import sys
    (tmp_path / 'tiny.onnx').write_bytes(b'fake-onnx-for-contract-test')
    (tmp_path / 'hybrid_manifest.json').write_text(json.dumps({
        'schema_version': 1, 'validated': True, 'calibrated': True,
        'model': 'tiny.onnx', 'labels': ['__background__', 'retry_text'],
        'input_hw': [64, 96],
    }))
    calls = {}
    class Options:
        intra_op_num_threads = None
        inter_op_num_threads = None
        graph_optimization_level = None
    class FakeSession:
        def __init__(self, path, sess_options, providers):
            calls['providers'] = providers
            calls['options'] = sess_options
        def get_inputs(self): return [types.SimpleNamespace(name='image')]
        def get_outputs(self): return [types.SimpleNamespace(name='probabilities')]
        def run(self, outputs, feed):
            arr = feed['image']
            calls['input_shape'] = arr.shape
            assert np.isfinite(arr).all() and arr.dtype == np.float32
            return [np.array([[0.001, 0.999]], dtype=np.float32)]
    stub = types.SimpleNamespace(
        SessionOptions=Options, InferenceSession=FakeSession,
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL='opt'))
    monkeypatch.setitem(sys.modules, 'onnxruntime', stub)
    backend = CpuOnnxBackend(tmp_path)
    patch = np.zeros((40, 80, 3), dtype=np.uint8)
    assert backend.score(patch, 'retry_text') == pytest.approx(.999, abs=1e-5)
    assert backend.score(patch, 'loot_scroll') is None
    assert calls['providers'] == ['CPUExecutionProvider']
    assert calls['options'].intra_op_num_threads == 1
    assert calls['options'].inter_op_num_threads == 1
    assert calls['input_shape'] == (1, 3, 64, 96)


def test_unvalidated_or_unsafe_manifest_rejected(tmp_path):
    import json
    (tmp_path / 'unused.onnx').write_bytes(b'dummy')
    manifest = {
        'schema_version': 1, 'validated': False, 'calibrated': True,
        'model': 'unused.onnx', 'labels': ['__background__', 'retry_text']}
    p = tmp_path / 'hybrid_manifest.json'
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='validated'):
        CpuOnnxBackend(tmp_path)
    manifest['validated'] = True
    manifest['model'] = '../outside.onnx'
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='local'):
        CpuOnnxBackend(tmp_path)
    manifest['model'] = 'unused.onnx'
    manifest['labels'] = ['retry_text']  # must include negative class
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='background'):
        CpuOnnxBackend(tmp_path)


def test_switch_from_beta_to_standard_does_not_require_ai(macro_restore, monkeypatch):
    m = macro_restore
    monkeypatch.setattr(m, 'running', lambda: False)
    monkeypatch.setattr(m, 'save_stats', lambda: None)
    def missing_model():
        m.settings['recognition_mode'] = 'standard'
        return None
    monkeypatch.setattr(m, 'get_hybrid_advisor', missing_model)
    class FakeFlag:
        value = True
        def get(self): return self.value
        def set(self, v): self.value = v
    obj = types.SimpleNamespace(v_hybrid=FakeFlag(), say=lambda *a, **k: None)
    m.SettingsWindow.on_hybrid_toggle(obj)
    assert m.settings['recognition_mode'] == 'standard'
    # Real method's explicit missing-model guard must force user back to Standard.
    assert obj.v_hybrid.get() is False
    obj.v_hybrid.set(False)
    m.SettingsWindow.on_hybrid_toggle(obj)
    assert m.settings['recognition_mode'] == 'standard'


def test_loot_beta_rejects_nonconsecutive_false_hits(macro_restore, monkeypatch):
    m = macro_restore
    tpl = template('loot_scroll')
    screen = canvas(degrade(tpl, .782, .818), x=460, y=240)
    empty = np.zeros_like(screen)
    monkeypatch.setattr(m, '_hybrid_advisor', HybridAdvisor(CounterBackend()))
    monkeypatch.setitem(m.settings, 'recognition_mode', 'hybrid_beta')
    monkeypatch.setattr(m, '_loot_templates', {'normal': {'scroll': tpl}})
    monkeypatch.setattr(m, '_loot_extra_templates', {'normal': {}})
    monkeypatch.setattr(m, '_result_templates', {})
    monkeypatch.setattr(m, 'active_loot_kinds', lambda: ['scroll'])
    n = [0]
    def grab():
        n[0] += 1
        return screen if n[0] % 2 else empty
    monkeypatch.setattr(m, 'grab_screen', grab)
    monkeypatch.setattr(m, 'wait', lambda *args, **kwargs: None)
    found, _ = m.scan_end_screen(quality='normal', want_result=False, scan_time=.09)
    assert n[0] >= 4
    assert found == {}


def test_background_wgc_callback_copies_bgr_and_stops(macro_restore, monkeypatch):
    """Exercise the real callback/reader contract without a Windows desktop."""
    m = macro_restore
    sessions = []

    class FakeCapture:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.stopped = False
            sessions.append(self)

        def event(self, callback):
            setattr(self, callback.__name__, callback)
            return callback

        def start_free_threaded(self):
            pixels = np.zeros((15, 24, 4), dtype=np.uint8)
            pixels[:, :, 0] = 31  # B
            pixels[:, :, 1] = 42  # G
            pixels[:, :, 2] = 53  # R
            pixels[:, :, 3] = 255
            self.on_frame_arrived(types.SimpleNamespace(frame_buffer=pixels), None)
            pixels[:, :, 0] = 0  # native buffer can be reused after callback
            return types.SimpleNamespace(stop=lambda: setattr(self, 'stopped', True))

    monkeypatch.setattr(m, 'windows_capture', types.SimpleNamespace(WindowsCapture=FakeCapture))
    session = m._BackgroundWgcFrames(123)
    frame = session.latest(timeout=0.01)
    assert frame.shape == (15, 24, 3)
    assert tuple(frame[0, 0]) == (31, 42, 53)
    assert sessions[0].options['window_hwnd'] == 123
    assert sessions[0].options['cursor_capture'] is False
    session.stop()
    assert sessions[0].stopped


def test_background_wgc_closed_session_is_rejected(macro_restore, monkeypatch):
    m = macro_restore

    class ClosingCapture:
        def __init__(self, **kwargs):
            pass

        def event(self, callback):
            setattr(self, callback.__name__, callback)
            return callback

        def start_free_threaded(self):
            self.on_closed()
            return types.SimpleNamespace(stop=lambda: None)

    monkeypatch.setattr(m, 'windows_capture',
                        types.SimpleNamespace(WindowsCapture=ClosingCapture))
    session = m._BackgroundWgcFrames(123)
    with pytest.raises(m.BackgroundModeUnavailable, match='closed'):
        session.latest(timeout=0.01)
    session.stop()


def test_background_wgc_rejects_blank_client_frame(macro_restore):
    m = macro_restore
    import threading
    background = object.__new__(m._BackgroundRobloxWindow)
    background.lock = threading.RLock()
    background.hwnd = 111
    background._wgc_hwnd = 111
    background._wgc = types.SimpleNamespace(
        latest=lambda: np.zeros((40, 60, 3), dtype=np.uint8))
    with pytest.raises(m.BackgroundModeUnavailable, match='blank or uniform'):
        background._capture_wgc({'width': 60, 'height': 40})


# Leave-only Roblox disconnections (including Error 282 in the user's recording).
# These fixtures are synthetic: the original image/ files remain untouched,
# and the tests never open Roblox, send a click, or create screenshot files.
def _roblox_client_frames():
    frames = [np.zeros((1080, 1920, 3), np.uint8) for _ in range(4)]

    def paste(dst, anchor, x, y):
        icon = rc._ui_template(anchor)
        dst[y:y + icon.shape[0], x:x + icon.shape[1], :] = cv2.cvtColor(
            icon, cv2.COLOR_GRAY2BGR)

    cv2.rectangle(frames[0], (775,595), (1155,635), (255,255,255), -1)
    paste(frames[0], 'disconnect_title', 891,415)
    paste(frames[0], 'leave_button', 916,597)
    paste(frames[1], 'home_game_title', 127,848)
    cv2.rectangle(frames[2], (789,252), (1018,301), (247,87,54), -1)
    # The real Roblox detail page uses a BLUE button with a WHITE Play
    # triangle. A blue rectangle alone must never pass the safety check.
    cv2.fillConvexPoly(frames[2], np.array([
        [894,264], [894,288], [917,276]], dtype=np.int32), (255,255,255))
    paste(frames[2], 'game_detail_title', 780,157)
    return frames


def test_leave_dialog_template_requires_heading_and_button():
    screens=_roblox_client_frames()
    hit=rc.find_leave_disconnect(screens[0])
    assert hit and hit.name=='leave' and hit.source=='opencv'
    assert (hit.x,hit.y)==(957,615)
    assert rc.find_leave_disconnect(screens[1]) is None
    assert rc.find_leave_disconnect(screens[2]) is None
    without_title=screens[0].copy()
    without_title[410:455,880:1050]=0
    assert rc.find_leave_disconnect(without_title) is None
    without_button=screens[0].copy()
    without_button[580:660,770:1170]=0
    assert rc.find_leave_disconnect(without_button) is None


def test_roblox_home_card_and_game_play_must_match_correct_game():
    frames=_roblox_client_frames()
    game=rc.find_roblox_home_game(frames[1])
    assert game and game.name=='home_game' and (game.x,game.y)==(204,774)
    play=rc.find_roblox_game_play(frames[2])
    assert play and play.name=='game_play' and abs(play.x-904)<=2 and abs(play.y-276)<=2
    assert rc.find_roblox_game_play(frames[1]) is None
    without_blue=frames[2].copy()
    without_blue[250:305,787:1022]=0
    assert rc.find_roblox_game_play(without_blue) is None


def test_leave_navigator_requires_successive_verified_screens():
    frames=_roblox_client_frames()
    class FakeSlot:
        def slot_a_select(self,frame):
            return rc.Match('slot_select',640,645,1.,110,25) if frame is frames[3] else None

    state={'index':0}
    clicked=[]
    def capture():return frames[state['index']]
    def click(hit):
        clicked.append(hit.name)
        state['index']+=1
    navigator=rc.RobloxLeaveNavigator(FakeSlot(),capture,click,lambda:None,lambda *a:None)
    assert navigator.run(first_frame=frames[0])
    assert clicked==['leave','home_game','game_play']

    # If the visible screen changes before interacting, fail closed.
    class MovedTarget(rc.RobloxLeaveNavigator):
        def _wait(self,state,timeout,stable=2):
            return rc.find_leave_disconnect(frames[0])
    did_click=[]
    moved=MovedTarget(FakeSlot(),lambda:frames[1],did_click.append,lambda:None,lambda *a:None)
    with pytest.raises(rc.ReconnectError,match='target moved'):
        moved._click_verified('leave','Leave disconnected game','home_game',3)
    assert not did_click


def test_party_retry_requires_both_labels_in_the_results_panel(macro, monkeypatch):
    retry = cv2.imread(str(ROOT / 'image' / 'party_retry_word.png'))
    modifiers = cv2.imread(str(ROOT / 'image' / 'party_modifiers_word.png'))
    monkeypatch.setattr(macro, '_party_retry_pair', (retry, modifiers))
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    h, w = retry.shape[:2]
    frame[420:420+h, 660:660+w] = retry
    assert macro._match_party_retry(frame, [1.0]) is None
    mh, mw = modifiers.shape[:2]
    frame[420:420+mh, 472:472+mw] = modifiers
    match, quality, scale = macro._match_party_retry(frame, [1.0])
    assert match[:2] == (660+w//2, 420+h//2)
    assert quality == 'normal' and scale == 1
    frame[420:420+h, 660:660+w] = 0
    assert macro._match_party_retry(frame, [1.0]) is None
