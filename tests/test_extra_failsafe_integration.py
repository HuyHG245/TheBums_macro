"""Exercise the actual integration functions without importing Windows GUI or sending input."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock
import thebums_extra_failsafe as extra_fs

ROOT=Path(__file__).resolve().parent
if not (ROOT/'image').exists():ROOT=ROOT.parent

def runtime_source():
    p=ROOT/'main.py'
    if not p.exists():
        from integration import RUNTIME
        return RUNTIME
    tree=ast.parse(p.read_text(encoding='utf-8'))
    names={'ExtraFailsafe','extra_failsafe_check','extra_failsafe_recover'}
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    assert len(selected)==3
    return compile(ast.Module(body=selected,type_ignores=[]),str(p),'exec')

class TestIntegration(TestCase):
    def setUp(self):
        self.state={'mission_t0':1.,'fs2_started':10.,'fs2_timeout':120.,'rescuing':True,'quit':False,'paused':False}
        self.clock=Mock(return_value=250.)
        self.env={'Disconnected':type('Disconnected',(Exception,),{}), 'state':self.state,
                  'time':SimpleNamespace(monotonic=self.clock,sleep=Mock()),'extra_fs':extra_fs,
                  'settings':{'family':'Universal','reconnect':{'enabled':True}},'COLOSSAL_RAIDS':{'Female Raid','Attack Titan Raid'},
                  'scan_for_retry':Mock(),'grab_screen':Mock(return_value='frame'),'raid_terminal_heading':Mock(return_value=None),
                  'RETRY_CONFIRM_DELAY':.4,'wait_raid_terminal_retry':Mock(),'check_colossal_chest':Mock(),
                  '_skip_spam':{'until':100.},'_custom_held':{'x'},'release_all_keys':Mock(),
                  'pydirectinput':SimpleNamespace(keyUp=Mock()),'log':Mock(),'notify_async':Mock(),'show':Mock(),
                  'tick':Mock(),'frame_to_screen':Mock(return_value=(10,20)),'click_middle':Mock(),
                  'image_rc':SimpleNamespace(find_leave_disconnect=Mock(),find_roblox_home_game=Mock(),find_roblox_game_play=Mock()),
                  '_disconnect_template':{'normal':None},'find_quality_template':Mock(), 'RECONNECT_THRESHOLD':.9,
                  'asset_path':lambda n:ROOT/'image'/n,'capture_error_snapshot':Mock(),
                  'reconnect_rejoin':Mock(return_value=True)}
        # Use real deadline math, but simulate the verified UI navigator here.
        self.nav=Mock(); self.nav.run.return_value=True
        self.env['extra_fs']=SimpleNamespace(level2_due=extra_fs.level2_due, LeaveError=extra_fs.LeaveError,
                                            LeaveMatcher=Mock(),LeaveNavigator=Mock(return_value=self.nav))
        exec(runtime_source(), self.env)
    def test_deadline_raises_to_unwind_level1(self):
        with self.assertRaises(self.env['ExtraFailsafe']):self.env['extra_failsafe_check']()
        self.env['scan_for_retry'].assert_called_once_with(check_grab=False)
        self.env['click_middle'].assert_not_called()
    def test_before_deadline_no_capture(self):
        self.clock.return_value=249.99
        self.env['extra_failsafe_check']();self.env['grab_screen'].assert_not_called()
    def test_blocked_during_chest_terminal_retry_reconnect(self):
        for flag in ('raid_chest_active','raid_terminal_wait','retry_processing','reconnecting'):
            self.state[flag]=True;self.env['extra_failsafe_check']();self.state[flag]=False
        self.env['scan_for_retry'].assert_not_called()
    def test_retry_priority(self):
        class Retry(Exception):pass
        self.env['scan_for_retry'].side_effect=Retry
        with self.assertRaises(Retry):self.env['extra_failsafe_check']()
    def test_terminal_priority(self):
        self.env['raid_terminal_heading'].return_value='completed'
        self.env['extra_failsafe_check']();self.env['wait_raid_terminal_retry'].assert_called_once_with('completed')
    def test_ambiguous_terminal_no_leave(self):
        self.env['raid_terminal_heading'].side_effect=['completed',None]
        self.env['extra_failsafe_check']();self.env['wait_raid_terminal_retry'].assert_not_called()
    def test_raid_chest_priority(self):
        class Chest(Exception):pass
        self.env['settings']['family']='Female Raid';self.env['check_colossal_chest'].side_effect=Chest
        with self.assertRaises(Chest):self.env['extra_failsafe_check']()
    def test_success_clears_stale_mission_then_rejoins(self):
        def run():
            self.assertTrue(self.state['reconnecting']);self.assertIsNone(self.state['mission_t0'])
            self.assertIsNone(self.state['fs2_started']);self.assertFalse(self.state['rescuing']);return True
        self.nav.run.side_effect=run
        self.assertTrue(self.env['extra_failsafe_recover']());self.assertFalse(self.state['reconnecting'])
        self.env['reconnect_rejoin'].assert_called_once();self.env['release_all_keys'].assert_called()
        self.assertEqual(self.env['_custom_held'],set())
    def test_leave_failure_no_rejoin(self):
        self.nav.run.side_effect=extra_fs.LeaveError('timeout')
        self.assertFalse(self.env['extra_failsafe_recover']());self.env['reconnect_rejoin'].assert_not_called()
        self.assertFalse(self.state['reconnecting'])
    def test_rejoin_failure_returns_false(self):
        self.env['reconnect_rejoin'].return_value=False
        self.assertFalse(self.env['extra_failsafe_recover']())
    def test_reconnect_disabled_respected(self):
        self.env['settings']['reconnect']['enabled']=False
        self.assertFalse(self.env['extra_failsafe_recover']());self.env['reconnect_rejoin'].assert_not_called()
    def test_stop_unwinds_without_rejoin(self):
        self.nav.run.side_effect=SystemExit
        with self.assertRaises(SystemExit):self.env['extra_failsafe_recover']()
        self.assertFalse(self.state['reconnecting']);self.env['reconnect_rejoin'].assert_not_called()

if __name__=='__main__':main()
