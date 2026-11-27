"""Offline tests for the standalone Failsafe Level 2 runtime."""
from pathlib import Path
import unittest
from thebums_extra_failsafe import Hit, LeaveMatcher, LeaveNavigator, LeaveError, level2_due
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
if not (ROOT/'image').is_dir():ROOT=ROOT.parent

class Clock:
    now=0.
    def __call__(self):return self.now
    def sleep(self, seconds):self.now+=seconds

class FakeMatcher:
    def identify(self, stage):
        return (stage, Hit(10,20,.99,10,10)) if stage in ('menu','leave','confirm') else (None,None)

class TestLevel2(unittest.TestCase):
    def test_mission_boundary(self):
        for t in (10,60,120,180,300,3600):
            with self.subTest(t=t):
                self.assertFalse(level2_due(100,t,100+2*t-.001))
                self.assertTrue(level2_due(100,t,100+2*t))
    def test_inactive(self):
        self.assertFalse(level2_due(100,120,500,active=False))
    def test_blocked(self):
        self.assertFalse(level2_due(100,120,500,blocked=True))
    def test_missing_timer(self):
        self.assertFalse(level2_due(None,120,500))
        self.assertFalse(level2_due(100,None,500))
    def test_invalid_timeout(self):
        for t in (0,-1,float('inf'),float('nan')):self.assertFalse(level2_due(100,t,500))
    def route(self, start=0, freeze=None, stop_at=None):
        clock=Clock(); stages=['menu','leave','confirm','handoff']; pos=[start]; clicks=[]
        def capture():return stages[pos[0]]
        def click(hit):
            clicks.append(stages[pos[0]])
            if stages[pos[0]]!=freeze:pos[0]+=1
        def check():
            if stop_at is not None and len(clicks)>=stop_at:raise SystemExit
        nav=LeaveNavigator(FakeMatcher(),capture,click,check,lambda f:f=='handoff',
                          clock=clock,sleep=clock.sleep,step_timeout=2,exit_timeout=3)
        return nav, clicks
    def test_full_route(self):
        nav,clicks=self.route();self.assertTrue(nav.run());self.assertEqual(clicks,['menu','leave','confirm'])
    def test_resume_menu_open(self):
        nav,clicks=self.route(start=1);self.assertTrue(nav.run());self.assertEqual(clicks,['leave','confirm'])
    def test_resume_confirmation(self):
        nav,clicks=self.route(start=2);self.assertTrue(nav.run());self.assertEqual(clicks,['confirm'])
    def test_already_handoff(self):
        nav,clicks=self.route(start=3);self.assertTrue(nav.run());self.assertEqual(clicks,[])
    def test_stale_menu_no_double_click(self):
        nav,clicks=self.route(freeze='menu')
        with self.assertRaises(LeaveError):nav.run()
        self.assertEqual(clicks,['menu'])
    def test_stale_confirm_no_double_click(self):
        nav,clicks=self.route(freeze='confirm')
        with self.assertRaises(LeaveError):nav.run()
        self.assertEqual(clicks,['menu','leave','confirm'])
    def test_stop_unwinds(self):
        nav,clicks=self.route(stop_at=1)
        with self.assertRaises(SystemExit):nav.run()
        self.assertEqual(clicks,['menu'])
    def test_no_visual_target(self):
        clock=Clock(); clicks=[]
        nav=LeaveNavigator(FakeMatcher(),lambda:None,clicks.append,lambda:None,lambda _:False,
                          clock=clock,sleep=clock.sleep,step_timeout=1)
        with self.assertRaises(LeaveError):nav.run()
        self.assertEqual(clicks,[])
    def test_moved_target(self):
        clock=Clock(); clicks=[]; i=[0]
        class M:
            def identify(self,_):
                i[0]+=1
                return 'menu',Hit(10 if i[0]<4 else 100,20,.99,10,10)
        nav=LeaveNavigator(M(),lambda:None,clicks.append,lambda:None,lambda _:False,
                          clock=clock,sleep=clock.sleep,step_timeout=1)
        with self.assertRaises(LeaveError):nav.run()
        self.assertEqual(clicks,[])
    def test_blank_frame(self):
        m=LeaveMatcher(lambda name:ROOT/'image'/name)
        self.assertEqual(m.identify(np.zeros((1080,1920,3),np.uint8)),(None,None))
    def test_missing_template(self):
        with self.assertRaises(LeaveError):LeaveMatcher(lambda name:ROOT/'missing'/name)
    def test_confirm_requires_cancel_and_blue(self):
        m=LeaveMatcher(lambda name:ROOT/'image'/name)
        f=np.zeros((1080,1920,3),np.uint8)
        def paste(name,x,y):
            im=cv2.imread(str(ROOT/'image'/name));h,w=im.shape[:2];f[y:y+h,x:x+w]=im
        paste('fs_confirm_dialog.png',712,335)
        paste('fs_confirm_leave.png',825,462)
        self.assertEqual(m.identify(f),(None,None))
    def test_logo_requires_menu_neighbor(self):
        m=LeaveMatcher(lambda name:ROOT/'image'/name)
        f=np.zeros((1080,1920,3),np.uint8)
        im=cv2.imread(str(ROOT/'image'/'fs_roblox_menu.png'));h,w=im.shape[:2];f[44:44+h,25:25+w]=im
        self.assertEqual(m.identify(f),(None,None))

if __name__=='__main__':unittest.main()
