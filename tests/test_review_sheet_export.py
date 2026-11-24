"""Provenance and contact sheet unit checks (no Roblox or GUI)."""
import tempfile
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"developer"))
from ai_studio import video_source_hint
from review_sheet_export import iou, feature, nearest, pixel_digest


class ReviewExportTests(unittest.TestCase):
    def test_v9_path_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/"sources.csv").write_text("path,video,seconds,left,top,right,bottom\n"
                "val/nov23_trim/retry_text/a.png,old.mp4,240,1,2,3,4\n",encoding="utf-8")
            self.assertEqual(video_source_hint(root,Path("old.mp4")),("val","nov23_trim"))
            self.assertIsNone(video_source_hint(root,Path("unknown.mp4")))

    def test_overlap_and_color(self):
        self.assertGreater(iou([818,177,37,31],[818,175,37,30]),.8)
        self.assertEqual(iou([0,0,10,10],[20,20,10,10]),0)
        a=np.zeros((20,30,3),np.uint8)
        b=np.full_like(a,255)
        self.assertNotEqual(pixel_digest(a),pixel_digest(b))
        self.assertEqual(nearest(a,[("white",feature(b)),("black",feature(a))]),("black",0.0))


if __name__=="__main__":
    unittest.main()
