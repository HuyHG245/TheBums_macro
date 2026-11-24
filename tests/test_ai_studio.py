"""Offline tests for the Studio archive safeguards (no game, no GUI)."""
from pathlib import Path
import sys
import zipfile
import unittest
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "developer"))
from ai_studio import import_baseline, import_videos


class StudioImportTests(unittest.TestCase):
    def test_baseline_preserves_splits_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive = root / "baseline.zip"
            with zipfile.ZipFile(archive, "w") as z:
                for split in ("train", "val", "test"):
                    z.writestr(f"wrapped/{split}/session_{split}/loot_body/sample.png", split)
                z.writestr("wrapped/sources.csv", "split,session,source\n")
            target = root / "v10"
            self.assertEqual(import_baseline(archive, target, reserve=0), 3)
            self.assertEqual((target / "test/session_test/loot_body/sample.png").read_text(), "test")
            self.assertTrue((target / "sources.csv").is_file())
            with self.assertRaises(FileExistsError):
                import_baseline(archive, target, reserve=0)

    def test_video_import_deduplicates_and_ignores_nonvideo(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive = root / "videos.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("folder/original.mp4", b"fake test bytes")
                z.writestr("folder/copy.mp4", b"fake test bytes")
                z.writestr("not_a_video.exe", b"ignored")
            paths = import_videos(archive, root / "imported", reserve=0)
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].read_bytes(), b"fake test bytes")
            self.assertEqual(import_videos(archive, root / "imported", reserve=0), paths)

    def test_reject_unsafe_archive_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../escape.mp4", b"bad")
            with self.assertRaises(ValueError):
                import_videos(archive, root / "imported", reserve=0)
            self.assertFalse((root / "escape.mp4").exists())


if __name__ == "__main__":
    unittest.main()
