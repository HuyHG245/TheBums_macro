"""The Bums AI Studio: local, developer-only video review and training launcher.

No macro input, no automatic ground-truth labels, no network except the explicit
Install CPU training dependencies action. All generated files are gitignored.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import queue
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "datasets" / "ai_studio"
DATASET = ROOT / "datasets" / "hybrid_v10"
RUNS = ROOT / "training_runs"
GIB = 1024 ** 3
VIDEO_TYPES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp"}
SPLITS = ("train", "val", "test")
SESSION = re.compile(r"[A-Za-z0-9_-]{4,80}")
LABEL = re.compile(r"[A-Za-z0-9_]{1,80}")
METADATA = {"sources.csv", "_video_sources.json", "README.txt", "dataset_report.json"}
PYTHON = sys.executable


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _digest(stream)


def _digest(stream):
    sha = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        sha.update(block)
    return sha.hexdigest()


def require_space(folder, bytes_needed=0, reserve=6 * GIB):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(folder).free < bytes_needed + reserve:
        raise OSError("Not enough free disk space. Keep at least 6 GB free after this operation.")


def zip_parts(info):
    name = info.filename
    if "\\" in name or ":" in name or name.startswith("/") or "\x00" in name:
        raise ValueError("Unsafe archive member: " + name)
    parts = PurePosixPath(name).parts
    if any(part in ("..", ".") for part in parts):
        raise ValueError("Unsafe archive path: " + name)
    if stat.S_ISLNK(info.external_attr >> 16):
        raise ValueError("Symbolic links are not accepted in archives")
    return parts


def copy_member(archive, info, dest):
    """Stream one approved member; no extractall or archive-controlled paths."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as src, dest.open("xb") as dst:
        remaining = info.file_size
        while remaining:
            block = src.read(min(1024 * 1024, remaining))
            if not block:
                raise OSError("Truncated archive member")
            dst.write(block)
            remaining -= len(block)
        if src.read(1):
            raise OSError("Archive member exceeded declared size")


def import_baseline(archive_path, target, *, reserve=6 * GIB):
    """Copy a versioned dataset once; original ZIP and existing datasets stay untouched."""
    target = Path(target)
    if target.exists():
        raise FileExistsError("Working dataset already exists. Refusing to overwrite it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        entries, prefixes = [], set()
        for info in archive.infolist():
            parts = zip_parts(info)
            if info.is_dir() or len(parts) < 4 or Path(parts[-1]).suffix.lower() not in IMAGE_TYPES:
                continue
            split, session, label, filename = parts[-4:]
            if split not in SPLITS:
                continue
            if not SESSION.fullmatch(session) or not LABEL.fullmatch(label):
                raise ValueError("Invalid session or label in baseline: " + info.filename)
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", filename):
                raise ValueError("Unsupported crop filename: " + filename)
            prefixes.add(parts[:-4])
            entries.append((info, (split, session, label, filename)))
        if len(prefixes) != 1 or {parts[0] for _, parts in entries} != set(SPLITS):
            raise ValueError("Expected one dataset containing train/val/test/session/label/image folders")
        prefix = next(iter(prefixes))
        metadata = {}
        for info in archive.infolist():
            parts = zip_parts(info)
            if info.is_dir() or not parts or parts[-1] not in METADATA:
                continue
            # Prefer files next to the split directories, otherwise a unique
            # parent-level source manifest (some dataset ZIPs wrap the data).
            if parts[:-1] == prefix:
                metadata[parts[-1]] = info
            elif prefix[:len(parts)-1] == parts[:-1]:
                metadata.setdefault(parts[-1], info)
        total = sum(info.file_size for info, _ in entries) + sum(i.file_size for i in metadata.values())
        if total > 2 * GIB or len(entries) > 100000:
            raise ValueError("Baseline exceeds the Studio import limit")
        require_space(target.parent, total, reserve)
        stage = Path(tempfile.mkdtemp(prefix="studio_import_", dir=target.parent))
        try:
            for split in SPLITS:
                (stage / split).mkdir()
            for info, parts in entries:
                copy_member(archive, info, stage.joinpath(*parts))
            for name, info in metadata.items():
                copy_member(archive, info, stage / name)
            (stage / "_studio_baseline.json").write_text(
                json.dumps({"archive_sha256": digest(archive_path), "crops": len(entries)}, indent=2),
                encoding="utf-8")
            stage.rename(target)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    return len(entries)


def import_videos(archive_path, destination, *, reserve=6 * GIB):
    """Extract only video members, stream to disk, deduplicate by full content hash."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    created = []
    paths = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            entries = []
            for info in archive.infolist():
                parts = zip_parts(info)
                if not info.is_dir() and parts and Path(parts[-1]).suffix.lower() in VIDEO_TYPES:
                    entries.append((info, parts[-1]))
            if not entries:
                raise ValueError("No supported video in this ZIP")
            total = sum(info.file_size for info, _ in entries)
            if total > 10 * GIB or len(entries) > 500:
                raise ValueError("Archive exceeds 10 GB / 500 video import limit")
            require_space(destination, total, reserve)
            for info, name in entries:
                # Do not use paths from the archive. Keep readable source filename
                # after the digest directory so old provenance can be reviewed.
                safe = re.sub(r"[^A-Za-z0-9_.() -]", "_", name).strip(" .")
                safe = safe[-140:] or ("video" + Path(name).suffix.lower())
                with tempfile.TemporaryDirectory(prefix="import_", dir=destination) as temp:
                    tmp = Path(temp) / "video.bin"
                    copy_member(archive, info, tmp)
                    key = digest(tmp)
                    folder = destination / key
                    prior = next((p for p in folder.glob("*") if p.is_file()), None)
                    if prior is not None:
                        if digest(prior) != key:
                            raise ValueError("Existing imported video failed checksum")
                        paths.append(prior)
                        continue
                    folder.mkdir(exist_ok=True)
                    final = folder / safe
                    tmp.rename(final)
                    created.append(final)
                    paths.append(final)
        return list(dict.fromkeys(paths))
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
            try:
                path.parent.rmdir()
            except OSError:
                pass
        raise


def video_source_hint(root, video):
    """Only return an unambiguous filename match in existing provenance."""
    path = Path(root) / "sources.csv"
    if not path.is_file():
        return None
    matches = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            row = {str(k).strip().lower(): str(v or "") for k, v in raw.items() if k is not None}
            if not any(v.replace("\\", "/").rsplit("/", 1)[-1].lower() == Path(video).name.lower()
                       for v in row.values()):
                continue
            split = row.get("split", "")
            session = row.get("session", row.get("session_id", ""))
            if split in SPLITS and SESSION.fullmatch(session):
                matches.add((split, session))
    return next(iter(matches)) if len(matches) == 1 else None


def review_video(video, dataset):
    # Use existing collector, audit label allowlist, and exact source registry.
    from video_dataset_collector import VideoCollector, get_registry, validate_rect
    from video_review_index import load_index
    from hybrid_dataset import allowed_labels, BACKGROUND

    class StudioCollector(VideoCollector):
        def move_suggestion(self, direction):
            if not self.suggestions:
                return super().move_suggestion(direction)
            super().move_suggestion(direction)
            event = self.suggestions[self.suggestion_index]
            # Slider callbacks may not fire when consecutive suggestions share time.
            self.load_frame(float(event["seconds"]))
            self.label.set(BACKGROUND)
            if event.get("context_only") or event["label"] not in allowed_labels() or self.frame is None:
                return
            x, y, w, h = event["bbox"]
            try:
                rect = validate_rect((x, y, x + w, y + h), self.frame.shape)
            except ValueError:
                return
            self.rect = rect
            self.label.set(event["label"])  # tentative, never auto-saved
            self.privacy_checked.set(False)
            self.drag_item = self.canvas.create_rectangle(
                *(round(v * self.scale) for v in rect), outline="#26b8e7", width=2)
            self.status.set(
                f"Suggestion {self.suggestion_index+1}/{len(self.suggestions)}: "
                f"{event['label']} ({event['score']:.3f}). Correct the label/box if needed; "
                "confirm label and privacy before Save. A wrong match is a hard negative "
                "only if no other allowed target is present.")

        def save(self):
            if self.frame is None or self.rect is None or not self.privacy_checked.get():
                return super().save()
            registered = get_registry(self.dataset).get(self.video_digest)
            if registered is None:
                hint = video_source_hint(self.dataset, self.video)
                if hint is not None and hint != (self.split.get(), self.session.get().strip()):
                    messagebox.showwarning("Existing source", f"Baseline source belongs to {hint[0]}/{hint[1]}. "
                                           "Use the same split/session; do not create a new one.", parent=self.root)
                    return
                if not messagebox.askyesno("Confirm recording provenance",
                        "Is this the original recording for the selected session/split? "
                        "A copy, trim or re-encode is NOT an independent session. "
                        "For old recordings use the v9 source's original split/session. "
                        "If unsure, cancel and inspect sources.csv.", parent=self.root):
                    return
            return super().save()

    window = tk.Tk()
    window.withdraw()
    try:
        app = StudioCollector(window, Path(video), Path(dataset))
        registered = get_registry(dataset).get(app.video_digest)
        hint = video_source_hint(dataset, video)
        if registered is not None:
            app.split.set(registered["split"])
            app.session.set(registered["session"])
        elif hint is not None:
            app.split.set(hint[0])
            app.session.set(hint[1])
        index = Path(dataset) / ("review_" + app.video_digest[:12] + ".json")
        if index.is_file():
            app.suggestions = load_index(index, source_sha256=app.video_digest, duration=app.duration)
            if app.suggestions:
                app.move_suggestion(1)
        window.deiconify()
        window.mainloop()
    except Exception as exc:
        messagebox.showerror("Cannot open review", str(exc), parent=window)
        window.destroy()
        raise


class Studio:
    def __init__(self, window):
        self.window = window
        self.window.title("The Bums AI Studio | Developer Only")
        self.window.geometry("1000x720")
        self.window.minsize(820, 580)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        WORK.mkdir(parents=True, exist_ok=True)
        self.events = queue.Queue()
        self.busy = False
        self.stop_requested = threading.Event()
        self.process = None
        self.buttons = []
        self.paths = self.load_queue()
        self.status = tk.StringVar(value="Ready")
        self.stats = tk.StringVar()
        self.fps = tk.StringVar(value="0.5")
        self.epochs = tk.StringVar(value="35")
        self.fingerprint_cache = {}

        outer = ttk.Frame(window, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="THE BUMS AI STUDIO", font=("Segoe UI", 19, "bold")).pack(anchor="w")
        ttk.Label(outer, text="Private developer tools  |  CPU-only  |  Original macro unchanged").pack(anchor="w")
        ttk.Label(outer, textvariable=self.stats, wraplength=900).pack(anchor="w", pady=(6, 10))
        tools = ttk.Frame(outer)
        tools.pack(fill="x")
        for title, cmd in (
            ("1. Import v9 ZIP", self.baseline), ("2. Add MP4 / videos", self.add_videos),
            ("Add video ZIP", self.video_zip), ("Remove selected", self.remove_selected),
        ):
            self.button(tools, title, cmd)
        self.listbox = tk.Listbox(outer, height=7, selectmode="extended", exportselection=False)
        self.listbox.pack(fill="x", pady=(8, 6))
        scan = ttk.Frame(outer)
        scan.pack(fill="x")
        ttk.Label(scan, text="Samples/sec:").pack(side="left")
        ttk.Combobox(scan, textvariable=self.fps, values=("0.5", "1.0", "2.0"),
                     width=5, state="readonly").pack(side="left", padx=4)
        self.button(scan, "3. Find candidates (selected/all)", self.scan)
        self.button(scan, "4. Review one video", self.review)
        ttk.Label(outer, text="Review is mandatory. Suggested labels are not training truth. "
                  "Use the original recording session and split.", wraplength=900).pack(anchor="w", pady=5)
        actions = ttk.Frame(outer)
        actions.pack(fill="x")
        self.button(actions, "5. Audit release data", self.audit)
        self.button(actions, "Install CPU training tools", self.install)
        ttk.Label(actions, text="Epochs:").pack(side="left", padx=(8, 0))
        ttk.Spinbox(actions, from_=1, to=200, textvariable=self.epochs, width=5).pack(side="left", padx=4)
        self.button(actions, "6. Train if audit passes", self.train)
        lower = ttk.Frame(outer)
        lower.pack(fill="x", pady=6)
        self.button(lower, "Open dataset / provenance", lambda: self.open_folder(DATASET))
        self.button(lower, "Open training reports", lambda: self.open_folder(RUNS))
        self.button(lower, "Open imported video folder", lambda: self.open_folder(WORK / "videos"))
        self.cancel_button = ttk.Button(lower, text="Cancel active task", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="right")
        logbox = ttk.Frame(outer)
        logbox.pack(fill="both", expand=True)
        self.log = tk.Text(logbox, height=12, wrap="word", state="disabled", font=("Consolas", 10))
        bar = ttk.Scrollbar(logbox, command=self.log.yview)
        self.log.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        ttk.Label(outer, textvariable=self.status).pack(anchor="w", pady=4)
        self.refresh()
        self.window.after(100, self.poll)

    def button(self, frame, title, command):
        b = ttk.Button(frame, text=title, command=command)
        b.pack(side="left", padx=(0, 5), pady=2)
        self.buttons.append(b)

    def load_queue(self):
        try:
            data = json.loads((WORK / "video_queue.json").read_text(encoding="utf-8"))
            return [str(p) for p in data if isinstance(p, str) and Path(p).is_file()]
        except (OSError, ValueError, TypeError):
            return []

    def refresh(self):
        self.listbox.delete(0, "end")
        for p in self.paths:
            self.listbox.insert("end", p)
        tmp = WORK / "video_queue.tmp"
        tmp.write_text(json.dumps(self.paths, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(WORK / "video_queue.json")
        n = sum(1 for split in SPLITS for p in (DATASET / split).rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_TYPES)
        disk = shutil.disk_usage(WORK).free / GIB
        self.stats.set(f"Dataset v10: {n} reviewed crops | Videos queued: {len(self.paths)} | "
                       f"Disk free: {disk:.1f} GB | Baseline v9 is never overwritten")

    def append(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", str(line) + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def run(self, description, work):
        if self.busy:
            return
        self.busy = True
        self.stop_requested.clear()
        self.status.set(description)
        self.append("\n--- " + description + " ---")
        for b in self.buttons:
            b.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        def worker():
            try:
                result = work()
                self.events.put(("done", result))
            except Exception as exc:
                self.events.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self.append(value)
                else:
                    self.busy = False
                    for b in self.buttons:
                        b.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
                    if kind == "done" and isinstance(value, list):
                        self.paths = list(dict.fromkeys(self.paths + [str(p) for p in value]))
                    self.refresh()
                    self.status.set("Task complete" if kind == "done" else "Stopped / action required")
                    self.append("Finished." if kind == "done" else "ERROR: " + value)
        except queue.Empty:
            pass
        self.window.after(100, self.poll)

    def emit(self, msg):
        self.events.put(("log", str(msg)))

    def command(self, args, *, check=True):
        if self.stop_requested.is_set():
            raise RuntimeError("Cancelled")
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                    "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                    "CUDA_VISIBLE_DEVICES": ""})
        proc = subprocess.Popen([PYTHON, *map(str, args)], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                encoding="utf-8", errors="replace", bufsize=1)
        self.process = proc
        try:
            if self.stop_requested.is_set():
                proc.terminate()
            for line in proc.stdout:
                self.emit(line.rstrip())
            code = proc.wait()
        finally:
            self.process = None
        if self.stop_requested.is_set():
            raise RuntimeError("Cancelled")
        if check and code:
            raise RuntimeError(f"Command exited {code}. Read the log above.")
        return code

    def baseline(self):
        path = filedialog.askopenfilename(title="Select original AOTR_Hybrid_Dataset_v9.zip",
                                         filetypes=[("ZIP dataset", "*.zip")])
        if path:
            def work():
                count = import_baseline(path, DATASET)
                self.emit(f"Imported {count} reviewed crops into {DATASET}. "
                          "Verify source history before adding old videos.")
            self.run("Import v9 as v10 working dataset", work)

    def add_videos(self):
        paths = filedialog.askopenfilenames(title="Select original gameplay recordings",
                    filetypes=[("Videos", "*.mp4 *.mov *.mkv *.avi *.webm")])
        if paths:
            self.paths = list(dict.fromkeys(self.paths + list(paths)))
            self.refresh()

    def video_zip(self):
        path = filedialog.askopenfilename(title="Select video ZIP", filetypes=[("ZIP", "*.zip")])
        if path:
            self.run("Import video archive without copying existing video files twice",
                     lambda: import_videos(path, WORK / "videos"))

    def remove_selected(self):
        for idx in reversed(self.listbox.curselection()):
            del self.paths[idx]
        self.refresh()
        self.append("Removed from queue only. Source videos were not deleted.")

    def ensure_data(self):
        if not DATASET.is_dir():
            messagebox.showinfo("Dataset required", "Import your v9 dataset ZIP first.")
            return False
        return True

    def scan(self):
        if not self.ensure_data():
            return
        selected = [self.paths[i] for i in self.listbox.curselection()] or self.paths[:]
        fps = float(self.fps.get())
        if not selected:
            messagebox.showinfo("No video", "Add or import at least one video.")
            return
        def work():
            for video in selected:
                if self.stop_requested.is_set():
                    raise RuntimeError("Cancelled")
                key = digest(video)
                index = DATASET / ("review_" + key[:12] + ".json")
                if index.is_file():
                    try:
                        previous = json.loads(index.read_text(encoding="utf-8"))
                        if previous.get("source_sha256") == key and previous.get("sample_fps") == fps:
                            self.emit(f"Reusing candidate index: {Path(video).name}")
                            continue
                    except (OSError, ValueError):
                        pass
                self.emit(f"Scanning {Path(video).name}; suggestions only")
                self.command(["developer/video_review_index.py", "--video", video,
                              "--out-dir", DATASET, "--fps", fps, "--threshold", "0.77",
                              "--min-gap", "2.5"])
        self.run("Find candidate timestamps (no screenshots written)", work)

    def review(self):
        if not self.ensure_data():
            return
        selected = self.listbox.curselection()
        if len(selected) != 1:
            messagebox.showinfo("Select one video", "Choose exactly one recording to review.")
            return
        path = self.paths[selected[0]]
        self.run("Review selected video; close review window to return",
                 lambda: self.command(["developer/ai_studio.py", "--review", path, "--root", DATASET]))

    def audit(self):
        if self.ensure_data():
            self.run("Audit real data and report release deficits",
                     lambda: self.command(["developer/hybrid_dataset.py", "audit", "--root", DATASET, "--release"]))

    def install(self):
        if not messagebox.askyesno("Install CPU-only packages",
                "Requires internet and several GB of temporary/free space. "
                "No CUDA packages or system-wide installations. Continue?"):
            return
        def work():
            require_space(WORK, 4 * GIB)
            self.command(["-m", "pip", "install", "--no-cache-dir", "--index-url",
                          "https://download.pytorch.org/whl/cpu", "torch>=2.4,<3"])
            self.command(["-m", "pip", "install", "--no-cache-dir",
                          "onnx>=1.16,<2", "onnxruntime>=1.20,<2"])
            self.command(["-c", "import torch, onnx, onnxruntime; "
                          "assert torch.version.cuda is None; print(torch.__version__, 'CPU only')"])
        self.run("Install optional CPU training dependencies", work)

    def train(self):
        if not self.ensure_data():
            return
        try:
            epochs = int(self.epochs.get())
            if not 1 <= epochs <= 200:
                raise ValueError()
        except ValueError:
            messagebox.showwarning("Epochs", "Enter 1–200.")
            return
        def work():
            self.command(["developer/hybrid_dataset.py", "audit", "--root", DATASET, "--release"])
            require_space(WORK, GIB)
            self.command(["-c", "import torch, onnx, onnxruntime; assert torch.version.cuda is None"])
            out = RUNS / time.strftime("v10_%Y%m%d_%H%M%S")
            if out.exists():
                raise FileExistsError(out)
            code = self.command(["developer/train_hybrid.py", "--root", DATASET,
                                 "--out", out, "--epochs", epochs], check=False)
            self.emit(f"Local reports: {out}")
            if code:
                raise RuntimeError("Trainer did not approve a model. Read the report/log; nothing deployed.")
            self.emit("Trainer passed its dataset/model gates. Independent Windows validation is still required.")
            self.emit("No model was copied to the macro or uploaded.")
        self.run("Train CNN on CPU only after release audit passes", work)

    def open_folder(self, path):
        if path == DATASET and not path.exists():
            messagebox.showinfo("No dataset", "Import v9 first.")
            return
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))

    def cancel(self):
        if not self.busy:
            return
        self.stop_requested.set()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.append("Cancellation requested. A ZIP import may need to finish its current operation.")

    def close(self):
        if self.busy:
            messagebox.showinfo("Task active", "Finish/cancel the active task before closing Studio.")
            return
        self.window.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--root", type=Path, default=DATASET)
    args = parser.parse_args()
    if args.review:
        review_video(args.review, args.root)
    else:
        root = tk.Tk()
        Studio(root)
        root.mainloop()


if __name__ == "__main__":
    main()
