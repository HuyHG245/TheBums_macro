"""Private, developer-only AOT:R video crop labeling tool.

Launch with: python developer/video_dataset_collector.py
Requires the ordinary application dependencies: OpenCV and NumPy; Tkinter ships
with Windows Python. Nothing is sent over the network or into Roblox.

Only an explicitly selected small ROI is written, AFTER a manual privacy check.
The source video is not copied into datasets or uploaded. Duplicated source
videos cannot silently become independent training/validation/test sessions.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from hybrid_dataset import BACKGROUND, SPLITS, VALID_SESSION, allowed_labels

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "datasets" / "hybrid"
SOURCE_REGISTRY = "_video_sources.json"
MAX_CROP_SIDE = 1024
MIN_CROP_SIDE = 12


def fingerprint_video(path: Path) -> str:
    """Full-file digest prevents a renamed/copied video from faking a new session."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def crop_hash(crop: np.ndarray) -> int:
    """64-bit difference hash used only to reject highly similar crop samples."""
    if not isinstance(crop, np.ndarray) or crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError("Expected a BGR crop")
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    pixels = cv2.resize(grey, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return sum(1 << i for i, bit in enumerate(bits) if bit)


def hash_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def get_registry(root: Path) -> dict:
    path = Path(root) / SOURCE_REGISTRY
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Video source registry is invalid")
    return data


def check_source(root: Path, digest: str, split: str, session: str):
    if split not in SPLITS or not VALID_SESSION.fullmatch(session):
        raise ValueError("Invalid split or session name")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid source video digest")
    registry = get_registry(root)
    existing = registry.get(digest)
    if existing is not None and existing != {"split": split, "session": session}:
        raise ValueError(
            "This video was already assigned to a different split/session. "
            "Copied or renamed recordings are NOT independent data."
        )
    for other_digest, value in registry.items():
        if other_digest != digest and isinstance(value, dict) and value.get("session") == session:
            raise ValueError("Session ID belongs to a different video; choose a new session.")
    return registry


def register_source(root: Path, digest: str, split: str, session: str):
    registry = check_source(root, digest, split, session)
    registry[digest] = {"split": split, "session": session}
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    location = root / SOURCE_REGISTRY
    temp = root / (SOURCE_REGISTRY + ".tmp")
    temp.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(location)


def validate_rect(rect: tuple, frame_shape: tuple):
    if not isinstance(rect, tuple) or len(rect) != 4:
        raise ValueError("Select a rectangular crop first")
    x0, y0, x1, y1 = [int(v) for v in rect]
    height, width = frame_shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Crop is outside the video frame")
    w, h = x1 - x0, y1 - y0
    if min(w, h) < MIN_CROP_SIDE or max(w, h) > MAX_CROP_SIDE:
        raise ValueError("Crop must be between 12 and 1024 pixels per side")
    return x0, y0, x1, y1


def save_labeled_crop(
    root: Path, frame: np.ndarray, rect: tuple, split: str, session: str,
    label: str, source_digest: str, milliseconds: int,
) -> Path:
    """Save one reviewed crop, never a screenshot of the whole source frame."""
    if label not in allowed_labels():
        raise ValueError("Unknown label: " + label)
    if not isinstance(milliseconds, int) or milliseconds < 0:
        raise ValueError("Invalid video timestamp")
    check_source(root, source_digest, split, session)
    x0, y0, x1, y1 = validate_rect(rect, frame.shape)
    crop = frame[y0:y1, x0:x1].copy()
    target = Path(root) / split / session / label
    h = crop_hash(crop)

    if target.is_dir():
        # Existing crops of the same label must not be near duplicates.
        for previous in target.glob("*.png"):
            earlier = cv2.imread(str(previous), cv2.IMREAD_COLOR)
            if earlier is not None and hash_distance(h, crop_hash(earlier)) <= 3:
                raise ValueError("Crop is visually near-duplicate. Seek to a different scene.")
    ok, png = cv2.imencode(".png", crop)
    if not ok:
        raise OSError("Could not encode crop")
    data = png.tobytes()
    suffix = hashlib.sha256(data).hexdigest()[:10]
    path = target / f"{milliseconds:09d}_{suffix}.png"
    if path.exists():
        raise ValueError("This crop is already saved")
    target.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    try:
        register_source(root, source_digest, split, session)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


class VideoCollector:
    def __init__(self, root: tk.Tk, video: Path, dataset: Path):
        self.root = root
        self.video = Path(video)
        self.dataset = Path(dataset)
        self.video_digest = fingerprint_video(self.video)
        self.cap = cv2.VideoCapture(str(self.video))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        if not 0.1 <= self.fps <= 300:
            self.fps = 30.0
        count = float(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.duration = max(0.1, count / self.fps)
        self.frame = None
        self.frame_time = 0.0
        self.photo = None
        self.scale = 1.0
        self.rect = None
        self.drag_origin = None
        self.drag_item = None
        self.image_w = 0
        self.image_h = 0

        self.root.title("The Bums Macro - Private AI Video Collector")
        self.root.geometry("1070x830")
        self.root.minsize(900, 680)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        ttk.Label(
            root, text="PRIVATE / DEVELOPER ONLY - Select a CROPPED target; never a PS code, username or chat.",
            foreground="#b33b37", wraplength=1000,
        ).pack(anchor="w", padx=12, pady=(10, 6))
        ttk.Label(root, text=f"Video: {self.video.name}    SHA256: {self.video_digest[:14]}...").pack(
            anchor="w", padx=12
        )
        controls = ttk.Frame(root)
        controls.pack(fill="x", padx=12, pady=8)
        self.split = tk.StringVar(value="train")
        self.session = tk.StringVar(value="video_" + self.video_digest[:12])
        self.label = tk.StringVar(value=BACKGROUND)
        self.privacy_checked = tk.BooleanVar(value=False)
        ttk.Label(controls, text="Split").grid(row=0, column=0, sticky="w")
        ttk.Combobox(controls, values=SPLITS, textvariable=self.split, width=8, state="readonly").grid(
            row=1, column=0, padx=(0, 12)
        )
        ttk.Label(controls, text="Recording session").grid(row=0, column=1, sticky="w")
        ttk.Entry(controls, textvariable=self.session, width=25).grid(row=1, column=1, padx=(0, 12))
        ttk.Label(controls, text="Verified crop label").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            controls, values=sorted(allowed_labels()), textvariable=self.label,
            width=26, state="readonly",
        ).grid(row=1, column=2, padx=(0, 12))
        ttk.Button(controls, text="Save reviewed crop", command=self.save).grid(row=1, column=3)

        timeline = ttk.Frame(root)
        timeline.pack(fill="x", padx=12, pady=4)
        for title, step in (("−5s", -5), ("−1s", -1), ("+1s", 1), ("+5s", 5)):
            ttk.Button(timeline, text=title, command=lambda s=step: self.jump(s)).pack(side="left", padx=2)
        self.time_slider = tk.Scale(
            timeline, from_=0, to=self.duration, resolution=0.2,
            orient="horizontal", showvalue=True, length=600, command=self.seek,
        )
        self.time_slider.pack(side="left", fill="x", expand=True, padx=8)
        self.canvas = tk.Canvas(root, width=940, height=530, bg="#151515", highlightthickness=0)
        self.canvas.pack(padx=12, pady=6)
        self.canvas.bind("<ButtonPress-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)

        ttk.Checkbutton(
            root, variable=self.privacy_checked,
            text="I have reviewed the selected ROI: correct label and NO private information.",
        ).pack(anchor="w", padx=12, pady=6)
        self.status = tk.StringVar(
            value="Seek to a frame, drag over ONE UI element, choose its exact template label, "
                  "review privacy, then Save. Other game scenes: __background__."
        )
        ttk.Label(root, textvariable=self.status, wraplength=990).pack(anchor="w", padx=12)
        self.load_frame(0)

    def seek(self, value):
        self.load_frame(float(value))

    def jump(self, seconds: float):
        self.time_slider.set(min(self.duration, max(0, self.frame_time + seconds)))

    def load_frame(self, seconds: float):
        seconds = min(self.duration, max(0, seconds))
        self.cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        success, frame = self.cap.read()
        if not success:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(seconds * self.fps) - 1))
            success, frame = self.cap.read()
        if not success:
            self.status.set("Unable to decode frame here; move the slider slightly backward.")
            return
        self.frame = frame
        self.frame_time = seconds
        height, width = frame.shape[:2]
        self.scale = min(940 / width, 530 / height, 1.0)
        self.image_w, self.image_h = int(width * self.scale), int(height * self.scale)
        scaled = cv2.resize(frame, (self.image_w, self.image_h), interpolation=cv2.INTER_AREA)
        ok, image_data = cv2.imencode(".png", scaled)
        if not ok:
            return
        self.photo = tk.PhotoImage(data=base64.b64encode(image_data.tobytes()).decode("ascii"))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.rect = self.drag_origin = self.drag_item = None
        self.privacy_checked.set(False)

    def mouse_down(self, event):
        if self.frame is None:
            return
        x = max(0, min(event.x, self.image_w))
        y = max(0, min(event.y, self.image_h))
        self.drag_origin = (x, y)
        self.rect = None
        self.privacy_checked.set(False)
        if self.drag_item is not None:
            self.canvas.delete(self.drag_item)
        self.drag_item = self.canvas.create_rectangle(x, y, x, y, outline="#26b8e7", width=2)

    def mouse_drag(self, event):
        if self.drag_origin is not None and self.drag_item is not None:
            x = max(0, min(event.x, self.image_w))
            y = max(0, min(event.y, self.image_h))
            self.canvas.coords(self.drag_item, *self.drag_origin, x, y)

    def mouse_up(self, event):
        if self.drag_origin is None or self.frame is None:
            return
        x = max(0, min(event.x, self.image_w))
        y = max(0, min(event.y, self.image_h))
        left, right = sorted((self.drag_origin[0], x))
        top, bottom = sorted((self.drag_origin[1], y))
        factor = 1.0 / self.scale
        rect = (round(left * factor), round(top * factor),
                round(right * factor), round(bottom * factor))
        try:
            validate_rect(rect, self.frame.shape)
            self.rect = rect
            self.status.set(
                f"Selected ROI: {rect}. Set the true label and review privacy before saving."
            )
        except ValueError as exc:
            self.rect = None
            self.status.set(str(exc))
        self.drag_origin = None

    def save(self):
        try:
            if self.frame is None or self.rect is None:
                raise ValueError("Select a small crop on the current frame first")
            if not self.privacy_checked.get():
                raise ValueError("Confirm the selected crop contains no private information")
            session = self.session.get().strip()
            path = save_labeled_crop(
                self.dataset, self.frame, self.rect, self.split.get(), session,
                self.label.get(), self.video_digest, round(self.frame_time * 1000),
            )
            self.status.set(f"Saved private labeled crop: {path.relative_to(self.dataset)}")
            self.privacy_checked.set(False)
        except Exception as exc:
            messagebox.showwarning("Crop not saved", str(exc), parent=self.root)

    def close(self):
        self.cap.release()
        self.root.destroy()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="Local video, selected with a dialog if omitted")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    root = tk.Tk()
    root.withdraw()
    video = args.video or filedialog.askopenfilename(
        parent=root, title="Select your private AOT:R recording",
        filetypes=[("Gameplay videos", "*.mp4 *.mov *.mkv *.avi *.webm"), ("All files", "*.*")],
    )
    if not video:
        root.destroy()
        return
    try:
        collector = VideoCollector(root, Path(video), args.root)
    except Exception as exc:
        messagebox.showerror("Cannot open video", str(exc), parent=root)
        root.destroy()
        return
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()
