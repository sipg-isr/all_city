"""
Visualization + Demo Runner — Small Object Edition
====================================================
Usage examples:

  # Interactive display (default)
  python run_tracker.py --source video.mp4

  # Headless / silent — no window, CSV only, progress every 100 frames
  python run_tracker.py --source video.mp4 --silent --csv tracks.csv

  # Silent + save annotated video for later review
  python run_tracker.py --source video.mp4 --silent --csv tracks.csv --output out.mp4

  # Webcam, no SAHI, display only
  python run_tracker.py --source 0 --no-sahi

CSV columns:
  frame_id, timestamp_s, track_id, class_id, class_name,
  state, x1, y1, x2, y2, width, height, cx, cy
"""

import argparse
import csv
import datetime
import os
import sys
import time

import cv2
import numpy as np

from tracker import LowFPSTracker, TrackerConfig

# ── Label maps ────────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "person", 2: "car", 3: "moto", 5: "bus", 7: "truck"}
CLASS_COLORS = {
    0: (100, 220, 100),
    2: ( 80, 160, 255),
    3: (255, 180,  60),
    5: (200,  80, 255),
    7: ( 60, 220, 220),
}

# ─────────────────────────────────────────────
# CSV Result Writer
# ─────────────────────────────────────────────

CSV_COLUMNS = [
    "frame_id",    # integer index of the raw input frame
    "timestamp_s", # wall-clock seconds since run start (float, 4 dp)
    "track_id",    # unique track ID (persists across frames)
    "class_id",    # COCO class integer
    "class_name",  # human-readable label
    "state",       # "tentative" | "confirmed"
    "x1", "y1",    # top-left corner (pixels)
    "x2", "y2",    # bottom-right corner (pixels)
    "width",       # box width  (x2 - x1)
    "height",      # box height (y2 - y1)
    "cx", "cy",    # box centre (pixels)
]


class ResultWriter:
    """
    Writes one CSV row per active track per processed frame.
    Flushes after every frame so data survives a killed process.

    Use as a context manager:
        with ResultWriter("tracks.csv") as rw:
            rw.write(frame_id, timestamp_s, tracks)
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._file = None
        self._writer = None

    def open(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
        self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        print(f"[CSV] Writing to {self.csv_path}")
        return self

    def close(self):
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
            print(f"[CSV] Saved → {self.csv_path}")

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()

    def write(self, frame_id: int, timestamp_s: float, tracks: list):
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            self._writer.writerow({
                "frame_id":    frame_id,
                "timestamp_s": round(timestamp_s, 4),
                "track_id":    t["id"],
                "class_id":    t["class_id"],
                "class_name":  CLASS_NAMES.get(t["class_id"], "unknown"),
                "state":       t["state"],
                "x1":          round(x1, 2),
                "y1":          round(y1, 2),
                "x2":          round(x2, 2),
                "y2":          round(y2, 2),
                "width":       round(x2 - x1, 2),
                "height":      round(y2 - y1, 2),
                "cx":          round((x1 + x2) / 2, 2),
                "cy":          round((y1 + y2) / 2, 2),
            })
        self._file.flush()


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def draw_tracks(frame: np.ndarray, tracks: list,
                frame_id: int, dt_ms: float) -> np.ndarray:
    vis = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = map(int, t["bbox"])
        color     = CLASS_COLORS.get(t["class_id"], (200, 200, 200))
        label     = f'{CLASS_NAMES.get(t["class_id"], "obj")} #{t["id"]}'
        thickness = 2 if t["state"] == "confirmed" else 1

        box_h    = max(y2 - y1, 1)
        dot_r    = max(2, min(5, box_h // 8))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(vis, ((x1 + x2) // 2, (y1 + y2) // 2), dot_r, color, -1)

        font_scale = max(0.35, min(0.55, box_h / 80))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        lx, ly = x1, max(y1 - 2, th + 4)
        cv2.rectangle(vis, (lx, ly - th - 4), (lx + tw + 4, ly), color, -1)
        cv2.putText(vis, label, (lx + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    hud = f"Frame {frame_id} | Tracks: {len(tracks)} | {dt_ms:.0f}ms"
    cv2.putText(vis, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return vis


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Low-FPS multi-object tracker for people and vehicles."
    )
    ap.add_argument("--source",     default="0",
                    help="Video file path or webcam index (default: 0)")
    ap.add_argument("--fps-sim",    type=int, default=1,
                    help="Process every N-th frame to simulate low FPS (default: 1)")
    ap.add_argument("--conf",       type=float, default=0.25,
                    help="Detection confidence threshold (default: 0.25)")
    ap.add_argument("--max-age",    type=int, default=30,
                    help="Frames to keep a lost track alive (default: 30)")
    ap.add_argument("--no-sahi",    action="store_true",
                    help="Disable SAHI tiling (faster but lower small-object recall)")
    ap.add_argument("--slice-size", type=int, default=640,
                    help="SAHI tile size in pixels (default: 640)")
    ap.add_argument("--model",      default="yolov8m.pt",
                    help="YOLOv8 weights file (default: yolov8m.pt)")
    ap.add_argument("--output",     default="",
                    help="Annotated video output path (optional)")
    ap.add_argument("--csv",        default=None,
                    help="CSV output path (e.g. tracks.csv). "
                         "Pass flag with no value for auto-timestamped filename.")
    ap.add_argument("--silent",     action="store_true",
                    help=(
                        "Headless / silent mode: no OpenCV display window, "
                        "no per-frame console output. Progress is printed every "
                        "100 processed frames. Use for servers, SSH, batch jobs. "
                        "--output still works in silent mode (renders to file only)."
                    ))
    args = ap.parse_args()

    # Auto-name CSV when --csv flag is present but no value given
    if "--csv" in sys.argv and args.csv is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.csv = f"tracks_{ts}.csv"

    # Convenience: silence decorator
    def log(*msg):
        if not args.silent:
            print(*msg)

    # ── Build tracker ──────────────────────────────────────────────────────────
    cfg = TrackerConfig(
        yolo_model=args.model,
        det_conf_thresh=args.conf,
        max_age=args.max_age,
        use_sahi=not args.no_sahi,
        sahi_slice_size=args.slice_size,
    )
    tracker = LowFPSTracker(cfg)

    # ── Open source ────────────────────────────────────────────────────────────
    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    orig_fps     = cap.get(cv2.CAP_PROP_FPS) or 30
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 0 for live streams

    print(
        f"[Tracker] source={source}  {W}x{H} @ {orig_fps:.1f}fps  "
        f"silent={'yes' if args.silent else 'no'}  "
        f"sahi={'on' if cfg.use_sahi else 'off'}  "
        f"csv={args.csv or 'off'}  video_out={args.output or 'off'}"
    )

    # ── Video writer ───────────────────────────────────────────────────────────
    vid_writer = None
    if args.output:
        vid_writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1, orig_fps // args.fps_sim),
            (W, H),
        )
        log(f"[Tracker] Writing annotated video → {args.output}")

    # ── CSV writer ─────────────────────────────────────────────────────────────
    csv_writer = ResultWriter(args.csv) if args.csv else None
    if csv_writer:
        csv_writer.open()

    # ── Main loop ──────────────────────────────────────────────────────────────
    t_start    = time.perf_counter()
    fidx       = 0    # raw frame index (including skipped)
    proc_count = 0    # processed (non-skipped) frames
    last_vis   = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fidx += 1

            # Skip frames to simulate low FPS
            if fidx % args.fps_sim != 0:
                if not args.silent and last_vis is not None:
                    cv2.imshow("Tracker", last_vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            # Run tracker
            t0          = time.perf_counter()
            tracks      = tracker.update(frame)
            dt_ms       = (time.perf_counter() - t0) * 1000
            timestamp_s = time.perf_counter() - t_start
            proc_count += 1

            # Save CSV
            if csv_writer:
                csv_writer.write(fidx, timestamp_s, tracks)

            # Display / video output
            if not args.silent:
                last_vis = draw_tracks(frame, tracks, fidx, dt_ms)
                cv2.imshow("Tracker", last_vis)
                if vid_writer:
                    vid_writer.write(last_vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif vid_writer:
                # Silent but video output requested — render without showing
                last_vis = draw_tracks(frame, tracks, fidx, dt_ms)
                vid_writer.write(last_vis)

            # Silent progress tick every 100 processed frames
            if args.silent and proc_count % 100 == 0:
                elapsed = timestamp_s
                eta_str = ""
                if total_frames > 0 and fidx > 0:
                    ratio   = fidx / total_frames
                    eta_s   = (elapsed / ratio) * (1.0 - ratio)
                    eta_str = f"  ETA {eta_s:.0f}s"
                print(
                    f"[Tracker] frame {fidx}"
                    f"{'/' + str(total_frames) if total_frames else ''}  "
                    f"proc={proc_count}  tracks={len(tracks)}  "
                    f"{elapsed:.1f}s elapsed{eta_str}"
                )

    finally:
        cap.release()
        if vid_writer:
            vid_writer.release()
        if csv_writer:
            csv_writer.close()
        if not args.silent:
            cv2.destroyAllWindows()

        elapsed = time.perf_counter() - t_start
        print(
            f"[Tracker] Done — {proc_count} frames processed in "
            f"{elapsed:.1f}s ({proc_count / max(elapsed, 0.001):.1f} fps throughput)"
        )


if __name__ == "__main__":
    main()
