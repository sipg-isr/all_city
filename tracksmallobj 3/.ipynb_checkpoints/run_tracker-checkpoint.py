"""
Visualization + Demo Runner — Small Object Edition
====================================================
Run:
  python run_tracker.py --source video.mp4
  python run_tracker.py --source 0 --no-sahi         (webcam, no tiling)
  python run_tracker.py --source video.mp4 --output out.mp4
"""

import argparse
import cv2
import numpy as np
import time
from tracker import LowFPSTracker, TrackerConfig

CLASS_NAMES  = {0: "person", 2: "car", 3: "moto", 5: "bus", 7: "truck"}
CLASS_COLORS = {
    0: (100, 220, 100),
    2: ( 80, 160, 255),
    3: (255, 180,  60),
    5: (200,  80, 255),
    7: ( 60, 220, 220),
}


def draw_tracks(frame: np.ndarray, tracks: list, frame_id: int, dt_ms: float) -> np.ndarray:
    vis = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = map(int, t["bbox"])
        color     = CLASS_COLORS.get(t["class_id"], (200, 200, 200))
        label     = f'{CLASS_NAMES.get(t["class_id"], "obj")} #{t["id"]}'
        thickness = 2 if t["state"] == "confirmed" else 1

        # Scale dot marker for very small boxes
        box_h = max(y2 - y1, 1)
        dot_r = max(2, min(5, box_h // 8))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(vis, ((x1+x2)//2, (y1+y2)//2), dot_r, color, -1)

        # Label background
        font_scale = max(0.35, min(0.55, box_h / 80))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        lx, ly = x1, max(y1 - 2, th + 4)
        cv2.rectangle(vis, (lx, ly - th - 4), (lx + tw + 4, ly), color, -1)
        cv2.putText(vis, label, (lx + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    # HUD
    hud = f"Frame {frame_id} | Tracks: {len(tracks)} | {dt_ms:.0f}ms"
    cv2.putText(vis, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",     default="0")
    ap.add_argument("--fps-sim",    type=int,   default=1,
                    help="Process every N-th frame to simulate low FPS")
    ap.add_argument("--conf",       type=float, default=0.25)
    ap.add_argument("--max-age",    type=int,   default=30)
    ap.add_argument("--no-sahi",    action="store_true",
                    help="Disable SAHI tiling (faster but worse small-obj recall)")
    ap.add_argument("--slice-size", type=int,   default=640,
                    help="SAHI tile size in pixels")
    ap.add_argument("--model",      default="yolov8m.pt")
    ap.add_argument("--output",     default="")
    args = ap.parse_args()

    cfg = TrackerConfig(
        yolo_model=args.model,
        det_conf_thresh=args.conf,
        max_age=args.max_age,
        use_sahi=not args.no_sahi,
        sahi_slice_size=args.slice_size,
    )
    tracker = LowFPSTracker(cfg)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Demo] {W}x{H} @ {orig_fps:.1f}fps  |  processing every {args.fps_sim} frame(s)")

    writer = None
    if args.output:
        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"mp4v"),
            max(1, orig_fps // args.fps_sim), (W, H)
        )

    fidx, last_vis = 0, None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fidx += 1

        if fidx % args.fps_sim != 0:
            #if last_vis is not None:
             #   cv2.imshow("Tracker", last_vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        t0 = time.perf_counter()
        tracks = tracker.update(frame)
        dt = (time.perf_counter() - t0) * 1000

        last_vis = draw_tracks(frame, tracks, fidx, dt)
        if writer:
            writer.write(last_vis)
        #cv2.imshow("Tracker", last_vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
