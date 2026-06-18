#!/usr/bin/env python3
"""
ingest_tracks.py — Load a tracker CSV into the tracking PostgreSQL database.

CSV columns expected:
    frame_id, timestamp_s, track_id, class_id, class_name, state,
    x1, y1, x2, y2, width, height, cx, cy

Usage:
    python ingest.py tracks.csv \
        --camera "cam_01" \
        --segment 1 \
        [--host localhost] [--port 5432] [--dbname tracking] [--user sipg] [--password sipg] \
        [--fps 6.47] [--location "entrance"] [--resolution "1920x1080"] \
        [--file-path "/data/cam01_seg1.mp4"] \
        [--batch-size 2000] \
        [--dry-run]

What it builds
--------------
  dim_camera     – upsert by camera_name
  dim_video      – upsert by (camera_id, segment_number)
  dim_class      – upsert by class_name (handles tracker labels not in seed)
  dim_time       – upsert one row per unique timestamp_s
  dim_frame      – upsert one row per (video_id, frame_number)
  dim_track      – one row per real-world track_id found in the CSV
  fact_tracklet  – one row per contiguous run of detections per track
                   (gaps > 1 frame → new tracklet)
  fact_detection – one row per CSV row

Speed columns (speed_px_per_s, avg_speed_px_per_s) are left NULL intentionally
and can be computed in a follow-up pass.
"""

import argparse
import csv
import math
import sys
import time
from collections import defaultdict

import psycopg2
import psycopg2.extras


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Ingest tracker CSV into PostgreSQL tracking DB")
    p.add_argument("csv_file", help="Path to the tracker CSV file")
    p.add_argument("--camera",      required=True, help="Camera name (e.g. 'cam_01')")
    p.add_argument("--segment",     required=True, type=int, help="Video segment number (e.g. 1)")
    p.add_argument("--host",        default="localhost")
    p.add_argument("--port",        default=5432, type=int)
    p.add_argument("--dbname",      default="tracking")
    p.add_argument("--user",        default="sipg")
    p.add_argument("--password",    default="sipg")
    p.add_argument("--fps",         default=None, type=float, help="Camera FPS (optional)")
    p.add_argument("--location",    default=None, help="Camera location label (optional)")
    p.add_argument("--resolution",  default=None, help="Camera resolution e.g. '1920x1080' (optional)")
    p.add_argument("--file-path",   default=None, help="Video file path stored in dim_video (optional)")
    p.add_argument("--batch-size",  default=2000, type=int, help="INSERT batch size (default 2000)")
    p.add_argument("--dry-run",     action="store_true", help="Parse and plan without writing to DB")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Class name normalisation
# Map tracker-specific labels -> canonical schema names
# ---------------------------------------------------------------------------

CLASS_NAME_MAP = {
    "moto": "motorbike",
}

def normalise_class(name):
    return CLASS_NAME_MAP.get(name.strip().lower(), name.strip().lower())


# ---------------------------------------------------------------------------
# CSV loading & tracklet segmentation
# ---------------------------------------------------------------------------

def load_csv(path):
    """Return list of dicts; coerce numeric fields."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "frame_number": int(r["frame_id"]),
                "timestamp_s":  float(r["timestamp_s"]),
                "src_track_id": int(r["track_id"]),
                "class_name":   normalise_class(r["class_name"]),
                "state":        r["state"].strip(),
                "x1":    float(r["x1"]),
                "y1":    float(r["y1"]),
                "x2":    float(r["x2"]),
                "y2":    float(r["y2"]),
                "width": float(r["width"]),
                "height":float(r["height"]),
                "cx":    float(r["cx"]),
                "cy":    float(r["cy"]),
            })
    return rows


def segment_tracklets(rows):
    """
    Group detections by src_track_id, sort by frame_number, then split into
    contiguous runs whenever there is a gap > 1 frame.

    Returns:
        tracklets: list of dicts, each with keys:
            src_track_id, class_name, confidence_state,
            start_ts, end_ts, detection_count,
            detections: [row, ...]
    """
    by_track = defaultdict(list)
    for r in rows:
        by_track[r["src_track_id"]].append(r)

    tracklets = []
    for tid, dets in by_track.items():
        dets.sort(key=lambda d: d["frame_number"])

        # Each contiguous run of frames (gap <= 1) is one tracklet.
        # We also treat a class change within a track as a new tracklet,
        # since class is assigned at tracklet level.
        run = [dets[0]]
        for prev, cur in zip(dets, dets[1:]):
            gap = cur["frame_number"] - prev["frame_number"]
            class_changed = cur["class_name"] != prev["class_name"]
            if gap > 1 or class_changed:
                tracklets.append(_build_tracklet(tid, run))
                run = [cur]
            else:
                run.append(cur)
        tracklets.append(_build_tracklet(tid, run))

    return tracklets


def _build_tracklet(src_track_id, dets):
    # Pick the most common confidence_state across the run.
    states = [d["state"] for d in dets]
    confidence_state = max(set(states), key=states.count)
    return {
        "src_track_id":    src_track_id,
        "class_name":      dets[0]["class_name"],
        "confidence_state": confidence_state,
        "start_ts":        dets[0]["timestamp_s"],
        "end_ts":          dets[-1]["timestamp_s"],
        "detection_count": len(dets),
        "detections":      dets,
    }


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def upsert_camera(cur, camera_name, location, fps, resolution):
    cur.execute("""
        INSERT INTO dim_camera (camera_name, location, fps, resolution)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (camera_name) DO UPDATE
            SET location   = EXCLUDED.location,
                fps        = EXCLUDED.fps,
                resolution = EXCLUDED.resolution
        RETURNING camera_id
    """, (camera_name, location, fps, resolution))
    return cur.fetchone()[0]


def upsert_video(cur, camera_id, segment_number, start_ts, end_ts, file_path):
    cur.execute("""
        INSERT INTO dim_video (camera_id, segment_number, start_ts, end_ts, file_path)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (camera_id, segment_number) DO UPDATE
            SET start_ts  = EXCLUDED.start_ts,
                end_ts    = EXCLUDED.end_ts,
                file_path = EXCLUDED.file_path
        RETURNING video_id
    """, (camera_id, segment_number, start_ts, end_ts, file_path))
    return cur.fetchone()[0]


def upsert_classes(cur, class_names):
    """Upsert all class names and return {class_name: class_id}."""
    psycopg2.extras.execute_values(cur, """
        INSERT INTO dim_class (class_name, class_category)
        VALUES %s
        ON CONFLICT (class_name) DO NOTHING
    """, [(name, _infer_category(name)) for name in class_names])
    cur.execute("SELECT class_name, class_id FROM dim_class WHERE class_name = ANY(%s)",
                (list(class_names),))
    return {row[0]: row[1] for row in cur.fetchall()}


def _infer_category(name):
    vehicles = {"car", "truck", "bus", "bicycle", "motorbike", "moto"}
    return "vehicle" if name in vehicles else "person" if name == "person" else "unknown"


def upsert_times(cur, timestamps):
    """Upsert all unique timestamps and return {timestamp_s: time_id}."""
    def ts_parts(ts):
        total_s = float(ts)
        h  = int(total_s // 3600)
        m  = int((total_s % 3600) // 60)
        s  = int(total_s % 60)
        ms = int(round((total_s - int(total_s)) * 1000))
        return (total_s, h, m, s, ms)

    data = [ts_parts(ts) for ts in sorted(timestamps)]
    psycopg2.extras.execute_values(cur, """
        INSERT INTO dim_time (timestamp_s, hour, minute, second, millisecond)
        VALUES %s
        ON CONFLICT (timestamp_s) DO NOTHING
    """, data)
    cur.execute("SELECT timestamp_s, time_id FROM dim_time WHERE timestamp_s = ANY(%s)",
                ([float(ts) for ts in timestamps],))
    return {float(row[0]): row[1] for row in cur.fetchall()}


def upsert_frames(cur, video_id, frame_rows, batch_size):
    """
    Upsert dim_frame rows.  frame_rows is a list of (frame_number, timestamp_s).
    Returns {frame_number: frame_id}.
    """
    unique = {fn: ts for fn, ts in frame_rows}  # last ts wins for a given frame_number
    data = [(video_id, fn, ts) for fn, ts in sorted(unique.items())]

    for i in range(0, len(data), batch_size):
        psycopg2.extras.execute_values(cur, """
            INSERT INTO dim_frame (video_id, frame_number, timestamp_s)
            VALUES %s
            ON CONFLICT (video_id, frame_number) DO NOTHING
        """, data[i:i+batch_size])

    cur.execute("SELECT frame_number, frame_id FROM dim_frame WHERE video_id = %s", (video_id,))
    return {row[0]: row[1] for row in cur.fetchall()}


def insert_tracks(cur, src_track_ids, camera_id, tracklets):
    """
    Insert one dim_track row per unique src_track_id.
    Derive first_seen_ts, last_seen_ts, tracklet_count from the tracklets list.
    Returns {src_track_id: track_id}.
    """
    # Aggregate per src_track_id
    agg = defaultdict(lambda: {"first": math.inf, "last": -math.inf, "count": 0})
    for tl in tracklets:
        tid = tl["src_track_id"]
        agg[tid]["first"] = min(agg[tid]["first"], tl["start_ts"])
        agg[tid]["last"]  = max(agg[tid]["last"],  tl["end_ts"])
        agg[tid]["count"] += 1

    mapping = {}
    for src_id in src_track_ids:
        a = agg[src_id]
        cur.execute("""
            INSERT INTO dim_track (camera_id, first_seen_ts, last_seen_ts, tracklet_count)
            VALUES (%s, %s, %s, %s)
            RETURNING track_id
        """, (camera_id, a["first"], a["last"], a["count"]))
        mapping[src_id] = cur.fetchone()[0]
    return mapping


def insert_tracklets(cur, tracklets, track_map, camera_id, video_id,
                     class_map, time_map, batch_size):
    """
    Insert fact_tracklet rows in batches.
    Returns list of (tracklet_index, tracklet_id) pairs in order.
    """
    data = []
    for tl in tracklets:
        data.append((
            track_map[tl["src_track_id"]],
            camera_id,
            video_id,
            class_map[tl["class_name"]],
            time_map[tl["start_ts"]],
            time_map[tl["end_ts"]],
            tl["src_track_id"],
            tl["confidence_state"],
            tl["detection_count"],
            # avg_speed_px_per_s → NULL (computed later)
            # embedding         → NULL (not in CSV)
        ))

    tracklet_ids = []
    for i in range(0, len(data), batch_size):
        psycopg2.extras.execute_values(cur, """
            INSERT INTO fact_tracklet
                (track_id, camera_id, video_id, class_id,
                 start_time_id, end_time_id,
                 source_track_id, confidence_state, detection_count)
            VALUES %s
            RETURNING tracklet_id
        """, data[i:i+batch_size], fetch=True)
        tracklet_ids.extend(row[0] for row in cur.fetchall())

    return tracklet_ids


def insert_detections(cur, tracklets, tracklet_ids, frame_map, batch_size):
    """Insert fact_detection rows in batches."""
    data = []
    for tl, tkl_id in zip(tracklets, tracklet_ids):
        for det in tl["detections"]:
            data.append((
                tkl_id,
                frame_map[det["frame_number"]],
                det["timestamp_s"],
                det["x1"], det["y1"], det["x2"], det["y2"],
                det["width"], det["height"],
                det["cx"],   det["cy"],
                # speed_px_per_s → NULL (computed later)
            ))

    for i in range(0, len(data), batch_size):
        psycopg2.extras.execute_values(cur, """
            INSERT INTO fact_detection
                (tracklet_id, frame_id, timestamp_s,
                 x1, y1, x2, y2, width, height, cx, cy)
            VALUES %s
        """, data[i:i+batch_size])

    return len(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── 1. Load & parse CSV ────────────────────────────────────────────────
    t0 = time.perf_counter()
    print(f"Reading {args.csv_file} …")
    rows = load_csv(args.csv_file)
    print(f"  {len(rows):,} detections loaded in {time.perf_counter()-t0:.1f}s")

    # ── 2. Segment tracklets ───────────────────────────────────────────────
    print("Segmenting tracklets …")
    tracklets = segment_tracklets(rows)
    src_track_ids = sorted({tl["src_track_id"] for tl in tracklets})
    class_names   = sorted({tl["class_name"]   for tl in tracklets})
    all_timestamps = sorted({d["timestamp_s"] for tl in tracklets for d in tl["detections"]})
    all_frames     = [(d["frame_number"], d["timestamp_s"])
                      for tl in tracklets for d in tl["detections"]]

    print(f"  {len(src_track_ids)} tracks → {len(tracklets)} tracklets")
    print(f"  {len(set(fn for fn, _ in all_frames))} unique frames, "
          f"{len(all_timestamps)} unique timestamps")
    print(f"  Classes: {class_names}")

    # ── 3. Infer video timestamps ─────────────────────────────────────────
    global_start_ts = min(all_timestamps)
    global_end_ts   = max(all_timestamps)

    if args.dry_run:
        print("\n[dry-run] No changes written to the database.")
        return

    # ── 4. Connect ────────────────────────────────────────────────────────
    print(f"\nConnecting to {args.user}@{args.host}:{args.port}/{args.dbname} …")
    conn = psycopg2.connect(
        host=args.host, port=args.port,
        dbname=args.dbname, user=args.user, password=args.password,
    )
    conn.autocommit = False

    try:
        cur = conn.cursor()

        # ── 5. Dimensions ─────────────────────────────────────────────────
        print("Upserting dim_camera …")
        camera_id = upsert_camera(cur, args.camera, args.location, args.fps, args.resolution)
        print(f"  camera_id = {camera_id}")

        print("Upserting dim_video …")
        video_id = upsert_video(cur, camera_id, args.segment,
                                global_start_ts, global_end_ts, args.file_path)
        print(f"  video_id = {video_id}")

        print("Upserting dim_class …")
        class_map = upsert_classes(cur, class_names)
        print(f"  {class_map}")

        print(f"Upserting {len(all_timestamps):,} dim_time rows …")
        time_map = upsert_times(cur, all_timestamps)

        print(f"Upserting {len(set(fn for fn,_ in all_frames)):,} dim_frame rows …")
        frame_map = upsert_frames(cur, video_id, all_frames, args.batch_size)

        # ── 6. dim_track ──────────────────────────────────────────────────
        print(f"Inserting {len(src_track_ids)} dim_track rows …")
        track_map = insert_tracks(cur, src_track_ids, camera_id, tracklets)

        # ── 7. fact_tracklet ──────────────────────────────────────────────
        print(f"Inserting {len(tracklets):,} fact_tracklet rows …")
        t1 = time.perf_counter()
        tracklet_ids = insert_tracklets(
            cur, tracklets, track_map, camera_id, video_id,
            class_map, time_map, args.batch_size,
        )
        print(f"  done in {time.perf_counter()-t1:.1f}s")

        # ── 8. fact_detection ─────────────────────────────────────────────
        print(f"Inserting {len(rows):,} fact_detection rows …")
        t1 = time.perf_counter()
        n = insert_detections(cur, tracklets, tracklet_ids, frame_map, args.batch_size)
        print(f"  {n:,} rows inserted in {time.perf_counter()-t1:.1f}s")

        # ── 9. Commit ─────────────────────────────────────────────────────
        conn.commit()
        elapsed = time.perf_counter() - t0
        print(f"\n✓ Done in {elapsed:.1f}s")
        print(f"  camera_id={camera_id}  video_id={video_id}")
        print(f"  tracks={len(src_track_ids)}  tracklets={len(tracklets)}  detections={n:,}")

    except Exception as e:
        conn.rollback()
        print(f"\n✗ Error — rolled back: {e}", file=sys.stderr)
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
