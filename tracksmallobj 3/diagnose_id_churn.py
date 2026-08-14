"""
diagnose_id_churn.py — estimate how often the tracker is fragmenting one
physical object into multiple track_ids, using the CSV you already have.

This is a heuristic, not ground truth: for each track that *ends*, it
estimates that object's velocity from its last few frames, projects forward
in a straight line to where it *should* be at various later frame_ids, and
looks for a track that *starts* near that projected position. This matters
because this tracker targets large interframe displacement / low FPS —
during a real occlusion gap a moving object can travel far from where it was
last seen, so comparing against its last known (static) position badly
undercounts churn. A pair found this way is flagged as a "candidate split" —
probably the same object, re-spawned as a new ID.

Won't catch everything (e.g. an object that changes direction/speed during
the gap — projection assumes constant velocity) and can occasionally
misfire (e.g. two different objects converging), but gives you a real
number instead of a guess, and lets you compare before/after when you tune
max_age / appearance_weight / max_cost.

Usage:
    python diagnose_id_churn.py output.csv
    python diagnose_id_churn.py output.csv --max-gap-frames 90 --max-dist-boxwidths 2.0
    python diagnose_id_churn.py output.csv --require-same-class
    python diagnose_id_churn.py output.csv --no-velocity   # old static-position behavior
"""

import argparse
import pandas as pd
import numpy as np


def load_tracks(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"frame_id", "track_id", "class_id", "cx", "cy", "width", "height"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")
    return df


def _estimate_velocity(track_df: pd.DataFrame, n_tail: int = 10):
    """(vx, vy) in pixels/frame, fit over a track's last n_tail rows via
    linear regression against frame_id. Falls back to (0, 0) — i.e. the old
    static-position assumption — if there's not enough history to fit."""
    tail = track_df.sort_values("frame_id").tail(n_tail)
    if len(tail) < 2:
        return 0.0, 0.0
    t = tail["frame_id"].values.astype(float)
    t = t - t[0]
    if t[-1] == 0:
        return 0.0, 0.0
    vx = float(np.polyfit(t, tail["cx"].values, 1)[0])
    vy = float(np.polyfit(t, tail["cy"].values, 1)[0])
    return vx, vy


def summarize_tracks(df: pd.DataFrame, n_tail: int = 10) -> pd.DataFrame:
    """One row per track_id: when/where it starts and ends, its typical class
    and size, and its estimated velocity near the end (used to project where
    it should be later). Size normalises 'nearby' — a truck can be many
    pixels from another truck and still be 'close', a distant person can't."""
    g = df.groupby("track_id")
    summary = g.agg(
        start_frame=("frame_id", "min"),
        end_frame=("frame_id", "max"),
        n_hits=("frame_id", "count"),
        class_id=("class_id", lambda s: s.mode().iat[0]),
        avg_w=("width", "mean"),
        avg_h=("height", "mean"),
    ).reset_index()

    starts = df.sort_values("frame_id").groupby("track_id").first()[["cx", "cy"]]
    ends = df.sort_values("frame_id").groupby("track_id").last()[["cx", "cy"]]
    summary = summary.merge(starts.rename(columns={"cx": "start_cx", "cy": "start_cy"}),
                             on="track_id")
    summary = summary.merge(ends.rename(columns={"cx": "end_cx", "cy": "end_cy"}),
                             on="track_id")

    _vel_fn = lambda g: pd.Series(_estimate_velocity(g, n_tail), index=["vx", "vy"])
    try:
        # pandas >= 2.2: avoids a FutureWarning about grouping columns
        vel = df.groupby("track_id").apply(_vel_fn, include_groups=False).reset_index()
    except TypeError:
        # older pandas: no include_groups kwarg, but works the same either way
        vel = df.groupby("track_id").apply(_vel_fn).reset_index()
    summary = summary.merge(vel, on="track_id")
    return summary


def find_candidate_splits(summary: pd.DataFrame, max_gap_frames: int = 90,
                           max_dist_boxwidths: float = 2.0,
                           require_same_class: bool = False,
                           use_velocity: bool = True) -> pd.DataFrame:
    """Greedy nearest-match: for each track's end, project its position
    forward using its own velocity, and find the closest track start that
    begins shortly after and near that projection. One candidate link per
    ending track — a heuristic, not a solver."""
    candidates = []
    for _, end_row in summary.sort_values("end_frame").iterrows():
        box_scale = max((end_row["avg_w"] + end_row["avg_h"]) / 2.0, 1.0)
        pool = summary[
            (summary["track_id"] != end_row["track_id"]) &
            (summary["start_frame"] > end_row["end_frame"]) &
            (summary["start_frame"] <= end_row["end_frame"] + max_gap_frames)
        ]
        if require_same_class:
            pool = pool[pool["class_id"] == end_row["class_id"]]
        if pool.empty:
            continue

        dt = pool["start_frame"] - end_row["end_frame"]
        if use_velocity:
            proj_x = end_row["end_cx"] + end_row["vx"] * dt
            proj_y = end_row["end_cy"] + end_row["vy"] * dt
        else:
            proj_x = end_row["end_cx"]
            proj_y = end_row["end_cy"]

        dist = np.hypot(pool["start_cx"] - proj_x, pool["start_cy"] - proj_y)
        dist_norm = dist / box_scale
        best_idx = dist_norm.idxmin()
        best_dist_norm = dist_norm.loc[best_idx]

        if best_dist_norm <= max_dist_boxwidths:
            best = pool.loc[best_idx]
            candidates.append({
                "end_track_id":   end_row["track_id"],
                "end_frame":      int(end_row["end_frame"]),
                "end_class":      end_row["class_id"],
                "start_track_id": int(best["track_id"]),
                "start_frame":    int(best["start_frame"]),
                "start_class":    best["class_id"],
                "gap_frames":     int(best["start_frame"] - end_row["end_frame"]),
                "dist_boxwidths": round(float(best_dist_norm), 2),
                "same_class":     bool(end_row["class_id"] == best["class_id"]),
            })
    return pd.DataFrame(candidates)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path")
    ap.add_argument("--max-gap-frames", type=int, default=90,
                     help="Max frames between one track ending and another starting "
                          "to count as a candidate split (default: 90)")
    ap.add_argument("--max-dist-boxwidths", type=float, default=2.0,
                     help="Max distance from the PROJECTED position (see --no-velocity), "
                          "in units of the ending box's average width/height, to count "
                          "as 'same place' (default: 2.0)")
    ap.add_argument("--require-same-class", action="store_true",
                     help="Only flag candidates where both tracks share the same class_id")
    ap.add_argument("--no-velocity", action="store_true",
                     help="Compare against the track's last known (static) position "
                          "instead of projecting motion forward. Only useful for "
                          "near-stationary objects/cameras — for moving objects this "
                          "will undercount churn, sometimes to zero.")
    ap.add_argument("--velocity-tail", type=int, default=10,
                     help="Number of trailing frames used to estimate each track's "
                          "velocity (default: 10)")
    args = ap.parse_args()

    df = load_tracks(args.csv_path)
    summary = summarize_tracks(df, n_tail=args.velocity_tail)
    candidates = find_candidate_splits(
        summary, args.max_gap_frames, args.max_dist_boxwidths,
        args.require_same_class, use_velocity=not args.no_velocity,
    )

    n_tracks = len(summary)
    n_candidates = len(candidates)
    n_class_mismatch = int((~candidates["same_class"]).sum()) if n_candidates else 0

    speeds = np.hypot(summary["vx"], summary["vy"])
    print(f"Tracks in CSV:                  {n_tracks}")
    print(f"Median track speed near its end: {speeds.median():.1f} px/frame "
          f"(max {speeds.max():.1f}) — for context on why static-position "
          f"matching (--no-velocity) can undercount churn")
    print(f"Candidate fragment pairs found: {n_candidates}  "
          f"({n_candidates / max(n_tracks, 1) * 100:.1f}% of tracks look like they "
          f"continue another)")
    print(f"  ...of which class flipped:    {n_class_mismatch}")
    if n_candidates:
        print(f"  median gap:                    {candidates['gap_frames'].median():.0f} frames")
        print(f"  median distance:               {candidates['dist_boxwidths'].median():.2f} box-widths")
        out_path = args.csv_path.rsplit(".", 1)[0] + "_candidate_splits.csv"
        candidates.to_csv(out_path, index=False)
        print(f"\nFull candidate list written to: {out_path}")
        print("(review it — this is a heuristic, not ground truth)")
    else:
        print("\nNo candidate fragments found with current thresholds. If you suspect "
              "churn is being missed, try loosening --max-gap-frames / "
              "--max-dist-boxwidths.")


if __name__ == "__main__":
    main()