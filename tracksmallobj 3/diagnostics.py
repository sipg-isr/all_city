#!/usr/bin/env python3

"""
Track Consistency Analysis

Reads a tracker CSV and analyses:

1. Track consistency
2. Class transition matrix
3. Bounding box area vs switches
4. Spatial heatmap of class switches
5. Track duration vs consistency
6. Timeline plots

Author: ChatGPT
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter, defaultdict
from matplotlib.colors import LogNorm


##############################################################################
# Arguments
##############################################################################

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help="Tracking CSV"
    )

    parser.add_argument(
        "--output",
        default="analysis"
    )

    parser.add_argument(
        "--top-tracks",
        type=int,
        default=10,
        help="Number of representative tracks to plot"
    )

    return parser.parse_args()


##############################################################################
# Helpers
##############################################################################

CLASS_COLORS = {
    "person": "#4C72B0",
    "car": "#DD8452",
    "truck": "#55A868",
    "motorbike": "#C44E52",
    "bus": "#8172B3",
    "bicycle": "#937860",
}

CLASS_TO_INT = {
    "person":0,
    "car":1,
    "truck":2,
    "motorbike":3,
    "bus":4,
    "bicycle":5
}


##############################################################################
# Read CSV
##############################################################################

def load_data(csv_path):

    df = pd.read_csv(csv_path)

    df = df.sort_values(
        ["track_id","frame_id"]
    )

    df["area"] = df["width"] * df["height"]

    return df


##############################################################################
# Compute statistics for every track
##############################################################################

def compute_track_statistics(df):

    rows = []

    transition_counter = Counter()

    switch_locations = []

    representative_tracks = {}

    grouped = df.groupby("track_id")

    for track_id, track in grouped:

        track = track.sort_values("frame_id")

        classes = track["class_name"].tolist()

        frames = track["frame_id"].tolist()

        areas = track["area"].tolist()

        cx = track["cx"].tolist()

        cy = track["cy"].tolist()

        duration = len(track)

        counts = Counter(classes)

        majority = counts.most_common(1)[0][0]

        consistency = (
            counts[majority] /
            duration
        )

        switches = 0

        previous = classes[0]

        for i in range(1, len(classes)):

            current = classes[i]

            if current != previous:

                switches += 1

                transition_counter[
                    (previous,current)
                ] += 1

                switch_locations.append(
                    (
                        cx[i],
                        cy[i]
                    )
                )

            previous = current

        representative_tracks[track_id] = track

        rows.append({

            "track_id":track_id,

            "frames":duration,

            "majority_class":majority,

            "consistency":consistency,

            "switches":switches,

            "mean_area":np.mean(areas),

            "mean_x":np.mean(cx),

            "mean_y":np.mean(cy)

        })

    stats = pd.DataFrame(rows)

    return (
        stats,
        transition_counter,
        switch_locations,
        representative_tracks
    )


##############################################################################
# Save statistics
##############################################################################

def save_statistics(stats, output):

    stats = stats.sort_values(
        "consistency"
    )

    stats.to_csv(

        os.path.join(
            output,
            "track_statistics.csv"
        ),

        index=False
    )

    print()

    print("="*60)

    print("Tracks:",len(stats))

    print("Average duration:",
          round(stats.frames.mean(),2))

    print("Average consistency:",
          round(stats.consistency.mean(),3))

    print("Average switches:",
          round(stats.switches.mean(),2))

    print("="*60)


    ##############################################################################
# Figure 1
# Track consistency histogram
##############################################################################

def plot_consistency_histogram(stats, output):

    plt.figure(figsize=(8,5))

    plt.hist(
        stats["consistency"],
        bins=20,
        edgecolor="black"
    )

    plt.xlabel("Track consistency")
    plt.ylabel("Number of tracks")
    plt.title("Track Consistency Distribution")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output,
            "consistency_histogram.png"
        ),
        dpi=300
    )

    plt.close()


##############################################################################
# Figure 2
# Transition matrix
##############################################################################

def plot_transition_matrix(
        transition_counter,
        output):

    classes = sorted(CLASS_TO_INT.keys())

    matrix = np.zeros(
        (
            len(classes),
            len(classes)
        ),
        dtype=int
    )

    for (a,b), count in transition_counter.items():

        if a not in CLASS_TO_INT:
            continue

        if b not in CLASS_TO_INT:
            continue

        i = CLASS_TO_INT[a]
        j = CLASS_TO_INT[b]

        matrix[i,j] += count

    fig, ax = plt.subplots(figsize=(7,6))

    im = ax.imshow(
        matrix,
        cmap="Blues"
    )

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))

    ax.set_xticklabels(classes,
                       rotation=45,
                       ha="right")

    ax.set_yticklabels(classes)

    ax.set_xlabel("To")
    ax.set_ylabel("From")

    ax.set_title(
        "Class Transition Matrix"
    )

    for i in range(matrix.shape[0]):

        for j in range(matrix.shape[1]):

            ax.text(
                j,
                i,
                str(matrix[i,j]),
                ha="center",
                va="center",
                color="black"
            )

    fig.colorbar(im)

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            output,
            "transition_matrix.png"
        ),

        dpi=300
    )

    plt.close()


##############################################################################
# Figure 3
# Area vs switches
##############################################################################

def plot_area_vs_switches(
        stats,
        output):

    plt.figure(figsize=(7,5))

    plt.scatter(

        stats["mean_area"],

        stats["switches"],

        alpha=0.6

    )

    plt.xscale("log")

    plt.xlabel(
        "Average bounding box area (pixels²)"
    )

    plt.ylabel(
        "Number of class switches"
    )

    plt.title(
        "Bounding Box Area vs Class Switches"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            output,
            "area_vs_switches.png"
        ),

        dpi=300
    )

    plt.close()


##############################################################################
# Figure 4
# Duration vs consistency
##############################################################################

def plot_duration_vs_consistency(
        stats,
        output):

    plt.figure(figsize=(7,5))

    plt.scatter(

        stats["frames"],

        stats["consistency"],

        alpha=0.6

    )

    plt.xlabel(
        "Track duration (frames)"
    )

    plt.ylabel(
        "Consistency"
    )

    plt.title(
        "Track Duration vs Consistency"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            output,
            "duration_vs_consistency.png"
        ),

        dpi=300
    )

    plt.close()


##############################################################################
# Print useful summary
##############################################################################

def print_summary(stats):

    print()

    print("="*70)

    print("SUMMARY")

    print("="*70)

    print()

    print(
        f"Tracks analysed: {len(stats)}"
    )

    print(
        f"Average consistency: {stats.consistency.mean():.3f}"
    )

    print(
        f"Median consistency: {stats.consistency.median():.3f}"
    )

    print(
        f"Tracks with switches: {(stats.switches>0).sum()}"
    )

    print(
        f"Perfectly stable tracks: {(stats.consistency==1).sum()}"
    )

    print()

    print("Worst tracks:")

    print()

    print(

        stats.sort_values(
            "consistency"
        )[[
            "track_id",
            "majority_class",
            "frames",
            "consistency",
            "switches"
        ]].head(15)

    )

    print()

    print("="*70)
##############################################################################
# Figure 5
# Spatial heatmap of class switches
##############################################################################

def plot_switch_heatmap(switch_locations, output):

    if len(switch_locations) == 0:
        print("No class switches detected.")
        return

    pts = np.array(switch_locations)

    plt.figure(figsize=(8,6))

    plt.hist2d(
        pts[:,0],
        pts[:,1],
        bins=80,
        cmap="hot",
        norm=LogNorm()
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Image X")
    plt.ylabel("Image Y")

    plt.title("Spatial Distribution of Class Switches")

    plt.colorbar(label="Switch count")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output,
            "switch_heatmap.png"
        ),
        dpi=300
    )

    plt.close()


##############################################################################
# Figure 6
# Timeline of unstable tracks
##############################################################################

def plot_track_timelines(
        representative_tracks,
        stats,
        output,
        top_tracks=10):

    unstable = stats.sort_values(
        "consistency"
    ).head(top_tracks)

    fig, axes = plt.subplots(
        len(unstable),
        1,
        figsize=(12,2.5*len(unstable)),
        sharex=False
    )

    if len(unstable) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, unstable.iterrows()):

        track = representative_tracks[
            row.track_id
        ]

        frames = track.frame_id.values

        classes = track.class_name.values

        y = [
            CLASS_TO_INT.get(c, -1)
            for c in classes
        ]

        colors = [
            CLASS_COLORS.get(c,"gray")
            for c in classes
        ]

        ax.scatter(
            frames,
            y,
            c=colors,
            s=25
        )

        ax.plot(
            frames,
            y,
            alpha=0.25
        )

        ax.set_title(
            f"Track {row.track_id} "
            f"(consistency={row.consistency:.2f})"
        )

        ax.set_yticks(
            list(CLASS_TO_INT.values())
        )

        ax.set_yticklabels(
            list(CLASS_TO_INT.keys())
        )

        ax.grid(alpha=0.3)

    plt.xlabel("Frame")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output,
            "representative_tracks.png"
        ),
        dpi=300
    )

    plt.close()


##############################################################################
# Correlation matrix
##############################################################################

def plot_correlations(stats, output):

    corr = stats[
        [
            "frames",
            "consistency",
            "switches",
            "mean_area"
        ]
    ].corr()

    fig, ax = plt.subplots(figsize=(6,5))

    im = ax.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    labels = corr.columns

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):

            ax.text(
                j,
                i,
                f"{corr.iloc[i,j]:.2f}",
                ha="center",
                va="center"
            )

    fig.colorbar(im)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output,
            "correlation_matrix.png"
        ),
        dpi=300
    )

    plt.close()


##############################################################################
# Export every switch
##############################################################################

def export_switch_events(df, output):

    rows = []

    grouped = df.groupby("track_id")

    for track_id, track in grouped:

        track = track.sort_values("frame_id")

        prev = None

        for _, r in track.iterrows():

            if prev is not None:

                if prev.class_name != r.class_name:

                    rows.append({

                        "track_id":track_id,

                        "frame":r.frame_id,

                        "from":prev.class_name,

                        "to":r.class_name,

                        "cx":r.cx,

                        "cy":r.cy,

                        "area":r.area

                    })

            prev = r

    pd.DataFrame(rows).to_csv(

        os.path.join(
            output,
            "track_switches.csv"
        ),

        index=False
    )


##############################################################################
# Save text report
##############################################################################

def save_report(stats, output):

    report = []

    report.append("TRACK ANALYSIS REPORT")
    report.append("="*60)
    report.append("")

    report.append(
        f"Tracks analysed: {len(stats)}"
    )

    report.append(
        f"Average duration: {stats.frames.mean():.2f}"
    )

    report.append(
        f"Average consistency: {stats.consistency.mean():.3f}"
    )

    report.append(
        f"Median consistency: {stats.consistency.median():.3f}"
    )

    report.append(
        f"Tracks with switches: {(stats.switches>0).sum()}"
    )

    report.append(
        f"Maximum switches: {stats.switches.max()}"
    )

    report.append("")
    report.append("Worst tracks")
    report.append("")

    worst = stats.sort_values(
        "consistency"
    ).head(20)

    report.append(
        worst.to_string(index=False)
    )

    with open(
        os.path.join(
            output,
            "summary.txt"
        ),
        "w"
    ) as f:

        f.write("\n".join(report))


##############################################################################
# MAIN
##############################################################################

def main():

    args = parse_args()

    os.makedirs(
        args.output,
        exist_ok=True
    )

    print("Loading CSV...")

    df = load_data(args.csv)

    print("Computing statistics...")

    (
        stats,
        transition_counter,
        switch_locations,
        representative_tracks

    ) = compute_track_statistics(df)

    save_statistics(
        stats,
        args.output
    )

    print_summary(stats)

    print("Generating figures...")

    plot_consistency_histogram(
        stats,
        args.output
    )

    plot_transition_matrix(
        transition_counter,
        args.output
    )

    plot_area_vs_switches(
        stats,
        args.output
    )

    plot_duration_vs_consistency(
        stats,
        args.output
    )

    plot_switch_heatmap(
        switch_locations,
        args.output
    )

    plot_track_timelines(
        representative_tracks,
        stats,
        args.output,
        args.top_tracks
    )

    plot_correlations(
        stats,
        args.output
    )

    export_switch_events(
        df,
        args.output
    )

    save_report(
        stats,
        args.output
    )

    print()

    print("="*60)
    print("Analysis complete")
    print("="*60)
    print(f"Results saved to: {args.output}")



if __name__ == "__main__":
    main()

