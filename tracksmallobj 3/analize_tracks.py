#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from collections import defaultdict

##############################################################################
# PARAMETERS
##############################################################################

IOU_THRESHOLD = 0.25
MAX_DISTANCE = 60
AREA_RATIO_LIMIT = 2.0

##############################################################################
# ARGUMENTS
##############################################################################

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True
    )

    parser.add_argument(
        "--output",
        default="analysis"
    )

    return parser.parse_args()

##############################################################################
# IOU
##############################################################################

def compute_iou(box1, box2):

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])

    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0, xB-xA) * max(0, yB-yA)

    if inter == 0:
        return 0.0

    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

    return inter/(area1+area2-inter)

##############################################################################
# COST FUNCTION
##############################################################################

def match_cost(det1, det2):

    box1 = (
        det1.x1,
        det1.y1,
        det1.x2,
        det1.y2
    )

    box2 = (
        det2.x1,
        det2.y1,
        det2.x2,
        det2.y2
    )

    iou = compute_iou(box1, box2)

    distance = np.sqrt(
        (det1.cx-det2.cx)**2 +
        (det1.cy-det2.cy)**2
    )

    area_ratio = max(
        det1.area,
        det2.area
    ) / max(
        1,
        min(det1.area, det2.area)
    )

    if distance > MAX_DISTANCE:
        return 9999

    if area_ratio > AREA_RATIO_LIMIT:
        return 9999

    cost = (
        0.55*(1-iou)
        +
        0.30*(distance/MAX_DISTANCE)
        +
        0.15*min(area_ratio/AREA_RATIO_LIMIT,1)
    )

    if det1.track_id == det2.track_id:
        cost -= 0.15

    return cost

##############################################################################
# LOAD CSV
##############################################################################

def load_csv(path):

    df = pd.read_csv(path)

    df["area"] = df.width * df.height

    df = df.sort_values(
        ["frame_id","track_id"]
    )

    return df

##############################################################################
# GROUP BY FRAME
##############################################################################

def group_frames(df):

    frames = {}

    for frame_id, group in df.groupby("frame_id"):

        frames[frame_id] = group.reset_index(drop=True)

    return frames

##############################################################################
# MATCH TWO FRAMES
##############################################################################

def match_frames(frameA, frameB):

    n = len(frameA)
    m = len(frameB)

    cost = np.zeros((n,m))

    for i in range(n):

        for j in range(m):

            cost[i,j] = match_cost(
                frameA.iloc[i],
                frameB.iloc[j]
            )

    rows, cols = linear_sum_assignment(cost)

    matches = []

    for r,c in zip(rows,cols):

        if cost[r,c] > 1:
            continue

        matches.append(
            (
                frameA.iloc[r],
                frameB.iloc[c],
                cost[r,c]
            )
        )

    return matches
##############################################################################
# MATCH ALL CONSECUTIVE FRAMES
##############################################################################

def analyze_matches(frames):

    switch_events = []

    id_switches = []

    object_histories = defaultdict(list)

    frame_numbers = sorted(frames.keys())

    print(f"Analyzing {len(frame_numbers)} frames...")

    for idx in range(len(frame_numbers)-1):

        f1 = frame_numbers[idx]
        f2 = frame_numbers[idx+1]

        frameA = frames[f1]
        frameB = frames[f2]

        matches = match_frames(frameA, frameB)

        for detA, detB, cost in matches:

            boxA = (
                detA.x1,
                detA.y1,
                detA.x2,
                detA.y2
            )

            boxB = (
                detB.x1,
                detB.y1,
                detB.x2,
                detB.y2
            )

            iou = compute_iou(boxA, boxB)

            distance = np.sqrt(
                (detA.cx-detB.cx)**2 +
                (detA.cy-detB.cy)**2
            )

            area_ratio = max(
                detA.area,
                detB.area
            ) / max(
                1,
                min(detA.area, detB.area)
            )

            history = {

                "frame":f1,

                "next_frame":f2,

                "track1":int(detA.track_id),

                "track2":int(detB.track_id),

                "class1":detA.class_name,

                "class2":detB.class_name,

                "iou":iou,

                "distance":distance,

                "area_ratio":area_ratio,

                "cost":cost,

                "cx":detB.cx,

                "cy":detB.cy

            }

            object_histories[int(detA.track_id)].append(history)

            ####################################################
            # CLASS SWITCH
            ####################################################

            if detA.class_name != detB.class_name:

                switch_events.append({

                    "frame":f2,

                    "track_old":int(detA.track_id),

                    "track_new":int(detB.track_id),

                    "old_class":detA.class_name,

                    "new_class":detB.class_name,

                    "iou":iou,

                    "distance":distance,

                    "area_ratio":area_ratio,

                    "cost":cost,

                    "cx":detB.cx,

                    "cy":detB.cy,

                    "x1":detB.x1,
                    "y1":detB.y1,
                    "x2":detB.x2,
                    "y2":detB.y2

                })

            ####################################################
            # TRACK ID SWITCH
            ####################################################

            if detA.track_id != detB.track_id:

                id_switches.append({

                    "frame":f2,

                    "old_track":int(detA.track_id),

                    "new_track":int(detB.track_id),

                    "class":detB.class_name,

                    "iou":iou,

                    "distance":distance,

                    "cost":cost

                })

    switches = pd.DataFrame(switch_events)

    ids = pd.DataFrame(id_switches)

    return switches, ids, object_histories


##############################################################################
# OBJECT STATISTICS
##############################################################################

def compute_object_statistics(object_histories):

    rows = []

    for track_id, history in object_histories.items():

        if len(history) == 0:
            continue

        classes = []

        switches = 0

        previous = history[0]["class1"]

        ious = []

        distances = []

        areas = []

        for h in history:

            classes.append(h["class1"])

            ious.append(h["iou"])

            distances.append(h["distance"])

            areas.append(h["area_ratio"])

            if h["class2"] != previous:

                switches += 1

            previous = h["class2"]

        majority = Counter(classes).most_common(1)[0][0]

        consistency = classes.count(majority)/len(classes)

        rows.append({

            "track_id":track_id,

            "frames":len(history),

            "majority_class":majority,

            "consistency":consistency,

            "switches":switches,

            "mean_iou":np.mean(ious),

            "mean_distance":np.mean(distances),

            "mean_area_ratio":np.mean(areas)

        })

    return pd.DataFrame(rows)


##############################################################################
# SAVE CSVs
##############################################################################

def save_results(
    switches,
    ids,
    stats,
    output
):

    switches.to_csv(

        os.path.join(
            output,
            "switch_events.csv"
        ),

        index=False
    )

    ids.to_csv(

        os.path.join(
            output,
            "id_switches.csv"
        ),

        index=False
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

    print(f"Class switches : {len(switches)}")

    print(f"ID switches    : {len(ids)}")

    print(f"Tracks         : {len(stats)}")

    print("="*60)

    if len(switches):

        print()

        print("Most common class changes:")

        print(
            switches.groupby(
                ["old_class","new_class"]
            ).size().sort_values(
                ascending=False
            ).head(10)
        )