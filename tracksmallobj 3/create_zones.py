#!/usr/bin/env python3
"""
Automatic Scene Zone Detection

Pipeline:
    Video
        ↓
    YOLOv8 Segmentation
        ↓
    ByteTrack Tracking
        ↓
    Ground Point Extraction
        ↓
    Heatmap Generation
        ↓
    Automatic Zone Classification
        ↓
    Polygon Extraction
        ↓
    Export

Outputs

output/
    images/
    labels/
    background.png
    person_heatmap.png
    vehicle_heatmap.png
    road_mask.png
    sidewalk_mask.png
    crossing_mask.png
    zones_overlay.png
    zones.json
"""

import os
import cv2
import json
import argparse
import numpy as np

from collections import defaultdict

from ultralytics import YOLO

from tqdm import tqdm



#############################################################
# COCO CLASS IDS
#############################################################

PERSON = 0
CAR = 2
MOTORBIKE = 3
BUS = 5
TRUCK = 7

VEHICLE_CLASSES = {
    CAR,
    MOTORBIKE,
    BUS,
    TRUCK
}

TARGET_CLASSES = {
    PERSON,
    CAR,
    MOTORBIKE,
    BUS,
    TRUCK
}

CLASS_NAMES = {
    PERSON: "person",
    CAR: "car",
    MOTORBIKE: "motorbike",
    BUS: "bus",
    TRUCK: "truck"
}


#############################################################
# COLORS
#############################################################

ROAD_COLOR = (0, 0, 255)
SIDEWALK_COLOR = (0, 255, 0)
CROSSING_COLOR = (0, 255, 255)

PERSON_COLOR = (255, 0, 0)
VEHICLE_COLOR = (0, 0, 255)


#############################################################
# ARGUMENTS
#############################################################

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
        help="Input video"
    )

    parser.add_argument(
        "--output",
        default="output"
    )

    parser.add_argument(
        "--model",
        default="yolov8s-seg.pt"
    )

    parser.add_argument(
        "--device",
        default=None
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.40
    )

    parser.add_argument(
        "--frame-interval",
        type=int,
        default=5,
        help="Process every Nth frame"
    )

    parser.add_argument(
        "--save-labels",
        action="store_true"
    )

    parser.add_argument(
        "--skip-empty",
        action="store_true"
    )

    parser.add_argument(
        "--heat-radius",
        type=int,
        default=7
    )

    parser.add_argument(
        "--trajectory-thickness",
        type=int,
        default=5
    )

    parser.add_argument(
        "--background-samples",
        type=int,
        default=200
    )

    return parser.parse_args()


#############################################################
# OUTPUT FOLDERS
#############################################################

def create_directories(output_root):

    folders = {}

    folders["root"] = output_root

    folders["images"] = os.path.join(output_root, "images")

    folders["labels"] = os.path.join(output_root, "labels")

    os.makedirs(folders["root"], exist_ok=True)
    os.makedirs(folders["images"], exist_ok=True)
    os.makedirs(folders["labels"], exist_ok=True)

    return folders


#############################################################
# MEDIAN BACKGROUND ESTIMATION
#############################################################

def estimate_background(video_path, samples=200):

    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0:
        raise RuntimeError("Video contains no frames.")

    sample_indices = np.linspace(
        0,
        frame_count - 1,
        min(samples, frame_count),
        dtype=np.int32
    )

    frames = []

    print("Estimating background...")

    for idx in tqdm(sample_indices):

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

        ok, frame = cap.read()

        if ok:
            frames.append(frame)

    cap.release()

    background = np.median(
        np.stack(frames),
        axis=0
    ).astype(np.uint8)

    return background

#############################################################
# HEATMAP UTILITIES
#############################################################

def normalize_heatmap(hm):

    if hm.max() == 0:
        return hm

    return hm / hm.max()


def save_heatmap(path, heatmap):

    normalized = normalize_heatmap(heatmap)

    image = (normalized * 255).astype(np.uint8)

    image = cv2.applyColorMap(
        image,
        cv2.COLORMAP_JET
    )

    cv2.imwrite(path, image)

#############################################################
# TRACK HISTORY
#############################################################

class TrajectoryManager:

    def __init__(self):

        self.history = defaultdict(list)

    def add(self, track_id, point):

        self.history[track_id].append(point)

    def get(self, track_id):

        return self.history[track_id]

    def draw(
        self,
        image,
        track_id,
        color,
        thickness=3
    ):

        pts = self.history[track_id]

        if len(pts) < 2:
            return

        cv2.polylines(
            image,
            [np.array(pts, dtype=np.int32)],
            False,
            color,
            thickness
        )

#############################################################
# YOLO SEGMENTATION LABEL EXPORT
#############################################################

def save_yolo_segmentation_labels(path, result):

    if result.masks is None:
        return

    boxes = result.boxes
    masks = result.masks

    h, w = result.orig_shape

    with open(path, "w") as f:

        for box, polygon in zip(boxes, masks.xy):

            cls = int(box.cls.item())

            if cls not in TARGET_CLASSES:
                continue

            class_name = CLASS_NAMES[cls]

            polygon = np.asarray(polygon)

            if len(polygon) < 3:
                continue

            polygon[:, 0] /= w
            polygon[:, 1] /= h

            coords = []

            for x, y in polygon:
                coords.append(f"{x:.6f}")
                coords.append(f"{y:.6f}")

            class_id = {
                "person": 0,
                "car": 1,
                "motorbike": 2,
                "bus": 3,
                "truck": 4
            }[class_name]

            f.write(
                str(class_id) +
                " " +
                " ".join(coords) +
                "\n"
            )


#############################################################
# GROUND CONTACT POINT
#############################################################

def ground_contact(box):

    x1, y1, x2, y2 = box

    x = int((x1 + x2) / 2)
    y = int(y2)

    return (x, y)


#############################################################
# MAIN DETECTION LOOP
#############################################################

def process_video(args, folders):

    print("Loading model...")

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    person_heatmap = np.zeros(
        (height, width),
        dtype=np.float32
    )

    vehicle_heatmap = np.zeros(
        (height, width),
        dtype=np.float32
    )

    trajectory_manager = TrajectoryManager()

    frame_id = 0

    print("Running segmentation and tracking...")

    progress = tqdm(total=frame_count)

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        progress.update(1)

        if frame_id % args.frame_interval != 0:
            frame_id += 1
            continue

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=args.conf,
            device=args.device,
            verbose=False
        )

        if len(results) == 0:
            frame_id += 1
            continue

        result = results[0]

        annotated = result.plot()

        has_detection = False

        if result.boxes is not None:

            boxes = result.boxes

            for box in boxes:

                cls = int(box.cls.item())

                if cls not in TARGET_CLASSES:
                    continue

                has_detection = True

                xyxy = box.xyxy.cpu().numpy()[0]

                point = ground_contact(xyxy)

                if box.id is None:
                    track_id = -1
                else:
                    track_id = int(box.id.item())

                trajectory_manager.add(
                    track_id,
                    point
                )

                if cls == PERSON:

                    cv2.circle(
                        person_heatmap,
                        point,
                        args.heat_radius,
                        1,
                        -1
                    )

                    trajectory_manager.draw(
                        annotated,
                        track_id,
                        PERSON_COLOR,
                        args.trajectory_thickness
                    )

                else:

                    cv2.circle(
                        vehicle_heatmap,
                        point,
                        args.heat_radius,
                        1,
                        -1
                    )

                    trajectory_manager.draw(
                        annotated,
                        track_id,
                        VEHICLE_COLOR,
                        args.trajectory_thickness
                    )

                cv2.circle(
                    annotated,
                    point,
                    4,
                    (255, 255, 255),
                    -1
                )

        if args.skip_empty and not has_detection:
            frame_id += 1
            continue

        image_name = f"frame_{frame_id:06d}.jpg"

        image_path = os.path.join(
            folders["images"],
            image_name
        )

        cv2.imwrite(
            image_path,
            annotated
        )

        if args.save_labels:

            label_path = os.path.join(
                folders["labels"],
                image_name.replace(".jpg", ".txt")
            )

            save_yolo_segmentation_labels(
                label_path,
                result
            )

        frame_id += 1

    progress.close()

    cap.release()

    return (
        person_heatmap,
        vehicle_heatmap,
        trajectory_manager
    )

#############################################################
# HEATMAP POST-PROCESSING
#############################################################

def smooth_heatmap(heatmap):

    if heatmap.max() == 0:
        return heatmap

    heatmap = cv2.GaussianBlur(
        heatmap,
        (0, 0),
        sigmaX=20
    )

    heatmap = normalize_heatmap(heatmap)

    return heatmap


#############################################################
# AUTOMATIC ZONE CLASSIFICATION
#############################################################

def infer_zones(person, vehicle):

    road = np.zeros_like(vehicle, dtype=np.uint8)
    sidewalk = np.zeros_like(vehicle, dtype=np.uint8)
    crossing = np.zeros_like(vehicle, dtype=np.uint8)

    for y in range(vehicle.shape[0]):

        for x in range(vehicle.shape[1]):

            p = person[y, x]
            v = vehicle[y, x]

            if v > 0.60 and p < 0.30:
                road[y, x] = 255

            elif p > 0.60 and v < 0.30:
                sidewalk[y, x] = 255

            elif p > 0.40 and v > 0.40:
                crossing[y, x] = 255

    kernel = np.ones((9, 9), np.uint8)

    road = cv2.morphologyEx(
        road,
        cv2.MORPH_CLOSE,
        kernel
    )

    sidewalk = cv2.morphologyEx(
        sidewalk,
        cv2.MORPH_CLOSE,
        kernel
    )

    crossing = cv2.morphologyEx(
        crossing,
        cv2.MORPH_CLOSE,
        kernel
    )

    return road, sidewalk, crossing


#############################################################
# POLYGON EXTRACTION
#############################################################

def mask_to_polygons(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 500:
            continue

        contour = cv2.approxPolyDP(
            contour,
            5,
            True
        )

        poly = []

        for p in contour:

            poly.append([
                int(p[0][0]),
                int(p[0][1])
            ])

        polygons.append(poly)

    return polygons


#############################################################
# SAVE JSON
#############################################################

def save_zone_json(path,
                   road,
                   sidewalk,
                   crossing):

    data = {

        "road":
            mask_to_polygons(road),

        "sidewalk":
            mask_to_polygons(sidewalk),

        "crossing":
            mask_to_polygons(crossing)
    }

    with open(path, "w") as f:

        json.dump(
            data,
            f,
            indent=4
        )


#############################################################
# OVERLAY VISUALIZATION
#############################################################

def overlay_zones(
        background,
        road,
        sidewalk,
        crossing):

    overlay = background.copy()

    overlay[road > 0] = ROAD_COLOR

    overlay[sidewalk > 0] = SIDEWALK_COLOR

    overlay[crossing > 0] = CROSSING_COLOR

    result = cv2.addWeighted(
        background,
        0.6,
        overlay,
        0.4,
        0
    )

    return result


#############################################################
# MAIN
#############################################################

def main():

    args = parse_args()

    folders = create_directories(
        args.output
    )

    background = estimate_background(
        args.video,
        args.background_samples
    )

    cv2.imwrite(
        os.path.join(
            folders["root"],
            "background.png"
        ),
        background
    )

    person_heatmap, vehicle_heatmap, tracks = process_video(
        args,
        folders
    )

    person_heatmap = smooth_heatmap(
        person_heatmap
    )

    vehicle_heatmap = smooth_heatmap(
        vehicle_heatmap
    )

    save_heatmap(
        os.path.join(
            folders["root"],
            "person_heatmap.png"
        ),
        person_heatmap
    )

    save_heatmap(
        os.path.join(
            folders["root"],
            "vehicle_heatmap.png"
        ),
        vehicle_heatmap
    )

    road, sidewalk, crossing = infer_zones(
        person_heatmap,
        vehicle_heatmap
    )

    cv2.imwrite(
        os.path.join(
            folders["root"],
            "road_mask.png"
        ),
        road
    )

    cv2.imwrite(
        os.path.join(
            folders["root"],
            "sidewalk_mask.png"
        ),
        sidewalk
    )

    cv2.imwrite(
        os.path.join(
            folders["root"],
            "crossing_mask.png"
        ),
        crossing
    )

    overlay = overlay_zones(
        background,
        road,
        sidewalk,
        crossing
    )

    cv2.imwrite(
        os.path.join(
            folders["root"],
            "zones_overlay.png"
        ),
        overlay
    )

    save_zone_json(
        os.path.join(
            folders["root"],
            "zones.json"
        ),
        road,
        sidewalk,
        crossing
    )

    print()

    print("=" * 60)
    print("Processing complete.")
    print("=" * 60)
    print()

    print("Outputs saved to:")
    print(folders["root"])


if __name__ == "__main__":
    main()