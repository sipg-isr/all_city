#!/usr/bin/env python3
"""
segment_video.py

Turn a video into a folder of labeled/segmented images using a YOLOv8
segmentation model (Ultralytics), filtered to a fixed set of classes
(person, car, truck, motorbike by default).

For each sampled frame, this script can save:
  1. An annotated image (segmentation masks + boxes + class labels drawn
     on the frame) -> <output>/images/
  2. (optional) A YOLO-seg format label .txt with normalized polygon
     coordinates per instance -> <output>/labels/
     Plus a classes.txt / data.yaml so the folder is a ready-to-use
     YOLO segmentation dataset.

Usage:
    python segment_video.py --video path/to/video.mp4 --output out_dir

Common options:
    --model yolov8s-seg.pt      # bigger/more accurate model (default: yolov8n-seg.pt)
    --classes person,car,truck,motorcycle
    --conf 0.4                  # confidence threshold
    --frame-interval 5          # save every 5th frame instead of every frame
    --save-labels                # also write YOLO-seg label .txt files
    --skip-empty                 # don't save frames with zero target detections
    --device cuda                # force device (default: auto)

Requires: pip install ultralytics opencv-python
"""

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO class name -> the name we want to use in our output labels.
# Ultralytics' pretrained COCO weights call it "motorcycle"; this project's
# convention (see ingest_tracks.py normalization map) uses "motorbike".
CLASS_RENAME = {
    "motorcycle": "motorbike",
}

DEFAULT_CLASSES = ["person", "car", "truck", "motorcycle"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument("--output", default="segmented_output", help="Output directory (default: segmented_output)")
    p.add_argument("--model", default="yolov8n-seg.pt",
                    help="Ultralytics segmentation model/weights (default: yolov8n-seg.pt). "
                         "Use yolov8s-seg.pt / yolov8m-seg.pt / yolov8l-seg.pt for better accuracy "
                         "(slower). If you have a custom-trained .pt for this project, point at that instead.")
    p.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                    help=f"Comma-separated COCO class names to keep (default: {','.join(DEFAULT_CLASSES)})")
    p.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    p.add_argument("--frame-interval", type=int, default=1,
                    help="Process every Nth frame (default: 1 = every frame). Use e.g. 5-30 for long videos.")
    p.add_argument("--save-labels", action="store_true",
                    help="Also export YOLO-seg polygon label .txt files (for training a model later)")
    p.add_argument("--skip-empty", action="store_true",
                    help="Don't save frames that have zero detections of the target classes")
    p.add_argument("--device", default=None, help="Force device, e.g. 'cpu' or 'cuda:0' (default: auto)")
    p.add_argument("--img-prefix", default="frame", help="Filename prefix for saved images (default: frame)")
    return p.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"Error: video not found: {video_path}")

    out_dir = Path(args.output)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    if args.save_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)

    wanted_names = [c.strip() for c in args.classes.split(",") if c.strip()]

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Map wanted COCO class names -> their integer ids in this model
    model_names = model.names  # {id: name}
    name_to_id = {v: k for k, v in model_names.items()}
    missing = [n for n in wanted_names if n not in name_to_id]
    if missing:
        sys.exit(f"Error: these class names aren't in the model's class list: {missing}\n"
                  f"Available classes: {sorted(model_names.values())}")
    target_ids = [name_to_id[n] for n in wanted_names]

    # Build a stable local class index (0..N-1) for label export, using the
    # renamed labels (e.g. motorcycle -> motorbike) in a fixed order.
    output_class_names = [CLASS_RENAME.get(n, n) for n in wanted_names]
    local_id_for_model_id = {mid: i for i, mid in enumerate(target_ids)}

    if args.save_labels:
        (out_dir / "classes.txt").write_text("\n".join(output_class_names) + "\n")
        data_yaml = out_dir / "data.yaml"
        data_yaml.write_text(
            "path: .\n"
            "train: images\n"
            "val: images\n"
            f"nc: {len(output_class_names)}\n"
            f"names: {output_class_names}\n"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Error: could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    print(f"Video: {video_path.name} | ~{total_frames} frames | {fps:.1f} fps")
    print(f"Sampling every {args.frame_interval} frame(s), classes={wanted_names}, conf>={args.conf}")

    frame_idx = 0
    saved_count = 0
    detection_counts = {n: 0 for n in output_class_names}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % args.frame_interval == 0:
            results = model.predict(
                frame,
                classes=target_ids,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )
            result = results[0]
            num_dets = 0 if result.boxes is None else len(result.boxes)

            if num_dets > 0 or not args.skip_empty:
                stem = f"{args.img_prefix}_{frame_idx:06d}"

                # Annotated image with masks + boxes + labels drawn
                annotated = result.plot()  # BGR numpy array
                cv2.imwrite(str(images_dir / f"{stem}.jpg"), annotated)
                saved_count += 1

                if num_dets > 0:
                    for cls_id in result.boxes.cls.tolist():
                        name = CLASS_RENAME.get(model_names[int(cls_id)], model_names[int(cls_id)])
                        detection_counts[name] = detection_counts.get(name, 0) + 1

                if args.save_labels:
                    label_lines = []
                    if result.masks is not None:
                        # xyn: list of (N,2) arrays of normalized polygon points, one per instance
                        polys = result.masks.xyn
                        cls_ids = result.boxes.cls.tolist()
                        for poly, cls_id in zip(polys, cls_ids):
                            local_id = local_id_for_model_id[int(cls_id)]
                            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                            label_lines.append(f"{local_id} {coords}")
                    (labels_dir / f"{stem}.txt").write_text("\n".join(label_lines))

        frame_idx += 1
        if total_frames and frame_idx % max(1, total_frames // 20) == 0:
            pct = 100 * frame_idx / total_frames
            print(f"  ...{pct:.0f}% ({frame_idx}/{total_frames} frames read, {saved_count} saved)")

    cap.release()

    print("\nDone.")
    print(f"Frames read: {frame_idx}")
    print(f"Images saved: {saved_count}  -> {images_dir}")
    if args.save_labels:
        print(f"Labels saved -> {labels_dir}")
    print("Detections by class:")
    for name, count in detection_counts.items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()