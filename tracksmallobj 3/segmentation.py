"""
median_background_segmentation.py

Estimate the static background of a surveillance video (road, sidewalk,
crosswalks) using the per-pixel temporal median, then use that background
to segment moving foreground objects (cars, pedestrians) in each frame.

Pure OpenCV / NumPy - no deep learning models.

Usage:
    python median_background_segmentation.py INPUT_VIDEO OUTPUT_DIR [options]

Example:
    python median_background_segmentation.py cam01.mp4 out/ \
        --num-samples 150 --threshold 25 --frame-stride 1 --format png
"""

import argparse
import os
import numpy as np
import cv2


# --------------------------------------------------------------------------
# Background estimation
# --------------------------------------------------------------------------


def sample_frame_indices(total_frames, num_samples):
    """Evenly spaced frame indices covering the whole video."""
    num_samples = min(num_samples, total_frames)
    return np.linspace(0, total_frames - 1, num_samples, dtype=int)


def grab_frames_at_indices(cap, indices):
    """
    Step through the video collecting only the requested frames.

    Uses cap.grab() (cheap - no full decode/color conversion) to skip
    unwanted frames and cap.retrieve()-via-read() only for frames we need.
    This is more reliable than cap.set(CAP_PROP_POS_FRAMES, ...), which can
    seek to the wrong frame on long-GOP encoded video.
    """
    indices = sorted(set(int(i) for i in indices))
    frames = []
    current = 0
    for target in indices:
        while current < target:
            if not cap.grab():
                break
            current += 1
        ret, frame = cap.read()
        current += 1
        if not ret:
            break
        frames.append(frame)
    return frames


def compute_median_background(video_path, num_samples=120, resize_width=None, verbose=True):
    """
    Estimate the static background of a video as the per-pixel median over
    a set of frames sampled evenly across the whole video.

    Memory scales as num_samples * H * W * 3 bytes. At 1080p, 120 samples
    is ~750MB - fine on most machines. Lower --num-samples or use
    --resize-width if you're memory constrained.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError("Could not determine frame count (corrupt file or unsupported container).")

    indices = sample_frame_indices(total_frames, num_samples)
    if verbose:
        print(f"Video has {total_frames} frames; sampling {len(indices)} of them for the median.")

    frames = grab_frames_at_indices(cap, indices)
    cap.release()

    if len(frames) < 3:
        raise RuntimeError("Not enough frames were read to compute a reliable median.")

    if resize_width:
        h, w = frames[0].shape[:2]
        scale = resize_width / w
        new_size = (resize_width, int(h * scale))
        frames = [cv2.resize(f, new_size, interpolation=cv2.INTER_AREA) for f in frames]

    stack = np.stack(frames, axis=0)  # (N, H, W, 3)
    background = np.median(stack, axis=0).astype(np.uint8)

    if verbose:
        print(f"Background estimated from {len(frames)} frames, shape {background.shape}.")
    return background


# --------------------------------------------------------------------------
# Per-frame segmentation against the background
# --------------------------------------------------------------------------


def segment_frame(frame, background, threshold=25, min_area=400, blur_ksize=5):
    """
    Foreground/background segmentation of a single frame against the
    pre-computed median background.

    Returns:
        mask:  binary uint8 mask (255 = foreground / moving object)
        boxes: list of (x, y, w, h) bounding boxes for detected blobs
    """
    if frame.shape != background.shape:
        frame = cv2.resize(frame, (background.shape[1], background.shape[0]))

    diff = cv2.absdiff(frame, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    if blur_ksize:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]

    return mask, boxes


def save_image(path, image, ext):
    """Write images with (near-)zero compression as requested."""
    if ext.lower() == "png":
        cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    elif ext.lower() in ("tif", "tiff"):
        cv2.imwrite(path, image, [cv2.IMWRITE_TIFF_COMPRESSION, 1])  # 1 = no compression
    else:  # bmp - always uncompressed
        cv2.imwrite(path, image)


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------


def process_video(video_path, output_dir, num_samples=120, threshold=25,
                   min_area=400, frame_stride=1, image_ext="png",
                   resize_width=None, draw_boxes=True):
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    print("Step 1/2: estimating background via temporal median...")
    background = compute_median_background(video_path, num_samples=num_samples, resize_width=resize_width)

    bg_path = os.path.join(output_dir, f"background.{image_ext}")
    save_image(bg_path, background, image_ext)
    print(f"Saved background reference to {bg_path}")

    print("Step 2/2: segmenting frames against the background...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_stride == 0:
            if resize_width:
                h, w = frame.shape[:2]
                scale = resize_width / w
                frame_rs = cv2.resize(frame, (resize_width, int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                frame_rs = frame

            mask, boxes = segment_frame(frame_rs, background, threshold=threshold, min_area=min_area)

            frame_name = f"frame_{idx:06d}.{image_ext}"
            mask_name = f"mask_{idx:06d}.{image_ext}"

            out_frame = frame_rs.copy()
            if draw_boxes:
                for (x, y, w, h) in boxes:
                    cv2.rectangle(out_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            save_image(os.path.join(frames_dir, frame_name), out_frame, image_ext)
            save_image(os.path.join(masks_dir, mask_name), mask, image_ext)
            saved += 1

        idx += 1
        if idx % 200 == 0:
            print(f"  processed {idx} frames, saved {saved}...")

    cap.release()
    print(f"Done. Processed {idx} frames, saved {saved} frame/mask pairs to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Median-background segmentation for surveillance video (OpenCV only, no DL models)."
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("output_dir", help="Directory to save background, frames and masks.")
    parser.add_argument("--num-samples", type=int, default=120,
                         help="Number of frames sampled (evenly across the video) to estimate the median background.")
    parser.add_argument("--threshold", type=int, default=25,
                         help="Pixel intensity difference threshold for foreground (raise if you get noisy masks, e.g. from shadows).")
    parser.add_argument("--min-area", type=int, default=400,
                         help="Minimum contour area in pixels to keep as a detected object (filters small noise blobs).")
    parser.add_argument("--frame-stride", type=int, default=1,
                         help="Process every Nth frame (1 = every frame, 2 = every other, etc.) - useful to cut disk usage on huge videos.")
    parser.add_argument("--resize-width", type=int, default=None,
                         help="Optional: resize frames to this width before processing (speeds things up, shrinks output).")
    parser.add_argument("--format", default="png", choices=["png", "bmp", "tiff"],
                         help="Output image format. png/tiff are written with compression disabled; bmp is always raw.")
    args = parser.parse_args()

    process_video(
        args.video,
        args.output_dir,
        num_samples=args.num_samples,
        threshold=args.threshold,
        min_area=args.min_area,
        frame_stride=args.frame_stride,
        image_ext=args.format,
        resize_width=args.resize_width,
    )


if __name__ == "__main__":
    main()
