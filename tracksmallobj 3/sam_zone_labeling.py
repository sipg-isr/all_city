"""
sam_zone_labeling.py

Label static zones (road, sidewalk, crosswalk, ...) on a SINGLE reference
image - typically the median background produced by
median_background_segmentation.py - using Segment Anything (SAM) point
prompts.

Since a surveillance camera doesn't move, this only needs to be run ONCE
per camera. The resulting label map (a single-channel image where each
pixel's value is the zone's class ID) can then be reused downstream: for
any detected object, check its footpoint against this map to know if it's
on the road or the sidewalk, and flag violations of rules like "cars don't
belong on the sidewalk".

SAM itself is class-agnostic - it does not know "road" from "sidewalk" on
its own. What it's good at is turning a couple of clicks into a precise,
boundary-accurate mask of the region you clicked on. You do the labeling
(one click on the road, one on the sidewalk, etc.); SAM does the precise
pixel-level boundary tracing.

--------------------------------------------------------------------------
Setup
--------------------------------------------------------------------------
Two backend options - pick one:

A) Official SAM (more accurate, heavier):
    pip install torch torchvision opencv-python numpy
    pip install git+https://github.com/facebookresearch/segment-anything.git
    Download a checkpoint (vit_b is the smallest official one, ~375MB):
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    Run with: --backend sam --model-type vit_b --checkpoint sam_vit_b_01ec64.pth

B) MobileSAM (much lighter, ~40MB, near drop-in, slightly less precise on
   fine boundaries):
    pip install torch torchvision opencv-python numpy timm
    pip install git+https://github.com/ChaoningZhang/MobileSAM.git
    Checkpoint is bundled in the repo at MobileSAM/weights/mobile_sam.pt
    Run with: --backend mobile_sam --model-type vit_t --checkpoint mobile_sam.pt

Since this runs once per camera on a single still image (not per video
frame), the heavier official SAM is very usable here even on CPU - use
MobileSAM if you want faster iteration while you're tuning your clicks.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
    python sam_zone_labeling.py background.png \
        --backend sam --model-type vit_b --checkpoint sam_vit_b_01ec64.pth \
        --classes road sidewalk crosswalk --output-dir zones

Controls (interactive window):
    1 / 2 / 3 ...   switch active class (order = --classes order)
    left click      add a positive point for the current class
    right click     undo the last point for the current class
    n               compute / refresh the mask for the current class
    s               save the final label map and exit
    q               quit without saving
"""

import argparse
import os
import numpy as np
import cv2

CLASS_COLORS = [
    (0, 0, 0),        # 0 = unlabeled
    (80, 80, 80),     # 1 = first class (e.g. road)
    (255, 140, 0),    # 2 = second class (e.g. sidewalk)
    (0, 220, 220),    # 3 = third class (e.g. crosswalk)
    (180, 60, 220),   # 4 = spare
    (60, 200, 60),    # 5 = spare
]


def load_predictor(backend, checkpoint, model_type, device):
    if backend == "sam":
        from segment_anything import sam_model_registry, SamPredictor
    else:
        from mobile_sam import sam_model_registry, SamPredictor
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)


class ZoneLabeler:
    """
    Holds click state per class and turns clicks into SAM-predicted masks.
    The interactive run() loop needs a display; compute_mask()/finalize()
    can also be driven programmatically (e.g. from a notebook) if you'd
    rather hardcode pixel coordinates than click them.
    """

    def __init__(self, image_bgr, predictor, class_names):
        self.image = image_bgr
        self.predictor = predictor
        self.class_names = class_names
        self.points = {name: [] for name in class_names}
        self.active = class_names[0]
        self.masks = {}  # name -> bool mask (H, W)
        self.label_map = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

        self.predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    def compute_mask(self, name):
        pts = self.points[name]
        if not pts:
            print(f"No points for '{name}' yet - click on the image first.")
            return
        point_coords = np.array(pts)
        point_labels = np.ones(len(pts))
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        best = masks[int(np.argmax(scores))]
        self.masks[name] = best.astype(bool)
        print(f"'{name}' mask updated ({best.sum()} px, score {scores.max():.3f})")

    def finalize(self):
        # Later classes in --classes order win where masks overlap.
        for i, name in enumerate(self.class_names, start=1):
            if name in self.masks:
                self.label_map[self.masks[name]] = i
        return self.label_map

    def _redraw(self):
        vis = self.image.copy()
        overlay = np.zeros_like(vis)
        for i, name in enumerate(self.class_names, start=1):
            if name in self.masks:
                color = CLASS_COLORS[i] if i < len(CLASS_COLORS) else (255, 255, 255)
                overlay[self.masks[name]] = color
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

        for name in self.class_names:
            for (x, y) in self.points[name]:
                cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)

        cv2.putText(vis, f"active: {self.active}  [1-{len(self.class_names)} switch, "
                          f"click=add point, n=compute, s=save, q=quit]",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("SAM zone labeling", vis)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points[self.active].append((x, y))
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points[self.active]:
                self.points[self.active].pop()
                self._redraw()

    def run(self):
        cv2.namedWindow("SAM zone labeling")
        cv2.setMouseCallback("SAM zone labeling", self._on_mouse)
        self._redraw()
        print("Controls: [1-9] switch class | left-click add point | "
              "right-click undo | n compute mask | s save+quit | q quit")
        while True:
            key = cv2.waitKey(20) & 0xFF
            if ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                if idx < len(self.class_names):
                    self.active = self.class_names[idx]
                    self._redraw()
            elif key == ord('n'):
                self.compute_mask(self.active)
                self._redraw()
            elif key == ord('s'):
                cv2.destroyAllWindows()
                return self.finalize()
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None


def save_label_map(label_map, out_dir, class_names):
    os.makedirs(out_dir, exist_ok=True)

    raw_path = os.path.join(out_dir, "zone_labels.png")
    cv2.imwrite(raw_path, label_map, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    vis = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for i in range(len(class_names) + 1):
        color = CLASS_COLORS[i] if i < len(CLASS_COLORS) else (255, 255, 255)
        vis[label_map == i] = color
    vis_path = os.path.join(out_dir, "zone_labels_preview.png")
    cv2.imwrite(vis_path, vis, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    legend_path = os.path.join(out_dir, "zone_legend.txt")
    with open(legend_path, "w") as f:
        f.write("0 = unlabeled\n")
        for i, name in enumerate(class_names, start=1):
            f.write(f"{i} = {name}\n")

    print(f"Saved:\n  {raw_path}  (use this one programmatically)\n"
          f"  {vis_path}  (for a quick visual check)\n  {legend_path}")


def main():
    parser = argparse.ArgumentParser(description="Label static zones (road/sidewalk/...) using SAM point prompts.")
    parser.add_argument("background", help="Path to the background reference image.")
    parser.add_argument("--backend", default="sam", choices=["sam", "mobile_sam"],
                         help="'sam' = official Segment Anything, 'mobile_sam' = lightweight variant.")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--model-type", default="vit_b",
                         help="Must match the checkpoint. Official SAM: vit_b/vit_l/vit_h. MobileSAM: vit_t.")
    parser.add_argument("--classes", nargs="+", default=["road", "sidewalk", "crosswalk"],
                         help="Class names in priority order (later ones win where clicks overlap).")
    parser.add_argument("--output-dir", default="zones", help="Where to save the results.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    image = cv2.imread(args.background)
    if image is None:
        raise IOError(f"Could not read {args.background}")

    predictor = load_predictor(args.backend, args.checkpoint, args.model_type, args.device)
    labeler = ZoneLabeler(image, predictor, args.classes)
    label_map = labeler.run()

    if label_map is not None:
        save_label_map(label_map, args.output_dir, args.classes)
    else:
        print("Cancelled - nothing saved.")


if __name__ == "__main__":
    main()