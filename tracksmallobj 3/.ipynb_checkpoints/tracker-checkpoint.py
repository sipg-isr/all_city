"""
Low Frame Rate Multi-Object Tracker — Small Object Edition
============================================================
Optimised for:
  - High-res images (1280px+)
  - Small bounding boxes (people/vehicles far from camera)
  - Large interframe displacements (low FPS)

Key changes vs v1:
  - SAHI tiled inference for small object detection recall
  - YOLOv8m (medium) by default
  - Center-distance replaces IoU in cost matrix (IoU is ~0 for tiny boxes)
  - Padded crop extraction before Re-ID resizing
  - Kalman process noise increased for small+fast objects
  - Appearance weight raised to 0.80
  - max_cost raised to 0.92
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from filterpy.kalman import KalmanFilter
import cv2


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class TrackerConfig:
    # Detection
    yolo_model: str = "yolov8m.pt"           # medium for better small-obj recall
    det_conf_thresh: float = 0.25            # lower threshold catches small objects
    det_iou_thresh: float = 0.40             # tighter NMS to avoid merging close smalls
    target_classes: List[int] = field(default_factory=lambda: [0, 2, 3, 5, 7])

    # SAHI tiled inference
    use_sahi: bool = True
    sahi_slice_size: int = 640               # tile size — matches YOLO training size
    sahi_overlap_ratio: float = 0.20         # 20% overlap avoids missing edge objects
    sahi_postprocess_type: str = "NMM"       # Non-Maximum Merging (better than NMS for tiles)
    sahi_postprocess_match_threshold: float = 0.50

    # Re-ID
    reid_input_size: Tuple[int, int] = (256, 128)
    reid_embed_dim: int = 2048
    reid_crop_pad_ratio: float = 0.25        # pad crop by 25% each side before resize

    # Cost matrix — small object tuning
    appearance_weight: float = 0.80          # appearance dominates; IoU useless for tiny boxes
    use_center_distance: bool = True         # replaces 1-IoU for position cost
    image_width: int = 1280
    image_height: int = 720
    max_cost: float = 0.92                   # looser gate since center-dist is less tight

    # Kalman — large displacement + small box uncertainty
    process_noise_pos: float = 800.0         # larger: small objects move fast relative to size
    process_noise_vel: float = 300.0
    measurement_noise: float = 5.0

    # Track management
    max_age: int = 30
    min_hits: int = 2


# ─────────────────────────────────────────────
# Kalman Filter (state: [x,y,w,h,vx,vy,vw,vh])
# ─────────────────────────────────────────────

class BBoxKalmanFilter:
    def __init__(self, bbox: np.ndarray, cfg: TrackerConfig):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        x, y, w, h = self._xyxy_to_xywh(bbox)

        self.kf.F = np.eye(8)
        self.kf.F[:4, 4:] = np.eye(4)

        self.kf.H = np.zeros((4, 8))
        self.kf.H[:4, :4] = np.eye(4)

        self.kf.Q = np.diag([
            cfg.process_noise_pos, cfg.process_noise_pos,
            cfg.process_noise_pos, cfg.process_noise_pos,
            cfg.process_noise_vel, cfg.process_noise_vel,
            cfg.process_noise_vel, cfg.process_noise_vel,
        ])
        self.kf.R = np.eye(4) * cfg.measurement_noise
        self.kf.x = np.array([x, y, w, h, 0, 0, 0, 0], dtype=float).reshape(-1, 1)
        self.kf.P = np.eye(8) * 1000.0

    def predict(self) -> np.ndarray:
        self.kf.predict()
        return self._xywh_to_xyxy(self.kf.x[:4].flatten())

    def update(self, bbox: np.ndarray):
        self.kf.update(self._xyxy_to_xywh(bbox).reshape(-1, 1))

    def get_state(self) -> np.ndarray:
        return self._xywh_to_xyxy(self.kf.x[:4].flatten())

    @staticmethod
    def _xyxy_to_xywh(b):
        return np.array([(b[0]+b[2])/2, (b[1]+b[3])/2, b[2]-b[0], b[3]-b[1]])

    @staticmethod
    def _xywh_to_xyxy(b):
        return np.array([b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2])


# ─────────────────────────────────────────────
# Track
# ─────────────────────────────────────────────

_TRACK_ID_COUNTER = 0


class Track:
    def __init__(self, bbox: np.ndarray, embed: np.ndarray,
                 class_id: int, cfg: TrackerConfig):
        global _TRACK_ID_COUNTER
        _TRACK_ID_COUNTER += 1
        self.id = _TRACK_ID_COUNTER
        self.class_id = class_id
        self.kf = BBoxKalmanFilter(bbox, cfg)
        self.embeds: List[np.ndarray] = [embed]
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.state = "tentative"

    @property
    def mean_embed(self) -> np.ndarray:
        return np.stack(self.embeds).mean(axis=0)

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox: np.ndarray, embed: np.ndarray, min_hits: int):
        self.kf.update(bbox)
        self.hits += 1
        self.time_since_update = 0
        self.embeds.append(embed)
        if len(self.embeds) > 5:
            self.embeds.pop(0)
        if self.hits >= min_hits:
            self.state = "confirmed"

    def get_bbox(self) -> np.ndarray:
        return self.kf.get_state()


# ─────────────────────────────────────────────
# Re-ID Model (ResNet50 backbone)
# ─────────────────────────────────────────────

class ReIDModel(nn.Module):
    def __init__(self, embed_dim: int = 2048, device: torch.device = torch.device("cpu")):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.embed_dim = embed_dim
        self.device = device
        self.to(device)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).view(x.size(0), -1)
        return nn.functional.normalize(feat, p=2, dim=1)


_REID_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def crop_with_padding(frame: np.ndarray, bbox: np.ndarray,
                      pad_ratio: float = 0.25) -> np.ndarray:
    """
    Pad the crop by pad_ratio of box dimensions on each side.
    Critical for small boxes: a 20x40 px person needs surrounding context
    to produce meaningful Re-ID embeddings after resizing to 256x128.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    pw, ph = int(bw * pad_ratio), int(bh * pad_ratio)
    x1c = max(0, x1 - pw)
    y1c = max(0, y1 - ph)
    x2c = min(w, x2 + pw)
    y2c = min(h, y2 + ph)
    crop = frame[y1c:y2c, x1c:x2c]
    return crop if crop.size > 0 else np.zeros((64, 32, 3), dtype=np.uint8)


def crop_and_embed(frame: np.ndarray, bboxes: np.ndarray,
                   model: ReIDModel, device: torch.device,
                   pad_ratio: float = 0.25) -> np.ndarray:
    if len(bboxes) == 0:
        return np.zeros((0, model.embed_dim))
    crops = [_REID_TRANSFORM(crop_with_padding(frame, b, pad_ratio)) for b in bboxes]
    batch = torch.stack(crops).to(device)
    return model(batch).cpu().numpy()


# ─────────────────────────────────────────────
# Cost matrix — center distance for small boxes
# ─────────────────────────────────────────────

def center_distance_cost(track_bboxes: np.ndarray, det_bboxes: np.ndarray,
                         img_w: int, img_h: int) -> np.ndarray:
    """
    Normalised Euclidean distance between box centers [0, ~1.41].
    Works when IoU is ~0 for all small, far-displaced objects.
    """
    def centers(b):
        return np.stack([(b[:, 0]+b[:, 2])/2 / img_w,
                         (b[:, 1]+b[:, 3])/2 / img_h], axis=1)
    return cdist(centers(track_bboxes), centers(det_bboxes), metric="euclidean")


def iou_matrix(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)))
    ax1, ay1, ax2, ay2 = np.split(bboxes_a, 4, axis=1)
    bx1, by1, bx2, by2 = np.split(bboxes_b, 4, axis=1)
    inter_w = np.maximum(0, np.minimum(ax2, bx2.T) - np.maximum(ax1, bx1.T))
    inter_h = np.maximum(0, np.minimum(ay2, by2.T) - np.maximum(ay1, by1.T))
    inter = inter_w * inter_h
    union = (ax2-ax1)*(ay2-ay1) + ((bx2-bx1)*(by2-by1)).T - inter
    return inter / np.maximum(union, 1e-6)


def build_cost_matrix(tracks: List[Track], det_bboxes: np.ndarray,
                      det_embeds: np.ndarray, cfg: TrackerConfig) -> np.ndarray:
    track_bboxes = np.array([t.get_bbox() for t in tracks])
    track_embeds = np.array([t.mean_embed for t in tracks])

    # Appearance: cosine distance [0, 2]  (clamp at 1 for safety)
    app_cost = np.clip(cdist(track_embeds, det_embeds, metric="cosine"), 0, 1)

    # Position cost
    if cfg.use_center_distance:
        pos_cost = center_distance_cost(
            track_bboxes, det_bboxes, cfg.image_width, cfg.image_height
        )
        # Normalise: 0.5 diagonal = 1.0; objects more than half-frame away = max cost
        pos_cost = np.clip(pos_cost / 0.5, 0.0, 1.0)
    else:
        pos_cost = 1.0 - iou_matrix(track_bboxes, det_bboxes)

    alpha = cfg.appearance_weight
    return alpha * app_cost + (1 - alpha) * pos_cost


# ─────────────────────────────────────────────
# SAHI-based detector wrapper
# ─────────────────────────────────────────────

class SAHIDetector:
    """
    Wraps SAHI sliced inference around YOLOv8.
    Falls back gracefully to full-frame inference if SAHI is not installed.

    SAHI slices a 1280px frame into overlapping 640px tiles, runs YOLO
    on each tile, then merges with NMM. This dramatically improves recall
    for small objects that span only a few dozen pixels on a 1280px image.
    """

    def __init__(self, cfg: TrackerConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self._init()

    def _init(self):
        if not self.cfg.use_sahi:
            self._init_plain()
            return
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            self._get_sliced = get_sliced_prediction
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=self.cfg.yolo_model,
                confidence_threshold=self.cfg.det_conf_thresh,
                device=str(self.device),
            )
            self._mode = "sahi"
            print("[Detector] SAHI tiled inference active")
        except ImportError:
            print("[Detector] SAHI not installed — using full-frame (install: pip install sahi)")
            self._init_plain()

    def _init_plain(self):
        from ultralytics import YOLO
        self.yolo = YOLO(self.cfg.yolo_model)
        self.yolo.to(self.device)
        self._mode = "plain"

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._mode == "sahi":
            return self._detect_sahi(frame)
        return self._detect_plain(frame)

    def _detect_sahi(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        result = self._get_sliced(
            frame,
            self.sahi_model,
            slice_height=self.cfg.sahi_slice_size,
            slice_width=self.cfg.sahi_slice_size,
            overlap_height_ratio=self.cfg.sahi_overlap_ratio,
            overlap_width_ratio=self.cfg.sahi_overlap_ratio,
            postprocess_type=self.cfg.sahi_postprocess_type,
            postprocess_match_threshold=self.cfg.sahi_postprocess_match_threshold,
            verbose=0,
        )
        bboxes, classes = [], []
        for pred in result.object_prediction_list:
            if pred.category.id not in self.cfg.target_classes:
                continue
            b = pred.bbox
            bboxes.append([b.minx, b.miny, b.maxx, b.maxy])
            classes.append(pred.category.id)
        if not bboxes:
            return np.zeros((0, 4)), np.zeros(0, dtype=int)
        return np.array(bboxes, dtype=float), np.array(classes, dtype=int)

    def _detect_plain(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        results = self.yolo(
            frame,
            conf=self.cfg.det_conf_thresh,
            iou=self.cfg.det_iou_thresh,
            imgsz=max(h, w),     # run at native resolution
            verbose=False,
        )[0]
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0, dtype=int)
        xyxy = boxes.xyxy.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int)
        mask = np.isin(cls, self.cfg.target_classes)
        return xyxy[mask], cls[mask]


# ─────────────────────────────────────────────
# Core Tracker
# ─────────────────────────────────────────────

class LowFPSTracker:
    """
    Multi-object tracker for low FPS + small objects on high-res images.

    Quick start:
        cfg = TrackerConfig(image_width=1280, image_height=720)
        tracker = LowFPSTracker(cfg)
        for frame in frames:
            results = tracker.update(frame)

    Each result dict: {"id", "bbox" [x1,y1,x2,y2], "class_id", "state", "hits", "age"}
    """

    def __init__(self, cfg: Optional[TrackerConfig] = None):
        self.cfg = cfg or TrackerConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = SAHIDetector(self.cfg, self.device)
        self.reid = ReIDModel(device=self.device)
        self.tracks: List[Track] = []
        self.frame_id = 0
        print(f"[Tracker] device={self.device}  model={self.cfg.yolo_model}  "
              f"sahi={'on' if self.cfg.use_sahi else 'off'}")

    def _assign(self, det_bboxes, det_embeds, det_classes):
        if len(self.tracks) == 0:
            return [], [], list(range(len(det_bboxes)))
        if len(det_bboxes) == 0:
            return [], list(range(len(self.tracks))), []

        cost = build_cost_matrix(self.tracks, det_bboxes, det_embeds, self.cfg)
        row_ind, col_ind = linear_sum_assignment(cost)

        matched, unmatched_t, unmatched_d = [], [], []
        matched_t_set, matched_d_set = set(), set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= self.cfg.max_cost:
                matched.append((r, c))
                matched_t_set.add(r)
                matched_d_set.add(c)

        unmatched_t = [i for i in range(len(self.tracks)) if i not in matched_t_set]
        unmatched_d = [i for i in range(len(det_bboxes)) if i not in matched_d_set]
        return matched, unmatched_t, unmatched_d

    def update(self, frame: np.ndarray) -> List[Dict]:
        self.frame_id += 1
        cfg = self.cfg
        cfg.image_height, cfg.image_width = frame.shape[:2]

        # 1. Detect (SAHI tiled)
        det_bboxes, det_classes = self.detector.detect(frame)

        # 2. Re-ID embeddings with padded crops
        det_embeds = crop_and_embed(
            frame, det_bboxes, self.reid, self.device, cfg.reid_crop_pad_ratio
        )

        # 3. Kalman predict
        for t in self.tracks:
            t.predict()

        # 4. Assign
        matched, unmatched_tracks, unmatched_dets = self._assign(
            det_bboxes, det_embeds, det_classes
        )

        # 5. Update matched tracks
        for tr_idx, det_idx in matched:
            self.tracks[tr_idx].update(
                det_bboxes[det_idx], det_embeds[det_idx], cfg.min_hits
            )

        # 6. Mark unmatched as lost
        for tr_idx in unmatched_tracks:
            self.tracks[tr_idx].state = "lost"

        # 7. Spawn new tracks
        for det_idx in unmatched_dets:
            self.tracks.append(
                Track(det_bboxes[det_idx], det_embeds[det_idx],
                      det_classes[det_idx], cfg)
            )

        # 8. Prune dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= cfg.max_age]

        # 9. Return active tracks
        return [
            {
                "id":       t.id,
                "bbox":     t.get_bbox().tolist(),
                "class_id": t.class_id,
                "state":    t.state,
                "hits":     t.hits,
                "age":      t.age,
            }
            for t in self.tracks if t.state in ("confirmed", "tentative")
        ]
