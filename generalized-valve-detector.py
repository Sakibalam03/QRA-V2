"""
Generalized Valve Detector
--------------------------
Detects manual and actuated valves in P&ID patches using YOLOE SAVPE
(visual prompt mode). Reference images are full P&ID patches with bboxes
drawn around complete valve symbols in context (black outlines + colored
body + adjacent pipe) — not tight colored-blob crops.

Usage:
    python generalized-valve-detector.py <patch_dir> <output_dir>

To add more reference examples (improves recall):
    python image-prompt-draw-box.py <any_patch_with_a_valve.png>
    → copy the printed bbox into REFS below as a new entry with the right cls
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "yoloe-11s-seg.pt"
CONF  = 0.02   # SAVPE confidence on P&ID line art sits at 0.01–0.06

# Each entry is one reference example. Add more entries per class to improve
# recall — more visual diversity = more valve subtypes recognised.
# Run image-prompt-draw-box.py to get new (image, bbox) pairs.
REFS = [
    # ── manual valves (class 0) ──────────────────────────────────────────
    {"cls": 0, "image": "refs/manual_ref.png",   "bbox": [203, 443, 234, 494]},
    # {"cls": 0, "image": "refs/manual_ref2.png",  "bbox": [x1, y1, x2, y2]},  # gate valve variant
    # {"cls": 0, "image": "refs/manual_ref3.png",  "bbox": [x1, y1, x2, y2]},  # globe valve variant

    # ── actuated valves (class 1) ────────────────────────────────────────
    {"cls": 1, "image": "refs/actuated_ref.png", "bbox": [1, 276, 114, 368]},
    # {"cls": 1, "image": "refs/actuated_ref2.png","bbox": [x1, y1, x2, y2]},
]

CLASS_NAMES  = {0: "manual", 1: "actuated"}
CLASS_COLORS = {0: (34, 197, 94), 1: (30, 144, 255)}   # green / blue (BGR)

# Color gate
SAT_THRESH  = 50
COLOR_RATIO = 0.08   # full-bbox color ratio

# Bbox shape/size gate
MIN_SIDE = 15    # px
MAX_SIDE = 220   # px
MAX_AR   = 4.0   # long/short ratio

# NMS — suppresses duplicate boxes that land on the same valve symbol
IOU_THRESH = 0.35

# ---------------------------------------------------------------------------
# Reference builder
# ---------------------------------------------------------------------------

def build_reference():
    """Stack all reference patches side-by-side into one SAVPE canvas."""
    images, bboxes, cls_ids = [], [], []
    x_offset = 0
    for entry in REFS:
        img = cv2.imread(entry["image"])
        if img is None:
            raise FileNotFoundError(f"Reference image not found: {entry['image']}")
        x1, y1, x2, y2 = entry["bbox"]
        images.append(img)
        bboxes.append([x1 + x_offset, y1, x2 + x_offset, y2])
        cls_ids.append(entry["cls"])
        x_offset += img.shape[1]

    max_h  = max(i.shape[0] for i in images)
    canvas = np.full((max_h, x_offset, 3), 255, dtype=np.uint8)
    x = 0
    for img in images:
        h, w = img.shape[:2]
        canvas[:h, x:x + w] = img
        x += w

    return canvas, np.array(bboxes, dtype=np.float32), np.array(cls_ids, dtype=np.int32)

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _sat_ratio(image, x1, y1, x2, y2):
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    sat = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1]
    return float((sat > SAT_THRESH).mean())


def is_colored(image, x1, y1, x2, y2):
    """Full bbox must have enough color."""
    return _sat_ratio(image, x1, y1, x2, y2) >= COLOR_RATIO


def center_is_colored(image, x1, y1, x2, y2):
    """Inner 50% of bbox must also have color — rejects label boxes that only clip a pipe edge."""
    pad_x = (x2 - x1) // 4
    pad_y = (y2 - y1) // 4
    cx1, cy1 = x1 + pad_x, y1 + pad_y
    cx2, cy2 = x2 - pad_x, y2 - pad_y
    if cx2 <= cx1 or cy2 <= cy1:
        return True  # bbox too small to sub-crop — defer to outer check
    return _sat_ratio(image, cx1, cy1, cx2, cy2) >= COLOR_RATIO


def passes_shape(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    if w < MIN_SIDE or h < MIN_SIDE:
        return False
    if w > MAX_SIDE or h > MAX_SIDE:
        return False
    ar = max(w, h) / max(min(w, h), 1)
    return ar <= MAX_AR


def nms(detections):
    """detections: list of [x1,y1,x2,y2,cls_id,conf]. Returns pruned list."""
    if len(detections) <= 1:
        return detections
    detections.sort(key=lambda d: d[5], reverse=True)
    keep = []
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        suppress = False
        for k in keep:
            kx1, ky1, kx2, ky2 = k[:4]
            iw = max(0, min(x2, kx2) - max(x1, kx1))
            ih = max(0, min(y2, ky2) - max(y1, ky1))
            inter = iw * ih
            union = (x2-x1)*(y2-y1) + (kx2-kx1)*(ky2-ky1) - inter
            if union > 0 and inter / union > IOU_THRESH:
                suppress = True
                break
        if not suppress:
            keep.append(det)
    return keep

# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def draw_boxes(image, result):
    out = image.copy()
    raw = 0 if result.boxes is None else len(result.boxes)

    if raw == 0:
        return out, 0, 0

    # Collect candidates that pass shape + color checks
    candidates = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if not passes_shape(x1, y1, x2, y2):
            continue
        if not is_colored(image, x1, y1, x2, y2):
            continue
        if not center_is_colored(image, x1, y1, x2, y2):
            continue
        candidates.append([x1, y1, x2, y2, int(box.cls[0]), float(box.conf[0])])

    # NMS across all classes
    kept_boxes = nms(candidates)

    for x1, y1, x2, y2, cls_id, conf in kept_boxes:
        color = CLASS_COLORS.get(cls_id, (180, 180, 180))
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return out, raw, len(kept_boxes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(patch_dir, output_dir):
    patch_dir  = Path(patch_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patches = sorted(patch_dir.glob("*.png")) + sorted(patch_dir.glob("*.jpg"))
    if not patches:
        print(f"No images found in {patch_dir}")
        return

    print(f"Patches      : {len(patches)}")
    print(f"References   : {len(REFS)} ({sum(1 for r in REFS if r['cls']==0)} manual, "
          f"{sum(1 for r in REFS if r['cls']==1)} actuated)")

    canvas, bboxes, cls_ids = build_reference()
    visual_prompts = dict(bboxes=bboxes, cls=cls_ids)

    model = YOLOE(MODEL)

    # First call bakes VPE into model.model; cache predictor to avoid 64× banner.
    first = cv2.imread(str(patches[0]))
    model.predict(
        first,
        refer_image=canvas,
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        conf=CONF,
        verbose=False,
    )
    predictor = model.predictor

    saved = 0
    for path in patches:
        img = cv2.imread(str(path))
        if img is None:
            continue

        results              = predictor(source=img, stream=False)
        annotated, raw, kept = draw_boxes(img, results[0])

        if kept > 0:
            cv2.imwrite(str(output_dir / path.name), annotated)
            saved += 1
        status = f"raw={raw} kept={kept}"
        if raw > kept:
            status += f" filtered={raw - kept}"
        print(f"  {path.name}: {status}")

    print(f"\nDone — {saved}/{len(patches)} patches saved → {output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("patch_dir",  help="Folder of P&ID patch images")
    ap.add_argument("output_dir", help="Folder to write annotated output")
    args = ap.parse_args()
    run(args.patch_dir, args.output_dir)
