"""
P&ID Valve Detector — Geometry-Based (no ML, no training)
----------------------------------------------------------
Algorithm:
  1. Extract colored pipe mask via HSV thresholding.
  2. Dilate pipe mask into an envelope that covers valve symbols
     embedded in the pipe (valve symbols break the colored region).
  3. Find dark (black) connected components WITHIN the envelope —
     these are the geometric valve symbols.
  4. Filter by area, aspect ratio, size, and proximity to pipe.
  5. NMS to deduplicate overlapping hits on the same symbol.

Works for all valve types and orientations with zero references.
No YOLOE / no training required for detection.

Classification (manual vs actuated) can be layered on top once
detection is reliable — see CLASSIFY section at bottom.

Usage:
    python generalized-valve-detector.py <patch_dir> <output_dir>
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Pipe color ranges  (HSV — tweak lo/hi if your drawing uses different hues)
# ---------------------------------------------------------------------------

PIPE_HSV = [
    ((15,  80,  80), (38, 255, 255)),   # yellow  H ≈ 20–35
    (( 5, 100,  80), (15, 255, 255)),   # orange  H ≈ 8–15
]

# Dilation applied to the pipe mask to create an envelope that spans valve
# symbols which interrupt the colored band.  Increase if large actuated-valve
# actuator boxes are being missed.
PIPE_DILATE = 40   # px

# ---------------------------------------------------------------------------
# Dark-component (valve symbol) geometry filters
# ---------------------------------------------------------------------------

DARK_THRESH     = 140   # grayscale below this = "dark symbol pixel"
MIN_AREA        = 150   # px²  — filters single text chars & noise
MAX_AREA        = 9000  # px²  — filters large connected regions
MIN_SIDE        = 15    # px   — smaller than this = probably a character
MAX_SIDE        = 160   # px   — larger than this = not a single valve symbol
MAX_AR          = 4.0   # long/short — rejects elongated pipe borders & text lines
MIN_PIPE_PX     = 80    # colored pixels required within a 20 px pad around bbox

# Pipe-crossing gate: valve symbols SPAN the pipe (pipe enters from one side
# and exits the opposite).  Instrument circles are connected from ONE side only.
# We look CROSS_PAD px outside the bbox on each of the four sides and require
# that opposite pairs (left+right OR top+bottom) both see colored pipe pixels.
CROSS_PAD    = 25   # px outside bbox to sample for crossing pipe
MIN_CROSS_PX = 6    # colored pixels required on EACH side of the crossing pair

# ---------------------------------------------------------------------------
# NMS & display
# ---------------------------------------------------------------------------

IOU_THRESH  = 0.35
VALVE_COLOR = (34, 197, 94)   # green  — undifferentiated "valve" label

# ---------------------------------------------------------------------------
# Pipe mask helpers
# ---------------------------------------------------------------------------

def get_pipe_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for lo, hi in PIPE_HSV:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return mask


def get_pipe_envelope(pipe_mask: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * PIPE_DILATE + 1, 2 * PIPE_DILATE + 1)
    )
    return cv2.dilate(pipe_mask, k)

# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

def detect_valve_symbols(image: np.ndarray) -> list:
    """
    Return list of (x1, y1, x2, y2) bboxes for dark geometric symbols
    that sit on the colored pipe — i.e., valve symbols.
    """
    pipe_mask = get_pipe_mask(image)
    if not pipe_mask.any():
        return []

    envelope = get_pipe_envelope(pipe_mask)

    # Dark pixels inside the pipe envelope
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_in_env = np.zeros_like(gray)
    dark_in_env[(gray < DARK_THRESH) & (envelope > 0)] = 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark_in_env, connectivity=8
    )

    ih, iw = image.shape[:2]
    candidates = []

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (MIN_AREA <= area <= MAX_AREA):
            continue

        x  = int(stats[i, cv2.CC_STAT_LEFT])
        y  = int(stats[i, cv2.CC_STAT_TOP])
        w  = int(stats[i, cv2.CC_STAT_WIDTH])
        h  = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Size gate
        if w < MIN_SIDE or h < MIN_SIDE or w > MAX_SIDE or h > MAX_SIDE:
            continue

        # Aspect-ratio gate (rejects elongated pipe borders)
        if max(w, h) / max(min(w, h), 1) > MAX_AR:
            continue

        # Proximity gate: colored pipe must exist near the component
        pad  = 20
        rx1  = max(0,  x - pad);  ry1 = max(0,  y - pad)
        rx2  = min(iw, x + w + pad); ry2 = min(ih, y + h + pad)
        pipe_near = int(pipe_mask[ry1:ry2, rx1:rx2].sum()) // 255
        if pipe_near < MIN_PIPE_PX:
            continue

        # Pipe-crossing gate: pipe must enter from TWO opposite sides.
        # Valves span the pipe; instrument circles connect from one side only.
        lp = int(pipe_mask[y:y+h, max(0, x-CROSS_PAD):x          ].sum()) // 255
        rp = int(pipe_mask[y:y+h, x+w:min(iw, x+w+CROSS_PAD)     ].sum()) // 255
        tp = int(pipe_mask[max(0, y-CROSS_PAD):y,    x:x+w        ].sum()) // 255
        bp = int(pipe_mask[y+h:min(ih, y+h+CROSS_PAD), x:x+w      ].sum()) // 255

        h_cross = lp >= MIN_CROSS_PX and rp >= MIN_CROSS_PX
        v_cross = tp >= MIN_CROSS_PX and bp >= MIN_CROSS_PX
        if not (h_cross or v_cross):
            continue

        candidates.append((x, y, x + w, y + h))

    return candidates

# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def nms(boxes: list) -> list:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    keep = []
    for b in boxes:
        x1, y1, x2, y2 = b
        ok = True
        for k in keep:
            kx1, ky1, kx2, ky2 = k
            iw_ = max(0, min(x2, kx2) - max(x1, kx1))
            ih_ = max(0, min(y2, ky2) - max(y1, ky1))
            inter = iw_ * ih_
            union = (x2-x1)*(y2-y1) + (kx2-kx1)*(ky2-ky1) - inter
            if union > 0 and inter / union > IOU_THRESH:
                ok = False
                break
        if ok:
            keep.append(b)
    return keep

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

    print(f"Patches: {len(patches)}")
    saved = 0

    for path in patches:
        img = cv2.imread(str(path))
        if img is None:
            continue

        boxes = nms(detect_valve_symbols(img))

        out = img.copy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), VALVE_COLOR, 2)
            label = "valve"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 4, y1), VALVE_COLOR, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        if boxes:
            cv2.imwrite(str(output_dir / path.name), out)
            saved += 1
        print(f"  {path.name}: {len(boxes)} detections")

    print(f"\nDone — {saved}/{len(patches)} patches saved → {output_dir}")

# ---------------------------------------------------------------------------
# CLASSIFY — layer this on top once detection is reliable
# ---------------------------------------------------------------------------
# To distinguish manual vs actuated without training:
#
#   Option A — Geometry:
#     Actuated valves have an actuator box above/below the valve body.
#     They are typically larger vertically (h > 60px in a 640px patch).
#     A simple height threshold separates most cases:
#       cls = "actuated" if (y2-y1) > ACTUATED_MIN_HEIGHT else "manual"
#
#   Option B — YOLOE SAVPE as classifier (not detector):
#     For each detected bbox, extract a 96×96 crop and run SAVPE
#     with 1 reference per class (clean embedding, small search space).
#     SAVPE on a known candidate is a binary classification problem —
#     much more reliable than open-field detection.
#
#   Option C — Template matching:
#     Maintain small 30×30 templates for each valve subtype.
#     Match against each detected bbox crop.
#     cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("patch_dir",  help="Folder of P&ID patch images")
    ap.add_argument("output_dir", help="Folder to write annotated output")
    args = ap.parse_args()
    run(args.patch_dir, args.output_dir)
