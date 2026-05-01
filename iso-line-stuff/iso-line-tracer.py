"""
P&ID ISO Line Tracer — Skeleton-Based with Corridor Masking
-------------------------------------------------------------
For each pipe color:
  1. HSV mask → morphological cleanup
  2. Skeletonize to 1-px centerline
     (uses scikit-image → cv2.ximgproc → built-in fallback, in that order)
  3. Connected components on skeleton → individual pipe run segments
  4. Draw thickened skeleton over the original image
  5. Report segment count and pixel length per color

Corridor masking (--mask-corridor):
  After tracing all requested colors, builds a padded corridor around each
  detected pipe run. Pure-white background pixels outside the corridor are
  replaced with white (255,255,255). All non-white P&ID content (line art,
  annotations, symbols) is always preserved regardless of corridor membership.

  Corridor construction strategy per segment orientation:
    • Predominantly horizontal segment → axis-aligned horizontal band
      (row_min - PAD : row_max + PAD, col_min - PAD : col_max + PAD)
    • Predominantly vertical segment   → axis-aligned vertical band
      (same logic, columns drive the band)
    • Diagonal / curved segment        → morphological dilation of color mask
  The three strategies are unioned into a single corridor mask.

Usage:
    python iso-line-tracer.py <image_path> [-o output] [-c yellow orange ...]
    python iso-line-tracer.py patch.png --boxes
    python iso-line-tracer.py patch.png --mask-corridor
    python iso-line-tracer.py patch.png --mask-corridor --pad 20
    python iso-line-tracer.py patch.png --mask-corridor --pad 15 --white-thresh 250
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import Optional

# ---------------------------------------------------------------------------
# Skeletonize — best available backend, automatic fallback
# ---------------------------------------------------------------------------

def _skeletonize_morphological(mask: np.ndarray) -> np.ndarray:
    """Classic morphological skeleton — no external dependencies required."""
    skel = np.zeros_like(mask)
    k    = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img  = (mask > 0).astype(np.uint8) * 255
    while True:
        eroded = cv2.erode(img, k)
        skel  |= cv2.subtract(img, cv2.dilate(eroded, k))
        img    = eroded
        if not img.any():
            break
    return skel


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Return 1-px skeleton of a binary mask."""
    try:
        from skimage.morphology import skeletonize as sk_skel
        return (sk_skel(mask > 0) * 255).astype(np.uint8)
    except ImportError:
        pass
    try:
        return cv2.ximgproc.thinning((mask > 0).astype(np.uint8) * 255)
    except AttributeError:
        pass
    return _skeletonize_morphological(mask)


# ---------------------------------------------------------------------------
# Pipe color configuration
# ---------------------------------------------------------------------------

PIPE_COLORS: dict[str, dict] = {
    'yellow': {
        'ranges': [((15,  80,  80), (38, 255, 255))],
        'bgr':    (0, 220, 255),
    },
    'orange': {
        'ranges': [(( 5, 100,  80), (13, 255, 255))],
        'bgr':    (0, 140, 255),
    },
    'green': {
        'ranges': [((40, 100, 100), (80, 255, 255))],
        'bgr':    (30, 220,  30),
    },
    'red': {
        'ranges': [(( 0, 100, 100), (10, 255, 255)),
                   ((165, 100, 100), (180, 255, 255))],
        'bgr':    (30,  30, 220),
    },
    'blue': {
        'ranges': [((100, 100, 100), (130, 255, 255))],
        'bgr':    (230,  80,   0),
    },
    'purple': {
        'ranges': [((130, 100, 100), (160, 255, 255))],
        'bgr':    (200,   0, 160),
    },
}

_CLEANUP_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_DRAW_K    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
MIN_SEG_PX = 25   # skeleton pixels below this → stray noise, ignored

# Aspect ratio threshold to decide vertical vs horizontal vs diagonal/curved.
# A segment is "predominantly vertical" if its bounding-box height/width > ASPECT_THRESH.
# It is "predominantly horizontal" if width/height > ASPECT_THRESH.
# Otherwise it is treated as diagonal/curved and gets a dilated corridor.
ASPECT_THRESH = 2.5


# ---------------------------------------------------------------------------
# Per-color tracer
# ---------------------------------------------------------------------------

def get_color_mask(
    hsv: np.ndarray,
    ranges: list,
    exclude: Optional[np.ndarray] = None,
    apply_open: bool = True,
    close_iters: int = 2,
) -> np.ndarray:
    """
    Build a cleaned binary color mask from HSV ranges.

    Parameters
    ----------
    apply_open   : if True, apply MORPH_OPEN after MORPH_CLOSE to remove noise.
                   Set False for dashed/dotted lines where OPEN destroys dashes.
    close_iters  : number of MORPH_CLOSE iterations. Increase to bridge larger
                   gaps between dashes (default 2; use 3 for wide-gap dashes).
    """
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    if exclude is not None:
        mask &= cv2.bitwise_not(exclude)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLEANUP_K, iterations=close_iters)
    if apply_open:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _CLEANUP_K, iterations=1)
    return mask


def trace_color(
    image: np.ndarray,
    cfg: dict,
    exclude: Optional[np.ndarray] = None,
) -> tuple[Optional[np.ndarray], list[dict], np.ndarray, np.ndarray]:
    """
    Trace one pipe color in an image.

    Two masks are returned with different levels of cleanup:
      strict_mask  : CLOSE + OPEN applied — minimal noise, used for skeleton
                     drawing. Dash gaps are NOT bridged; dashes appear as dashes.
      extent_mask  : CLOSE only (more iterations) — gap-bridged, used for
                     corridor building so dashed lines produce a continuous
                     corridor rather than isolated per-dash blobs.

    Parameters
    ----------
    exclude : optional binary mask of pixels already claimed by a prior color.

    Returns
    -------
    skel        : np.ndarray | None  — 1-px skeleton (H×W uint8)
    segments    : list of dict       — [{px, bbox:(x1,y1,x2,y2)}, ...]
    strict_mask : np.ndarray         — cleaned mask with OPEN (for exclusion tracking)
    extent_mask : np.ndarray         — gap-bridged mask without OPEN (for corridor)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    strict_mask = get_color_mask(
        hsv, cfg['ranges'], exclude,
        apply_open=True,
        close_iters=2,
    )
    extent_mask = get_color_mask(
        hsv, cfg['ranges'], exclude,
        apply_open=False,
        close_iters=3,   # extra iteration bridges wider dash gaps
    )

    if not strict_mask.any() and not extent_mask.any():
        return None, [], strict_mask, extent_mask

    # Use strict_mask for skeletonization (less noise).
    # Fall back to extent_mask if strict kills all pixels (e.g. very thin dashes).
    skel_src = strict_mask if strict_mask.any() else extent_mask
    skel = skeletonize(skel_src)

    n, _, stats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
    segments: list[dict] = []
    for i in range(1, n):
        px = int(stats[i, cv2.CC_STAT_AREA])
        if px < MIN_SEG_PX:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        segments.append({'px': px, 'bbox': (x, y, x + w, y + h)})

    # If strict_mask skeleton produced no qualifying segments (all dashes too small),
    # try recovering segments from extent_mask skeleton instead.
    if not segments and extent_mask.any():
        skel = skeletonize(extent_mask)
        n, _, stats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
        for i in range(1, n):
            px = int(stats[i, cv2.CC_STAT_AREA])
            if px < MIN_SEG_PX:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            segments.append({'px': px, 'bbox': (x, y, x + w, y + h)})

    return skel, segments, strict_mask, extent_mask


# ---------------------------------------------------------------------------
# Corridor mask builder
# ---------------------------------------------------------------------------

def _segment_orientation(bbox: tuple[int, int, int, int]) -> str:
    """
    Classify a segment's dominant orientation from its bounding box.

    Returns one of: 'vertical', 'horizontal', 'diagonal'
    """
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    if h / w >= ASPECT_THRESH:
        return 'vertical'
    if w / h >= ASPECT_THRESH:
        return 'horizontal'
    return 'diagonal'


def build_corridor_mask(
    color_mask: np.ndarray,
    segments: list[dict],
    image_shape: tuple[int, int],
    pad: int,
) -> np.ndarray:
    """
    Build a binary corridor mask for a set of pipe segments.

    Strategy per segment orientation:
      horizontal → axis-aligned row-band (PAD rows above/below bbox)
      vertical   → axis-aligned col-band (PAD cols left/right of bbox)
      diagonal   → morphological dilation of the underlying color mask

    For horizontal and vertical segments the padding is applied in BOTH
    the perpendicular AND the along-pipe directions so the corridor has
    clean rectangular caps at the ends of each run.

    Parameters
    ----------
    color_mask   : cleaned binary color mask (uint8, 0/255)
    segments     : segment list from trace_color
    image_shape  : (H, W)
    pad          : half-width of the corridor in pixels

    Returns
    -------
    corridor : np.ndarray (H×W, bool)  True inside the corridor
    """
    H, W = image_shape
    corridor = np.zeros((H, W), dtype=bool)

    # Dilation kernel shared by all diagonal segments
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    need_dilation = False

    for seg in segments:
        x1, y1, x2, y2 = seg['bbox']
        orientation = _segment_orientation(seg['bbox'])

        # Clamp padded coordinates to image bounds
        r0 = max(0, y1 - pad)
        r1 = min(H - 1, y2 + pad)
        c0 = max(0, x1 - pad)
        c1 = min(W - 1, x2 + pad)

        if orientation in ('horizontal', 'vertical'):
            # Axis-aligned rectangular corridor covering the full bbox + PAD on all sides.
            # For a horizontal run this is a wide row-band.
            # For a vertical run this is a tall column-band.
            # In both cases the same padded rect is correct because we pad all 4 edges.
            corridor[r0:r1 + 1, c0:c1 + 1] = True
        else:
            # Curved / diagonal segment: circular dilation gives a better-shaped corridor
            # than a rectangle that might include large empty corners.
            need_dilation = True

    if need_dilation:
        # Dilate the entire color mask once and union into corridor.
        # This is cheaper than dilating per-segment and equally correct.
        dilated = cv2.dilate(color_mask, dil_k)
        corridor |= dilated.astype(bool)

    return corridor


# ---------------------------------------------------------------------------
# Corridor masker — apply to output image
# ---------------------------------------------------------------------------

def apply_corridor_mask(
    image: np.ndarray,
    corridor: np.ndarray,
    white_thresh: int = 252,
) -> np.ndarray:
    """
    White-out every pixel that falls outside the corridor.

    Two masking modes depending on white_thresh:

      white_thresh=0 (strict geometric mode, DEFAULT recommended):
        Every pixel outside the corridor is set to pure white, regardless of
        its original value. This matches the intended use-case: isolate the
        pipe run corridor completely. Annotations, symbols, and line art that
        fall outside the corridor are also erased. This is what a human editor
        would do with a paint-bucket fill on the background region.

      white_thresh=1..255 (selective mode, legacy):
        Only pixels where ALL channels >= white_thresh AND outside the corridor
        are whited out. Pixels with at least one channel below white_thresh
        (dark P&ID content) are preserved unconditionally.
        Set white_thresh=252 to preserve near-white P&ID annotations.

    Parameters
    ----------
    image        : BGR image (original, unmodified)
    corridor     : bool mask (H×W), True inside the padded corridor
    white_thresh : 0  → strict geometric erase (white out EVERYTHING outside
                          corridor, including dark P&ID line art)
                   1-255 → selective: only erase pixels above this brightness
                           threshold outside the corridor
    Returns
    -------
    out : copy of image with out-of-corridor pixels set to 255
    """
    out = image.copy()

    if white_thresh == 0:
        # Strict geometric mode: white out ALL pixels outside corridor
        out[~corridor] = 255
    else:
        # Selective mode: only white out near-white background pixels outside corridor
        is_background  = np.all(image >= white_thresh, axis=2)
        out[is_background & ~corridor] = 255

    return out


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    image_path: str,
    output_path: str,
    colors: list[str],
    draw_boxes: bool,
    mask_corridor: bool,
    pad: int,
    white_thresh: int = 0,
) -> None:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    out = img.copy()
    H, W = img.shape[:2]
    print(f"Image : {image_path}  ({W}×{H} px)\n")

    # Accumulate claimed pixels so later colors never re-detect what earlier
    # colors already own (prevents bleed-through artifacts between close hues).
    claimed = np.zeros((H, W), dtype=np.uint8)

    # Combined corridor across all colors (used when --mask-corridor is set)
    combined_corridor = np.zeros((H, W), dtype=bool)

    grand_total = 0

    for color_name in colors:
        cfg = PIPE_COLORS.get(color_name)
        if cfg is None:
            print(f"  [skip] unknown color '{color_name}'  "
                  f"(choices: {', '.join(PIPE_COLORS)})")
            continue

        skel, segments, strict_mask, extent_mask = trace_color(img, cfg, exclude=claimed)
        # Use strict_mask for exclusion — keeps color ownership boundaries tight
        claimed |= strict_mask

        if not segments:
            print(f"  {color_name:8s}: —")
            continue

        # Thicken 1-px skeleton to 3 px for visibility
        if skel is not None:
            thick = cv2.dilate(skel, _DRAW_K)
            out[thick > 0] = cfg['bgr']

        total_px = sum(s['px'] for s in segments)
        grand_total += len(segments)
        print(f"  {color_name:8s}: {len(segments)} segment(s)   {total_px} skeleton px")

        for idx, seg in enumerate(segments, 1):
            x1, y1, x2, y2 = seg['bbox']
            orientation = _segment_orientation(seg['bbox'])
            print(f"    seg {idx}: [{x1},{y1} → {x2},{y2}]  "
                  f"~{seg['px']} px  orientation={orientation}")
            if draw_boxes:
                cv2.rectangle(out, (x1, y1), (x2, y2), cfg['bgr'], 1)
                cv2.putText(
                    out,
                    f"{color_name[0].upper()}{idx}",
                    (x1, max(0, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    cfg['bgr'],
                    1,
                    cv2.LINE_AA,
                )

        if mask_corridor:
            # Use extent_mask (gap-bridged, no OPEN) for corridor so dashed lines
            # produce a continuous corridor rather than isolated per-dash blobs.
            color_corridor = build_corridor_mask(extent_mask, segments, (H, W), pad)
            combined_corridor |= color_corridor

    print(f"\nTotal pipe segments : {grand_total}")

    if mask_corridor:
        print(f"\nApplying corridor mask  (pad={pad}px, white_thresh={white_thresh})")
        corridor_px   = combined_corridor.sum()
        total_px      = H * W
        if white_thresh == 0:
            whited_out_px = (~combined_corridor).sum()
        else:
            whited_out_px = (~combined_corridor & np.all(out >= white_thresh, axis=2)).sum()
        print(f"  Corridor area  : {corridor_px:,} px  ({100*corridor_px/total_px:.1f}% of image)")
        print(f"  Pixels whited  : {whited_out_px:,} px")
        out = apply_corridor_mask(out, combined_corridor, white_thresh)

    cv2.imwrite(output_path, out)
    print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="P&ID ISO Line Tracer — skeleton-based with corridor masking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "image_path",
        help="Input P&ID patch image",
    )
    ap.add_argument(
        "-o", "--output",
        default=None,
        help="Output path (default: <stem>_traced.<ext> in same folder)",
    )
    ap.add_argument(
        "-c", "--colors",
        nargs="+",
        default=list(PIPE_COLORS.keys()),
        help="Colors to trace (default: all)",
    )
    ap.add_argument(
        "--boxes",
        action="store_true",
        help="Draw a bounding box around each detected segment",
    )
    ap.add_argument(
        "--mask-corridor",
        action="store_true",
        help="White out background pixels outside the padded corridor",
    )
    ap.add_argument(
        "--pad",
        type=int,
        default=65,
        help="Corridor half-width in pixels (default: 65). "
             "Applied perpendicular to each pipe run.",
    )
    ap.add_argument(
        "--white-thresh",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Controls what gets whited out outside the corridor. "
            "0 (default): strict geometric mode — white out EVERYTHING outside "
            "the corridor, including P&ID annotations and line art. "
            "1-255: selective mode — only white out pixels where ALL BGR "
            "channels >= N. Use 252 to preserve near-white P&ID content."
        ),
    )
    args = ap.parse_args()

    if args.output is None:
        p = Path(args.image_path)
        suffix = "_masked" if args.mask_corridor else "_traced"
        args.output = str(p.parent / f"{p.stem}{suffix}{p.suffix}")

    run(
        image_path=args.image_path,
        output_path=args.output,
        colors=args.colors,
        draw_boxes=args.boxes,
        mask_corridor=args.mask_corridor,
        pad=args.pad,
        white_thresh=args.white_thresh,
    )


if __name__ == "__main__":
    main()
