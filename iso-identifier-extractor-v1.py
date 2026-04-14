#iso-identifier-v1-gpu-11/03/2026
import cv2
import numpy as np
from PIL import Image
from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
import re
import colorsys
from pathlib import Path
from typing import List, Set, Tuple, Optional, Dict
import torch
import os

class TargetedPatchExtractor:
    """
    Targeted extension implementation - extends only specific regions
    """

    def __init__(self, patch_folder: str, patch_width: int = 640, patch_height: int = 640):
        self.patch_folder = Path(patch_folder)
        self.patch_width = patch_width
        self.patch_height = patch_height

        # ========== GPU SETUP ==========
        # Set CUDA device (use first GPU)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        # Check CUDA availability
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"\n{'='*60}")
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"{'='*60}\n")
        else:
            self.device = torch.device("cpu")
            print("\n⚠ CUDA not available, using CPU")
            print("  To enable GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n")
        # ===============================

        print("Initializing OCR...")
        try:
            # Try passing device to predictors
            self.foundation = FoundationPredictor(device=self.device)
            self.det_predictor = DetectionPredictor(device=self.device)
            self.rec_predictor = RecognitionPredictor(self.foundation, device=self.device)
        except TypeError:
            # If device parameter not supported, initialize without it
            # Surya should auto-detect CUDA
            print("  Note: Initializing predictors without explicit device parameter")
            self.foundation = FoundationPredictor()
            self.det_predictor = DetectionPredictor()
            self.rec_predictor = RecognitionPredictor(self.foundation)

        # Complete ISO identifier pattern
        self.complete_pattern = re.compile(
            r'(\d+(?:/\d+)?)\"\s*-\s*([A-Z0-9]{2})\s*-\s*(\d{4,6})(?:\s*-\s*([A-Z]\d{1,3}))?',
            re.IGNORECASE
        )

        # STRICT partial patterns - MUST have dashes and match ISO structure
        # ISO format: SIZE"-CODE-NUMBER-SUFFIX
        # SIZE: digits with optional /fraction
        # CODE: 2 letters (may be misread as 0W, etc)
        # NUMBER: 4-6 digits
        # SUFFIX: letter + 1-3 digits (but may be cut off as just "D", "B", "D0", etc.)

        self.partial_patterns = [
            # Missing size: -CODE-NUMBER-SUFFIX
            re.compile(r'^-\s*([A-Z0-9]{2})\s*-\s*(\d{4,6})\s*-\s*([A-Z]\d{0,3})$', re.I),

            # Missing suffix (with trailing dash): SIZE"-CODE-NUMBER-
            re.compile(r'^(\d+(?:/\d+)?)\"\s*-\s*([A-Z0-9]{2})\s*-\s*(\d{4,6})\s*-\s*$', re.I),

            # Partial suffix (no trailing dash): SIZE"-CODE-NUMBER-X or SIZE"-CODE-NUMBER-X0
            re.compile(r'^(\d+(?:/\d+)?)\"\s*-\s*([A-Z0-9]{2})\s*-\s*(\d{4,6})\s*-\s*([A-Z]\d{0,2})$', re.I),

            # Missing size and code: -NUMBER-SUFFIX
            re.compile(r'^-\s*(\d{4,6})\s*-\s*([A-Z]\d{0,3})$', re.I),

            # Missing code and suffix: SIZE"-NUMBER-
            re.compile(r'^(\d+(?:/\d+)?)\"\s*-\s*(\d{4,6})\s*-?\s*$', re.I),

            # Missing size: CODE-NUMBER-SUFFIX (no leading dash)
            re.compile(r'^([A-Z0-9]{2})\s*-\s*(\d{4,6})\s*-\s*([A-Z]\d{0,3})$', re.I),

            # Missing suffix: SIZE"-CODE-NUMBER (no trailing dash)
            re.compile(r'^(\d+(?:/\d+)?)\"\s*-\s*([A-Z0-9]{2})\s*-\s*(\d{4,6})$', re.I),

            # Missing size and suffix: CODE-NUMBER
            re.compile(r'^([A-Z0-9]{2})\s*-\s*(\d{4,6})$', re.I),

            # Just number-suffix: NUMBER-SUFFIX (including partial suffix)
            re.compile(r'^(\d{4,6})\s*-\s*([A-Z]\d{0,3})$', re.I),

            # SIZE"-CODE only (very partial)
            re.compile(r'^(\d+(?:/\d+)?)\"\s*-\s*([A-Z0-9]{2})$', re.I),

            # Missing everything except number: -NUMBER
            re.compile(r'^-\s*(\d{4,6})$', re.I),
        ]

    def normalize_identifier(self, size, code, number, suffix=None):
        """Normalize and fix OCR errors"""
        code = code.upper()

        # Fix OCR errors
        if code == '0W': code = 'OW'
        elif code == '0C': code = 'OC'
        elif code == 'C0': code = 'CO'
        elif code == 'CQ': code = 'CG'
        elif code == 'NC': code = 'NG'

        if not (len(code) == 2 and code.isalpha()):
            return None

        number = number.replace('O', '0').replace('I', '1')
        if not (4 <= len(number) <= 6 and number.isdigit()):
            return None

        if suffix:
            suffix = suffix.upper().replace('O', '0').replace('I', '1')
            if not (2 <= len(suffix) <= 4 and suffix[0].isalpha() and suffix[1:].isdigit()):
                return None
            return f'{size}"-{code}-{number}-{suffix}'

        return None

    def extract_text_from_predictions(self, predictions) -> List[Tuple[str, tuple]]:
        """Extract text with bboxes from Surya predictions"""
        results = []
        for line in predictions[0].text_lines:
            text = line.text.strip()
            if text:
                # bbox format: (x_min, y_min, x_max, y_max)
                results.append((text, line.bbox))
        return results

    def extract_complete_identifiers(self, texts_with_bboxes: List[Tuple[str, tuple]]) -> Set[str]:
        """Extract complete ISO identifiers"""
        found = set()

        for text, _ in texts_with_bboxes:
            match = self.complete_pattern.search(text)
            if match:
                size = match.group(1)
                code = match.group(2)
                number = match.group(3)
                suffix = match.group(4)

                formatted = self.normalize_identifier(size, code, number, suffix)
                if formatted:
                    found.add(formatted)

        return found

    def has_partial_pattern(self, text: str) -> bool:

        text = text.strip()

        # Must contain dash
        if '-' not in text:
            return False

        # If contains quote mark ("), it's likely an ISO identifier with size
        if '"' in text:
            # Check against partial patterns
            for pattern in self.partial_patterns:
                if pattern.match(text):
                    return True
            return False

        # No quote mark - more careful checking needed

        # Reject equipment tags (2-3 letters + dash + exactly 4 digits, nothing else)
        # Example: EI-8508, FI-8530, PT-8113
        equipment_tag_pattern = re.compile(r'^[A-Z]{2,3}-\d{4}$', re.I)
        if equipment_tag_pattern.match(text):
            return False

        # Reject document numbers (contains more than 4 dash-separated parts)
        # Example: SPP-11-09-0110 (4 parts), THAI-3C-SPP-11-09-0304 (6 parts)
        parts = text.split('-')
        if len(parts) > 4:  # ISO identifiers have max 4 parts
            return False

        # Additional check: if it has exactly 4 parts and none look like ISO components, reject
        if len(parts) == 4:
            # Check if any part looks like it could be ISO number (4+ digits)
            has_long_number = any(part.isdigit() and len(part) >= 4 for part in parts)
            if not has_long_number:
                return False

        # Check against valid partial patterns
        for pattern in self.partial_patterns:
            if pattern.match(text):
                return True

        return False

    def get_patch_position(self, patch_name: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract row and column from patch name"""
        match = re.search(r'_r(\d+)_c(\d+)', patch_name)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    def determine_edge_and_direction(self, bbox: tuple, width: int, height: int,
                                     edge_threshold: int = 80) -> Tuple[Optional[str], Optional[str]]:

        x_min, y_min, x_max, y_max = bbox

        # Check which edge the bbox is closest to
        if y_min < edge_threshold:
            return 'top', 'up'
        elif y_max > (height - edge_threshold):
            return 'bottom', 'down'
        elif x_min < edge_threshold:
            return 'left', 'left'
        elif x_max > (width - edge_threshold):
            return 'right', 'right'

        return None, None

    def get_neighbor_patch(self, patch_name: str, direction: str) -> Optional[str]:
        """Get neighbor patch name based on direction"""
        row, col = self.get_patch_position(patch_name)
        if row is None:
            return None

        # Calculate neighbor position
        if direction == 'up':
            row -= 1
        elif direction == 'down':
            row += 1
        elif direction == 'left':
            col -= 1
        elif direction == 'right':
            col += 1

        # Check bounds (4 rows × 6 columns)
        if row < 0 or row >= 4 or col < 0 or col >= 6:
            return None

        base_name = patch_name.split('_r')[0]
        return f"{base_name}_r{row}_c{col}.png"

    def create_targeted_extension(self, current_patch: np.ndarray,
                                  neighbor_patch: np.ndarray,
                                  bbox: tuple,
                                  direction: str,
                                  padding: int = 40,
                                  lateral_margin: int = 50) -> Optional[np.ndarray]:

        # Convert bbox to integers
        x_min, y_min, x_max, y_max = map(int, bbox)
        h_curr, w_curr = current_patch.shape[:2]
        h_neigh, w_neigh = neighbor_patch.shape[:2]

        # Add lateral margins (convert to int)
        x_min_expanded = int(max(0, x_min - lateral_margin))
        x_max_expanded = int(min(w_curr, x_max + lateral_margin))
        y_min_expanded = int(max(0, y_min - lateral_margin))
        y_max_expanded = int(min(h_curr, y_max + lateral_margin))

        if direction == 'up':
            # Extend upward into neighbor's bottom region
            # Take: [bottom of neighbor] + [top portion of current around bbox]

            # From neighbor: take bottom padding pixels at same x-range
            x_min_neigh = int(max(0, x_min_expanded))
            x_max_neigh = int(min(w_neigh, x_max_expanded))
            neighbor_strip = neighbor_patch[-padding:, x_min_neigh:x_max_neigh]

            # From current: take region around bbox
            current_region = current_patch[y_min_expanded:y_max_expanded, x_min_expanded:x_max_expanded]

            # Stack vertically: neighbor bottom + current region
            extended = np.vstack([neighbor_strip, current_region])

        elif direction == 'down':
            # Extend downward into neighbor's top region

            # From current: take region around bbox
            current_region = current_patch[y_min_expanded:y_max_expanded, x_min_expanded:x_max_expanded]

            # From neighbor: take top padding pixels at same x-range
            x_min_neigh = int(max(0, x_min_expanded))
            x_max_neigh = int(min(w_neigh, x_max_expanded))
            neighbor_strip = neighbor_patch[:padding, x_min_neigh:x_max_neigh]

            # Stack vertically: current region + neighbor top
            extended = np.vstack([current_region, neighbor_strip])

        elif direction == 'left':
            # Extend leftward into neighbor's right region

            # From neighbor: take right padding pixels at same y-range
            y_min_neigh = int(max(0, y_min_expanded))
            y_max_neigh = int(min(h_neigh, y_max_expanded))
            neighbor_strip = neighbor_patch[y_min_neigh:y_max_neigh, -padding:]

            # From current: take region around bbox
            current_region = current_patch[y_min_expanded:y_max_expanded, x_min_expanded:x_max_expanded]

            # Stack horizontally: neighbor right + current region
            extended = np.hstack([neighbor_strip, current_region])

        elif direction == 'right':
            # Extend rightward into neighbor's left region

            # From current: take region around bbox
            current_region = current_patch[y_min_expanded:y_max_expanded, x_min_expanded:x_max_expanded]

            # From neighbor: take left padding pixels at same y-range
            y_min_neigh = int(max(0, y_min_expanded))
            y_max_neigh = int(min(h_neigh, y_max_expanded))
            neighbor_strip = neighbor_patch[y_min_neigh:y_max_neigh, :padding]

            # Stack horizontally: current region + neighbor left
            extended = np.hstack([current_region, neighbor_strip])

        else:
            return None

        return extended

    def run_ocr_on_image(self, img: np.ndarray) -> Tuple[Set[str], List[Tuple[str, tuple]]]:
        """Run OCR on image array"""
        # Convert BGR to RGB for PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Run OCR
        predictions = self.rec_predictor([img_pil], det_predictor=self.det_predictor)

        # Extract text with bboxes
        texts_with_bboxes = self.extract_text_from_predictions(predictions)

        # Find complete identifiers
        complete_ids = self.extract_complete_identifiers(texts_with_bboxes)

        return complete_ids, texts_with_bboxes

    def process_single_patch(self, patch_path: Path) -> Tuple[Set[str], List[dict]]:
        """Process a single patch to find complete and partial identifiers"""
        print(f"\nProcessing: {patch_path.name}")

        # Load patch
        patch = cv2.imread(str(patch_path))
        if patch is None:
            print(f"  ERROR: Could not load patch")
            return set(), []

        height, width = patch.shape[:2]

        # Run OCR
        complete_ids, texts_with_bboxes = self.run_ocr_on_image(patch)

        if complete_ids:
            print(f"  ✓ Complete: {complete_ids}")

        # Debug: Show all detected text
        print(f"  📝 All detected text ({len(texts_with_bboxes)} items):")
        for text, bbox in texts_with_bboxes[:10]:  # Show first 10 only
            print(f"     '{text}'")
        if len(texts_with_bboxes) > 10:
            print(f"     ... and {len(texts_with_bboxes) - 10} more")

        # Find partials
        partials_info = []
        for text, bbox in texts_with_bboxes:
            # Skip if part of complete identifier
            if any(text in cid for cid in complete_ids):
                continue

            # Check if partial pattern
            if self.has_partial_pattern(text):
                edge, direction = self.determine_edge_and_direction(bbox, width, height)

                if edge and direction:
                    print(f"  ⚠ Partial on {edge}: '{text}' at bbox {bbox}")
                    partials_info.append({
                        'text': text,
                        'bbox': bbox,
                        'edge': edge,
                        'direction': direction
                    })
                else:
                    # Partial found but not on edge
                    print(f"  ℹ Partial (not on edge): '{text}'")

        return complete_ids, partials_info

    def process_extensions(self, patch_path: Path, partials_info: List[dict]) -> Set[str]:
        """Process targeted extensions for partial identifiers"""
        found_in_extensions = set()

        if not partials_info:
            return found_in_extensions

        # Load current patch once
        current_patch = cv2.imread(str(patch_path))
        if current_patch is None:
            return found_in_extensions

        for partial in partials_info:
            text = partial['text']
            bbox = partial['bbox']
            direction = partial['direction']

            # Get neighbor patch
            neighbor_name = self.get_neighbor_patch(patch_path.name, direction)
            if not neighbor_name:
                print(f"  ⚠ No neighbor for '{text}' in {direction} direction")
                continue

            neighbor_path = self.patch_folder / neighbor_name
            if not neighbor_path.exists():
                print(f"  ⚠ Neighbor not found: {neighbor_name}")
                continue

            # Load neighbor patch
            neighbor_patch = cv2.imread(str(neighbor_path))
            if neighbor_patch is None:
                print(f"  ⚠ Could not load neighbor: {neighbor_name}")
                continue

            print(f"  → Creating targeted extension to {neighbor_name} ({direction}) for '{text}'")

            # Create targeted extended region
            extended_region = self.create_targeted_extension(
                current_patch,
                neighbor_patch,
                bbox,
                direction,
                padding=40,
                lateral_margin=50
            )

            if extended_region is None or extended_region.size == 0:
                print(f"  ✗ Failed to create extended region")
                continue

            # Run OCR on small extended region
            try:
                complete_ids, _ = self.run_ocr_on_image(extended_region)

                if complete_ids:
                    print(f"  ✓ Found in extension: {complete_ids}")
                    found_in_extensions.update(complete_ids)
                else:
                    print(f"  ⚠ No complete identifiers in extension")

            except Exception as e:
                print(f"  ✗ OCR error: {e}")

        return found_in_extensions

    def extract_from_page(self, page_name: str) -> dict:
        """Extract all ISO identifiers from a page"""
        print("="*80)
        print(f"EXTRACTING FROM PAGE: {page_name}")
        print("="*80)

        # Find all patches
        patch_files = sorted(self.patch_folder.glob(f"{page_name}_r*_c*.png"))
        print(f"\nFound {len(patch_files)} patches")

        all_complete = set()
        all_from_extensions = set()

        # Process each patch
        for patch_path in patch_files:
            # Get complete and partials from patch
            complete_ids, partials_info = self.process_single_patch(patch_path)
            all_complete.update(complete_ids)

            # Process extensions for partials
            if partials_info:
                extension_ids = self.process_extensions(patch_path, partials_info)
                all_from_extensions.update(extension_ids)

        # Combine results
        all_identifiers = all_complete | all_from_extensions

        return {
            'page_name': page_name,
            'total_patches': len(patch_files),
            'identifiers': sorted(all_identifiers),
            'total_found': len(all_identifiers),
            'from_complete': len(all_complete),
            'from_extensions': len(all_from_extensions),
        }


def check_cuda():
    """Verify CUDA availability and GPU info"""
    print("\n" + "="*80)
    print("CUDA CHECK")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ WARNING: CUDA not available - will use CPU")
        print("  Install CUDA-enabled PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("="*80 + "\n")


def main():
    # Configuration
    patch_folder = r"E:\Projects\P&ID Versions\P&ID V2\iso-identifier-test-patches\test-node1.4-patches"
    page_name = "ERCP_2"

    # Create extractor
    extractor = TargetedPatchExtractor(patch_folder)

    # Extract
    results = extractor.extract_from_page(page_name)

    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\nPage: {results['page_name']}")
    print(f"Total patches: {results['total_patches']}")
    print(f"\nTotal identifiers: {results['total_found']}")
    print(f"  - From complete patches: {results['from_complete']}")
    print(f"  - From targeted extensions: {results['from_extensions']}")

    print(f"\nAll identifiers ({len(results['identifiers'])}):")
    for idx, identifier in enumerate(results['identifiers'], 1):
        print(f"  {idx:2d}. {identifier}")

    # Check critical identifiers
    print("\n" + "="*80)
    print("CHECKING CRITICAL IDENTIFIERS")
    print("="*80)

    critical = [
        "1/2\"-CP-8113-D48",
        "6\"-OW-8104-D03",
        "8\"-OW-8105-D03",
        "6\"-OW-8601-D03",
        "8\"-CL-8903-D03",
        "24\"-CG-8111",
    ]

    found_set = set(results['identifiers'])

    for c in critical:
        if c in found_set:
            print(f"  ✓ {c} - FOUND")
        else:
            # Check for similar (size might be wrong)
            similar = [v for v in found_set if c.split('-')[1:] == v.split('-')[1:] if '-' in v and '-' in c]
            if similar:
                print(f"  ~ {c} - Similar: {similar[0]}")
            else:
                print(f"  ✗ {c} - MISSING")

    print("\n" + "="*80)

    return results


# ============================================================
# Color Utility Functions (shared with iso-legends-extractor-v4)
# ============================================================

def rgb_to_hex(rgb):
    """Convert RGB tuple to hex string"""
    return '#%02x%02x%02x' % tuple(int(x) for x in rgb)


def rgb_to_hsv(rgb):
    """Convert RGB to HSV (OpenCV scale: H 0-180, S 0-255, V 0-255)"""
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return (int(h * 180), int(s * 255), int(v * 255))


def is_colorful(rgb, brightness_thresh=30, color_diff_thresh=20):
    """Return True if color is not near-grayscale (black / white / gray)"""
    r, g, b = rgb
    brightness = np.mean(rgb)
    color_diff = np.max(rgb) - np.min(rgb)
    return (brightness > brightness_thresh) and (brightness < 240) and (color_diff > color_diff_thresh)


def detect_line_pattern(color_region, iso_name="unknown"):
    """
    Detect whether a color region represents a solid or dashed line.
    Returns: 'solid', 'dashed', or 'unknown'
    """
    if color_region.size == 0:
        return 'unknown'

    h, w = color_region.shape[:2]
    if h < 2 or w < 10:
        return 'unknown'

    sample_rows = []
    if h >= 6:
        sample_rows = [h // 4, h // 2, 3 * h // 4]
    elif h >= 3:
        sample_rows = [h // 3, 2 * h // 3]
    else:
        sample_rows = [h // 2]

    all_gaps = []
    all_coverage = []

    for row_idx in sample_rows:
        if row_idx >= h:
            continue
        row = color_region[row_idx, :]
        colorful_mask = np.array([is_colorful(pixel) for pixel in row])
        colored_count = np.sum(colorful_mask)
        if colored_count < 2:
            continue
        colored_indices = np.where(colorful_mask)[0]
        if len(colored_indices) < 2:
            continue
        start_idx = colored_indices[0]
        end_idx = colored_indices[-1]
        line_segment = colorful_mask[start_idx:end_idx + 1]
        if len(line_segment) < 5:
            continue

        gaps = 0
        in_gap = False
        gap_lengths = []
        current_gap_length = 0
        for val in line_segment:
            if not val:
                if not in_gap:
                    gaps += 1
                    in_gap = True
                    current_gap_length = 1
                else:
                    current_gap_length += 1
            else:
                if in_gap:
                    gap_lengths.append(current_gap_length)
                    current_gap_length = 0
                in_gap = False

        coverage = np.sum(line_segment) / len(line_segment)
        all_gaps.append(gaps)
        all_coverage.append(coverage)

    if not all_gaps:
        total_colorful = sum(1 for row in color_region for pixel in row if is_colorful(pixel))
        total_pixels = h * w
        overall_coverage = total_colorful / total_pixels if total_pixels > 0 else 0
        if overall_coverage > 0.3:
            return 'solid'
        elif overall_coverage > 0.1:
            return 'dashed'
        else:
            return 'solid'

    avg_gaps = np.mean(all_gaps)
    avg_coverage = np.mean(all_coverage)

    if avg_gaps >= 3:
        return 'dashed'
    elif avg_gaps >= 1.5 and avg_coverage < 0.85:
        return 'dashed'
    elif avg_gaps < 1 and avg_coverage > 0.8:
        return 'solid'
    elif avg_coverage >= 0.85:
        return 'solid'
    elif avg_coverage < 0.6:
        return 'dashed'
    else:
        return 'dashed' if avg_gaps >= 1 else 'solid'


# ============================================================
# ISO Legend Extraction (logic from iso-legends-extractor-v4)
# ============================================================

def hue_to_color_name(hue_opencv: int) -> str:
    """Map an OpenCV hue value (0-180) to an approximate English color name."""
    h = hue_opencv * 2  # convert to 0-360
    if h < 15 or h >= 345:   return "red"
    elif h < 45:              return "orange"
    elif h < 75:              return "yellow"
    elif h < 150:             return "green"
    elif h < 195:             return "cyan"
    elif h < 255:             return "blue"
    elif h < 285:             return "indigo/violet"
    elif h < 345:             return "magenta/pink"
    return "red"


def dominant_color_in_row(
    img_rgb: np.ndarray,
    row_y1: int, row_y2: int,
    exclude_x1: int, exclude_x2: int,
    min_count: int = 8,
    min_density: float = 0.06,
) -> Optional[Tuple]:
    """
    Find the dominant ISO-line colour in a horizontal legend row, ignoring
    the X range [exclude_x1, exclude_x2] (which contains the text label).

    Strategy:
      1. Collect all colorful pixels in the row (outside the label area).
      2. Build a hue histogram over those pixels.
      3. Find the peak hue bin.
      4. Return the median RGB of pixels in that hue cluster.

    This is layout-agnostic (works whether the colour sample is to the left
    or right of the label) and immune to neighbour-row bleed (each row is
    searched independently with its own tight Y window).
    """
    h_img, w_img = img_rgb.shape[:2]
    row_y1 = max(0, row_y1)
    row_y2 = min(h_img, row_y2)
    if row_y2 <= row_y1:
        return None

    # Build a column mask that excludes the label text region
    col_indices = np.array(
        [x for x in range(w_img) if not (exclude_x1 <= x <= exclude_x2)]
    )
    if len(col_indices) == 0:
        return None

    row_slice = img_rgb[row_y1:row_y2, :, :]          # shape (h, w, 3)
    outside   = row_slice[:, col_indices, :]            # shape (h, ncols, 3)
    pixels    = outside.reshape(-1, 3)                  # shape (N, 3)

    # Filter to colorful pixels only
    bright     = pixels[:, 0].astype(int)
    green      = pixels[:, 1].astype(int)
    blue_ch    = pixels[:, 2].astype(int)
    brightness = (bright + green + blue_ch) / 3.0
    color_diff = pixels.max(axis=1).astype(int) - pixels.min(axis=1).astype(int)
    colorful_mask = (brightness > 30) & (brightness < 245) & (color_diff > 20)
    colorful_pixels = pixels[colorful_mask]

    count   = len(colorful_pixels)
    density = count / len(pixels) if len(pixels) > 0 else 0.0
    if count < min_count or density < min_density:
        return None

    # Compute hue for each colorful pixel (use OpenCV H 0-180 scale)
    cp_f  = colorful_pixels.astype(np.float32) / 255.0
    r, g, b = cp_f[:, 0], cp_f[:, 1], cp_f[:, 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c + 1e-9

    hue = np.zeros(len(r))
    m = max_c == r; hue[m] = (60 * ((g[m] - b[m]) / delta[m])) % 360
    m = max_c == g; hue[m] = (60 * ((b[m] - r[m]) / delta[m]) + 120)
    m = max_c == b; hue[m] = (60 * ((r[m] - g[m]) / delta[m]) + 240)
    hue_opencv = (hue / 2.0).astype(int) % 180   # 0-180

    # Histogram over hue (36 bins × 5° each)
    hist, edges = np.histogram(hue_opencv, bins=36, range=(0, 180))
    peak_bin    = int(np.argmax(hist))

    # Cluster: all pixels within ±2 bins (~±10°) of the peak
    h_center = (edges[peak_bin] + edges[peak_bin + 1]) / 2.0
    in_cluster = np.abs(hue_opencv.astype(float) - h_center) <= 12
    if np.sum(in_cluster) < min_count:
        return None

    cluster_rgb = colorful_pixels[in_cluster]
    return tuple(np.median(cluster_rgb, axis=0).astype(int))


def detect_iso_texts_and_colors(image_path, debug=False, save_color_regions=False):
    """
    Extract ISO line definitions and their colors from a legend image using Surya OCR.

    Args:
        image_path: Path to the legend image
        debug:      Save a debug visualisation if True
        save_color_regions: Save per-color crops for inspection if True

    Returns:
        iso_color_map: dict  {iso_name: {'rgb', 'hsv', 'hex', 'pattern'}}
    """
    foundation = FoundationPredictor()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation)

    img = cv2.imread(image_path)
    if img is None:
        print(f"  ERROR: Cannot load legend image: {image_path}")
        return {}
    h, w = img.shape[:2]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)

    predictions = rec_predictor([pil_image], det_predictor=det_predictor)
    text_lines = predictions[0].text_lines
    print(f"  Detected {len(text_lines)} text lines from legend image")

    # Node-identifier patterns (same set as legend extractor v4)
    node_patterns = [
        r'GBN-CPP-\d+',
        r'CP\s*ISO\s*[\d.]+',
        r'BDD-CPP-\d+',
        r'\d+\.',
        r'\d+$',
    ]

    iso_boxes = []
    for line in text_lines:
        text = line.text.strip()
        iso_name = None
        matched_pattern = None
        for pattern in node_patterns:
            match = re.search(pattern, text)
            if match:
                iso_name = match.group(0)
                matched_pattern = pattern
                break
        if iso_name:
            if matched_pattern == r'\d+$':
                if len(text) > 5:
                    continue
                if text != iso_name and not re.match(r'^\d+\.?$', text):
                    continue
            bbox = line.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            iso_boxes.append((iso_name, (x1, y1, x2, y2)))

    print(f"  Matched {len(iso_boxes)} ISO identifiers in legend\n")

    iso_color_map = {}
    if debug:
        debug_img = img.copy()

    for iso_name, (x1, y1, x2, y2) in iso_boxes:
        # ── Full-row dominant-colour extraction ──────────────────────────
        # We scan the ENTIRE ROW height (±5 px to catch lines slightly off
        # the text baseline) while EXCLUDING the label text X-range.
        # The dominant hue cluster in that row is the ISO line colour.
        # This is layout-agnostic (left/right sample) and immune to
        # neighbour-row bleed because each entry's Y window is independent.
        expand_y    = 5
        row_y1      = max(0, y1 - expand_y)
        row_y2      = min(h, y2 + expand_y)

        avg_color_tuple = dominant_color_in_row(
            img_rgb,
            row_y1, row_y2,
            exclude_x1=x1, exclude_x2=x2,
        )

        # Fallback: if full-row approach failed, try a wider Y expansion
        if avg_color_tuple is None:
            avg_color_tuple = dominant_color_in_row(
                img_rgb,
                max(0, y1 - 15), min(h, y2 + 15),
                exclude_x1=x1, exclude_x2=x2,
                min_count=5, min_density=0.03,
            )

        if avg_color_tuple is not None:
            hsv_color = rgb_to_hsv(avg_color_tuple)
            iso_normalized = re.sub(r'\s+', ' ', iso_name.upper())

            # Build a colour-region crop for visual inspection
            best_color_region = img_rgb[row_y1:row_y2, :, :]
            pattern = detect_line_pattern(best_color_region, iso_name)

            if save_color_regions:
                os.makedirs('color_regions', exist_ok=True)
                region_path = f'color_regions/{iso_normalized.replace("/", "_").replace(" ", "_")}.png'
                cv2.imwrite(region_path, cv2.cvtColor(best_color_region, cv2.COLOR_RGB2BGR))

            iso_color_map[iso_normalized] = {
                'rgb':     avg_color_tuple,
                'hsv':     hsv_color,
                'hex':     rgb_to_hex(avg_color_tuple),
                'pattern': pattern,
            }
        else:
            print(f"  ⚠ Could not extract colour for {iso_name} — row had no distinct colorful region")

        if debug:
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if avg_color_tuple is not None:
                iso_normalized = re.sub(r'\s+', ' ', iso_name.upper())
                cv2.putText(debug_img, iso_normalized, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if debug and iso_boxes:
        cv2.imwrite('legend_debug.png', debug_img)

    return iso_color_map


# ============================================================
# ISOLineAssociator
# ============================================================

class ISOLineAssociator(TargetedPatchExtractor):
    """
    Extends TargetedPatchExtractor to associate detected ISO identifiers
    with the colored ISO lines they are physically adjacent to in a P&ID patch.

    Workflow per patch:
      1. Quick color scan  →  skip if no colored pixels (saves OCR time)
      2. Build HSV proximity masks for each legend color present in the patch
      3. Run OCR to detect ISO identifiers with their bounding boxes
      4. For each identifier, find the nearest proximity mask  →  ISO line name
      5. Handle partial identifiers at patch edges via the same extension logic
         as TargetedPatchExtractor, re-using the current patch's proximity masks
         for the association step
    """

    def __init__(
        self,
        patch_folder: str,
        patch_width: int = 640,
        patch_height: int = 640,
        proximity_px: int = 12,
        min_overlap_px: int = 8,
        color_h_tol: int = 8,
        color_s_min_ratio: float = 0.50,
        min_color_pixels: int = 30,
        min_line_span_px: int = 15,
    ):
        super().__init__(patch_folder, patch_width, patch_height)
        self.iso_color_map: Dict = {}
        # How many pixels to dilate each color mask (covers the gap between
        # a drawn line and its nearby text label)
        self.proximity_px = proximity_px
        # Minimum overlapping pixels between a text bbox and a proximity mask
        # before we accept the association
        self.min_overlap_px = min_overlap_px
        # HSV hue tolerance (out of 180) — kept tight to avoid cross-colour
        # confusion between legend entries with similar hues
        self.color_h_tol = color_h_tol
        # Saturation floor = this fraction of the legend color's saturation
        self.color_s_min_ratio = color_s_min_ratio
        # Minimum raw matching pixels in a patch before an ISO color is
        # considered "present" — filters scattered artifact noise
        self.min_color_pixels = min_color_pixels
        # A connected component must span at least this many pixels in one
        # axis to be counted as a real drawn line (not isolated noise)
        self.min_line_span_px = min_line_span_px

    # ------------------------------------------------------------------
    # Legend loading
    # ------------------------------------------------------------------

    def load_legend(self, legend_path: str) -> Dict:
        """Load ISO line color definitions from a legend image."""
        print(f"\nLoading legend: {legend_path}")
        self.iso_color_map = detect_iso_texts_and_colors(legend_path, save_color_regions=True)
        if not self.iso_color_map:
            print("  WARNING: No ISO definitions extracted from legend")
        else:
            print(f"  Loaded {len(self.iso_color_map)} ISO line definitions:")
            print(f"  {'Name':<22} {'Hex':<10} {'Approx color':<16} H   S   V   Pattern")
            print(f"  {'-'*22} {'-'*10} {'-'*16} {'-'*3} {'-'*3} {'-'*3} {'-'*8}")
            for name, info in sorted(self.iso_color_map.items()):
                h, s, v = info['hsv']
                color_name = hue_to_color_name(h)
                print(f"  {name:<22} {info['hex']:<10} {color_name:<16} {h:3d} {s:3d} {v:3d} "
                      f"{info.get('pattern', '?')}")
            # Warn if any two legend colors map to the same approximate color name
            # (likely a missampled legend row)
            from collections import defaultdict
            name_groups: dict = defaultdict(list)
            for name, info in self.iso_color_map.items():
                color_name = hue_to_color_name(info['hsv'][0])
                name_groups[color_name].append(name)
            for color_name, entries in name_groups.items():
                if len(entries) > 1:
                    print(f"  ⚠ WARNING: {entries} all extracted as '{color_name}' — "
                          f"check color_regions/ crops, likely a legend row mis-sampled")
            # Also warn on very close hues
            names = sorted(self.iso_color_map.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    h1 = self.iso_color_map[names[i]]['hsv'][0]
                    h2 = self.iso_color_map[names[j]]['hsv'][0]
                    diff = min(abs(h1 - h2), 180 - abs(h1 - h2))
                    if diff < self.color_h_tol * 2:
                        print(f"  ⚠ WARNING: {names[i]} and {names[j]} have very similar hues "
                              f"(ΔH={diff}) — may cause cross-detection in patches")
        print(f"\n  NOTE: Color region crops saved to ./color_regions/ — verify each crop shows "
              f"the correct ISO line color sample for that entry")
        return self.iso_color_map

    # ------------------------------------------------------------------
    # Color detection helpers
    # ------------------------------------------------------------------

    def has_colored_lines(self, img_rgb: np.ndarray) -> bool:
        """
        Fast scan: return True if the patch contains >= 50 colorful
        (non-grayscale, non-black) pixels.

        NOTE: The upper V ceiling was intentionally removed.  Yellow ISO lines
        have V≈255 in HSV (pure bright yellow), so `v < 240` would silently
        exclude every yellow pixel, causing entire yellow-line patches to be
        skipped.  White pixels are already filtered by their S≈0 (they fail
        the s > 40 check), so no upper V bound is needed.
        """
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        s = img_hsv[:, :, 1]
        v = img_hsv[:, :, 2]
        colorful_count = int(np.sum((s > 40) & (v > 30)))
        return colorful_count >= 50

    def get_color_mask(self, img_rgb: np.ndarray, target_rgb: tuple) -> np.ndarray:
        """
        Binary mask of pixels in img_rgb whose HSV values fall within
        the tolerance window centred on target_rgb.

        Handles hue wraparound (e.g. reds that straddle H=0/H=180).
        """
        target_h, target_s, target_v = rgb_to_hsv(target_rgb)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        s_min = max(20, int(target_s * self.color_s_min_ratio))
        v_min = 25
        # No hard ceiling on V: bright yellows (and other saturated light colours)
        # have V=255.  White is excluded by s_min (white has S≈0), so there is
        # no risk of false-positives from near-white pixels.
        v_max = 255

        h_low  = (target_h - self.color_h_tol) % 180
        h_high = (target_h + self.color_h_tol) % 180

        if h_low <= h_high:
            mask = cv2.inRange(
                img_hsv,
                np.array([h_low,  s_min, v_min]),
                np.array([h_high, 255,   v_max]),
            )
        else:
            # Hue wraps around 0/180
            m1 = cv2.inRange(
                img_hsv,
                np.array([h_low, s_min, v_min]),
                np.array([179,   255,   v_max]),
            )
            m2 = cv2.inRange(
                img_hsv,
                np.array([0,      s_min, v_min]),
                np.array([h_high, 255,   v_max]),
            )
            mask = cv2.bitwise_or(m1, m2)

        return mask

    def _has_line_structure(self, mask: np.ndarray) -> bool:
        """
        Return True if `mask` contains at least one connected component
        whose bounding box spans >= self.min_line_span_px pixels in at least
        one axis AND has area >= 10 px.

        This distinguishes real drawn lines (long, thin connected runs) from
        scattered noise pixels (many tiny disconnected blobs) that can match
        a colour range accidentally.
        """
        if int(np.sum(mask > 0)) < 10:
            return False
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):  # 0 is background
            comp_w    = stats[i, cv2.CC_STAT_WIDTH]
            comp_h    = stats[i, cv2.CC_STAT_HEIGHT]
            comp_area = stats[i, cv2.CC_STAT_AREA]
            if comp_area >= 10 and (comp_w >= self.min_line_span_px or
                                    comp_h >= self.min_line_span_px):
                return True
        return False

    def _build_color_masks(
        self, img_rgb: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Build raw and dilated binary masks for every ISO line color present
        in this patch.

        An ISO color is included only if:
          1. Its raw mask has >= self.min_color_pixels matching pixels, AND
          2. Those pixels form at least one line-like connected component
             (span >= self.min_line_span_px in one axis) — filters noise blobs.

        Returns:
            raw_masks       : undilated pixel masks  {iso_name: mask}
            proximity_masks : dilated masks           {iso_name: mask}
              (proximity_masks are kept for the edge-extension association step)
        """
        kernel = np.ones(
            (self.proximity_px * 2 + 1, self.proximity_px * 2 + 1),
            np.uint8,
        )
        raw_masks:       Dict[str, np.ndarray] = {}
        proximity_masks: Dict[str, np.ndarray] = {}

        for iso_name, color_info in self.iso_color_map.items():
            if not is_colorful(color_info['rgb']):
                continue
            raw_mask = self.get_color_mask(img_rgb, color_info['rgb'])
            pixel_count = int(np.sum(raw_mask > 0))

            if pixel_count < self.min_color_pixels:
                print(f"      [COLOR] {iso_name:<22} {color_info['hex']}  "
                      f"pixels={pixel_count:4d} → SKIP (< {self.min_color_pixels} px)")
                continue

            if not self._has_line_structure(raw_mask):
                print(f"      [COLOR] {iso_name:<22} {color_info['hex']}  "
                      f"pixels={pixel_count:4d} → SKIP (scattered noise, no line)")
                continue

            dilated = cv2.dilate(raw_mask, kernel, iterations=1)
            raw_masks[iso_name]       = raw_mask
            proximity_masks[iso_name] = dilated
            print(f"      [COLOR] {iso_name:<22} {color_info['hex']}  "
                  f"pixels={pixel_count:4d} → ACCEPTED")

        return raw_masks, proximity_masks

    # Legacy wrapper kept for any external callers
    def detect_color_proximity_masks(self, img_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        _, proximity_masks = self._build_color_masks(img_rgb)
        return proximity_masks

    # ------------------------------------------------------------------
    # Association logic
    # ------------------------------------------------------------------

    def find_nearest_iso_line(
        self,
        bbox: tuple,
        raw_masks: Dict[str, np.ndarray],
        img_h: int,
        img_w: int,
    ) -> Optional[str]:
        """
        Return the ISO line name whose drawn pixels are geometrically closest
        to the centre of `bbox`.

        WHY distance-to-centroid instead of max dilated-mask overlap:
          The previous approach (most overlap in dilated mask) failed when two
          ISO lines were both near the text — the one with a broader or denser
          dilated blob won regardless of which line the text actually labels.
          Euclidean distance to the nearest raw line pixel is unambiguous: the
          identifier text is always placed next to ITS own line, so the correct
          ISO color's pixels will always be closest.

        Returns None if the nearest line pixel is farther than
        proximity_px * 6 (generous to handle text placed slightly off the line).
        """
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(img_w, int(bbox[2]))
        y2 = min(img_h, int(bbox[3]))

        if x2 <= x1 or y2 <= y1:
            return None

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        # Tighter threshold: identifier text must be within ~40 px of the line.
        # proximity_px=12 → max_dist=40 px.  This rejects identifiers that are
        # near a coloured area but not physically hugging their own ISO line.
        max_dist = self.proximity_px * 3 + 4  # e.g., 40 px at proximity_px=12

        best_iso  = None
        best_dist = float('inf')
        candidates = []

        for iso_name, raw_mask in raw_masks.items():
            ys, xs = np.where(raw_mask > 0)
            if len(xs) == 0:
                continue
            dists    = np.sqrt((xs.astype(float) - cx) ** 2 +
                               (ys.astype(float) - cy) ** 2)
            min_dist = float(np.min(dists))
            candidates.append((iso_name, min_dist))
            if min_dist < best_dist:
                best_dist = min_dist
                best_iso  = iso_name

        # Diagnostic: log all candidates so wrong associations are visible
        if candidates:
            cand_str = "  ".join(
                f"{n}={d:.1f}px" for n, d in sorted(candidates, key=lambda x: x[1])
            )
            print(f"      [ASSOC] bbox=({x1},{y1})-({x2},{y2})  cx={cx:.0f},cy={cy:.0f}  "
                  f"candidates: {cand_str}")

        return best_iso if best_dist <= max_dist else None

    # ------------------------------------------------------------------
    # OCR with bbox tracking
    # ------------------------------------------------------------------

    def run_ocr_with_bbox_map(
        self, img: np.ndarray
    ) -> Tuple[Dict[str, tuple], List[Tuple[str, tuple]]]:
        """
        Run Surya OCR on a BGR patch image.

        Returns:
            identifier_bbox_map : {complete_identifier_str: bbox}
            texts_with_bboxes   : raw [(text, bbox), ...] from OCR
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        predictions = self.rec_predictor([img_pil], det_predictor=self.det_predictor)
        texts_with_bboxes = self.extract_text_from_predictions(predictions)

        identifier_bbox_map: Dict[str, tuple] = {}
        for text, bbox in texts_with_bboxes:
            match = self.complete_pattern.search(text)
            if match:
                formatted = self.normalize_identifier(
                    match.group(1), match.group(2), match.group(3), match.group(4)
                )
                if formatted and formatted not in identifier_bbox_map:
                    identifier_bbox_map[formatted] = bbox

        return identifier_bbox_map, texts_with_bboxes

    # ------------------------------------------------------------------
    # Extension handling with association
    # ------------------------------------------------------------------

    def process_extensions_with_associations(
        self,
        patch_path: Path,
        current_patch: np.ndarray,
        partials_info: List[dict],
        raw_masks: Dict[str, np.ndarray],
        img_h: int,
        img_w: int,
    ) -> List[Tuple[str, str, dict]]:
        """
        For partial identifiers found near patch edges, stitch in a strip from
        the neighbouring patch to recover the complete identifier (same logic as
        TargetedPatchExtractor.process_extensions), then associate each complete
        identifier with the nearest ISO line using the *current* patch's
        raw (undilated) color masks and the partial's *original* bbox.

        Returns list of (identifier, iso_line_name, color_info).
        """
        results: List[Tuple[str, str, dict]] = []

        for partial in partials_info:
            text      = partial['text']
            bbox      = partial['bbox']
            direction = partial['direction']

            neighbor_name = self.get_neighbor_patch(patch_path.name, direction)
            if not neighbor_name:
                continue
            neighbor_path = self.patch_folder / neighbor_name
            if not neighbor_path.exists():
                continue
            neighbor_patch = cv2.imread(str(neighbor_path))
            if neighbor_patch is None:
                continue

            print(f"    → Extending to {neighbor_name} ({direction}) for partial '{text}'")

            extended_region = self.create_targeted_extension(
                current_patch, neighbor_patch, bbox, direction,
                padding=40, lateral_margin=50,
            )
            if extended_region is None or extended_region.size == 0:
                continue

            try:
                complete_ids, _ = self.run_ocr_on_image(extended_region)
            except Exception as e:
                print(f"    ✗ OCR error on extension: {e}")
                continue

            for identifier in complete_ids:
                # Use the partial's original bbox for distance-based ISO line lookup
                iso_line = self.find_nearest_iso_line(bbox, raw_masks, img_h, img_w)
                if iso_line:
                    color_info = self.iso_color_map[iso_line]
                    results.append((identifier, iso_line, color_info))
                    print(
                        f"    ✓ [ext] {identifier} → {iso_line} "
                        f"({color_info.get('pattern', '?')}, {color_info['hex']})"
                    )

        return results

    # ------------------------------------------------------------------
    # Per-patch entry point
    # ------------------------------------------------------------------

    def process_patch_with_associations(
        self, patch_path: Path
    ) -> List[Tuple[str, str, dict]]:
        """
        Full pipeline for one patch:
          1. Color scan  →  skip if no colored lines
          2. Build proximity masks for legend colors present in patch
          3. Run OCR with bbox tracking
          4. Associate complete identifiers to nearest ISO line
          5. Handle edge-partial identifiers via extension + association

        Returns list of (identifier, iso_line_name, color_info).
        """
        patch = cv2.imread(str(patch_path))
        if patch is None:
            return []

        img_h, img_w = patch.shape[:2]
        img_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        # ── Step 1: fast color scan ──────────────────────────────────────
        if not self.has_colored_lines(img_rgb):
            print(f"  [SKIP] {patch_path.name}  (no colored pixels)")
            return []

        print(f"\n  [PROCESS] {patch_path.name}")

        # ── Step 2: build raw + proximity masks ──────────────────────────
        raw_masks, proximity_masks = self._build_color_masks(img_rgb)
        if not raw_masks:
            print(f"    No legend colors matched in patch")
            return []
        print(f"    Colors present: {list(raw_masks.keys())}")

        # ── Step 3: OCR with bbox map ────────────────────────────────────
        identifier_bbox_map, texts_with_bboxes = self.run_ocr_with_bbox_map(patch)

        associations: List[Tuple[str, str, dict]] = []
        seen_ids: Set[str] = set()

        # ── Step 4: associate complete identifiers (distance-based) ──────
        for identifier, bbox in identifier_bbox_map.items():
            iso_line = self.find_nearest_iso_line(bbox, raw_masks, img_h, img_w)
            if iso_line:
                color_info = self.iso_color_map[iso_line]
                associations.append((identifier, iso_line, color_info))
                seen_ids.add(identifier)
                print(
                    f"    ✓ {identifier} → {iso_line} "
                    f"({color_info.get('pattern', '?')}, {color_info['hex']})"
                )
            else:
                print(f"    ✗ {identifier} → no ISO line within range")

        # ── Step 5: partial identifiers at patch edges ───────────────────
        partials_info: List[dict] = []
        for text, bbox in texts_with_bboxes:
            if any(text in cid for cid in seen_ids):
                continue
            if self.has_partial_pattern(text):
                edge, direction = self.determine_edge_and_direction(bbox, img_w, img_h)
                if edge and direction:
                    partials_info.append({
                        'text':      text,
                        'bbox':      bbox,
                        'edge':      edge,
                        'direction': direction,
                    })

        if partials_info:
            ext_assoc = self.process_extensions_with_associations(
                patch_path, patch, partials_info,
                raw_masks, img_h, img_w,
            )
            # Only add if not already found as a direct complete match
            for identifier, iso_line, color_info in ext_assoc:
                if identifier not in seen_ids:
                    associations.append((identifier, iso_line, color_info))
                    seen_ids.add(identifier)

        return associations

    # ------------------------------------------------------------------
    # Per-page entry point
    # ------------------------------------------------------------------

    def process_page_with_associations(self, page_name: str) -> Dict:
        """
        Process every patch for `page_name`, collect all
        (identifier → ISO line) associations, deduplicate, and return
        a summary dict.
        """
        print("=" * 80)
        print(f"ISO LINE ASSOCIATION  —  PAGE: {page_name}")
        print("=" * 80)

        patch_files = sorted(self.patch_folder.glob(f"{page_name}_r*_c*.png"))
        print(f"\nFound {len(patch_files)} patches\n")

        all_associations: List[Tuple[str, str, dict]] = []
        processed = 0
        skipped   = 0

        for patch_path in patch_files:
            assoc = self.process_patch_with_associations(patch_path)
            if assoc:
                all_associations.extend(assoc)
                processed += 1
            else:
                skipped += 1

        # Deduplicate: keep first occurrence of each (identifier, iso_line) pair
        seen: Set[tuple] = set()
        unique_associations: List[Tuple[str, str, dict]] = []
        for identifier, iso_line, color_info in all_associations:
            key = (identifier, iso_line)
            if key not in seen:
                seen.add(key)
                unique_associations.append((identifier, iso_line, color_info))

        return {
            'page_name':        page_name,
            'total_patches':    len(patch_files),
            'processed_patches': processed,
            'skipped_patches':  skipped,
            'associations':     unique_associations,
            'total_found':      len(unique_associations),
        }


# ============================================================
# Entry point: ISO line association mode
# ============================================================

def main_with_legend():
    patch_folder = r"C:\Users\saroc\Desktop\P&ID V1 Updated\test-node1.4-patches"
    legend_path  = input("Enter path to ISO legend image: ").strip().strip("'\"")
    page_name    = input("Enter page name (e.g. ERCP_2): ").strip() or "ERCP_2"

    associator = ISOLineAssociator(patch_folder)
    associator.load_legend(legend_path)

    if not associator.iso_color_map:
        print("\nERROR: No legend colors loaded. Cannot proceed.")
        return

    results = associator.process_page_with_associations(page_name)

    # ── Print results ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL RESULTS  —  ISO IDENTIFIER → ISO LINE ASSOCIATIONS")
    print("=" * 80)
    print(f"\nPage             : {results['page_name']}")
    print(f"Total patches    : {results['total_patches']}")
    print(f"  Processed      : {results['processed_patches']}  (had colored lines)")
    print(f"  Skipped        : {results['skipped_patches']}  (no colored lines — OCR skipped)")
    print(f"\nTotal associations: {results['total_found']}\n")

    if results['associations']:
        col_w = 27
        print(f"{'ISO Identifier':<{col_w}} {'ISO Line':<22} {'Pattern':<10} Color")
        print("-" * 72)
        for identifier, iso_line, color_info in sorted(results['associations']):
            pattern   = color_info.get('pattern', 'unknown')
            hex_color = color_info['hex']
            print(f"{identifier:<{col_w}} {iso_line:<22} {pattern:<10} {hex_color}")
    else:
        print("  No associations found.")

    print("\n" + "=" * 80)
    return results


if __name__ == "__main__":
    check_cuda()
    print("\nSelect mode:")
    print("  1. Extract ISO identifiers only (original)")
    print("  2. Extract ISO identifiers + associate with ISO lines (new)")
    mode = input("Mode (1/2): ").strip()

    if mode == '2':
        main_with_legend()
    else:
        main()