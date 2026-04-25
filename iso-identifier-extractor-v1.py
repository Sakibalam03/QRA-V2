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
            r'(\d+(?:/\d+)?)\"\s*-\s*([A-Z0-9]{2})\s*-\s*(\d{4,6})(?:\s*-\s*([A-Z]\d{2}))?',
            re.IGNORECASE
        )

        # Strips a single spurious OCR character inserted between SIZE"- and the
        # 2-char code, e.g. "24"-JNG-8831-D48" → "24"-NG-8831-D48"
        # Only fires when exactly one extra char precedes a valid 2-letter code.
        self._spurious_infix_re = re.compile(
            r'(\d+(?:/\d+)?\")\s*-\s*[A-Z0-9]([A-Z]{2}\s*-\s*\d)',
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

    def _clean_identifier_text(self, text: str) -> str:
        """Remove a single spurious OCR character between SIZE\"- and the 2-char code.
        E.g. '24\"-JNG-8831-D48' → '24\"-NG-8831-D48'
             '24\"-1VG-8131-D48' → '24\"-VG-8131-D48'
        """
        return self._spurious_infix_re.sub(r'\1-\2', text)

    def normalize_identifier(self, size, code, number, suffix=None):
        """Normalize and fix OCR errors"""
        code = code.upper()

        # Fix OCR errors
        if code == '0W': code = 'OW'
        elif code == '0C': code = 'OC'
        elif code == 'C0': code = 'CO'
        elif code == 'CQ': code = 'CG'
        elif code == 'NC': code = 'NG'
        elif code == 'VG': code = 'NG'

        if not (len(code) == 2 and code.isalpha()):
            return None

        number = number.replace('O', '0').replace('I', '1')
        if not (4 <= len(number) <= 6 and number.isdigit()):
            return None

        if suffix:
            suffix = suffix.upper().replace('O', '0').replace('I', '1')
            if not (len(suffix) == 3 and suffix[0].isalpha() and suffix[1:].isdigit()):
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


def _colorful_pixels_in_region(
    img_rgb: np.ndarray, y1: int, y2: int, x1: int, x2: int
) -> np.ndarray:
    """Return array of colorful (non-grey, non-black) pixels in the given crop."""
    crop = img_rgb[y1:y2, x1:x2].reshape(-1, 3).astype(int)
    if len(crop) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    brightness = crop.mean(axis=1)
    color_diff = crop.max(axis=1) - crop.min(axis=1)
    mask = (brightness > 30) & (brightness < 248) & (color_diff > 20)
    return crop[mask].astype(np.uint8)


def _hue_cluster_median(colorful_pixels: np.ndarray, min_count: int = 6) -> Optional[Tuple]:
    """
    Given an array of colorful RGB pixels, find the dominant hue cluster
    (peak of a 36-bin hue histogram) and return the median RGB of that cluster.
    Returns None if fewer than min_count pixels fall in the peak cluster.
    """
    if len(colorful_pixels) < min_count:
        return None

    cp_f  = colorful_pixels.astype(np.float32) / 255.0
    r, g, b = cp_f[:, 0], cp_f[:, 1], cp_f[:, 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c + 1e-9

    hue = np.zeros(len(r))
    m = max_c == r; hue[m] = (60.0 * ((g[m] - b[m]) / delta[m])) % 360
    m = max_c == g; hue[m] =  60.0 * ((b[m] - r[m]) / delta[m]) + 120
    m = max_c == b; hue[m] =  60.0 * ((r[m] - g[m]) / delta[m]) + 240
    hue_cv = (hue / 2.0).astype(int) % 180   # OpenCV 0-180 scale

    hist, edges = np.histogram(hue_cv, bins=36, range=(0, 180))
    peak_bin    = int(np.argmax(hist))
    h_center    = (edges[peak_bin] + edges[peak_bin + 1]) / 2.0

    in_cluster = np.abs(hue_cv.astype(float) - h_center) <= 12
    if int(np.sum(in_cluster)) < min_count:
        return None

    return tuple(np.median(colorful_pixels[in_cluster], axis=0).astype(int))


def find_color_sample_column(
    img_rgb: np.ndarray,
    min_sat: int = 80,
    min_v: int = 40,
    max_v: int = 252,
    min_colorful_rows: int = 4,
    merge_gap: int = 8,
) -> Tuple[int, int]:
    """
    Locate the X column range in the legend image that contains the ISO line
    colour samples.

    The colour sample column (e.g. "Node Legend" in a GBN EPIC legend table)
    is the only column where MANY different rows each contain a distinct
    colourful horizontal segment.  All other columns are either:
      • Black text on white background  (not colourful)
      • A lightly-tinted header/HAZOP background  (low saturation, filtered
        out by min_sat=80 because those backgrounds are typically S ≈ 30-55)

    Algorithm:
      1. Build a per-X-column count of how many Y-rows have at least one
         pixel with S > min_sat and min_v < V < max_v.
      2. Find contiguous X-bands where that count >= min_colorful_rows.
      3. Return the widest such band.  Falls back to (0, image_width) if
         nothing is found.
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_img, w_img = img_hsv.shape[:2]

    s = img_hsv[:, :, 1]   # (h, w)
    v = img_hsv[:, :, 2]   # (h, w)
    colorful_mask = (s > min_sat) & (v > min_v) & (v < max_v)   # (h, w) bool

    # For each X column, how many rows contain at least one colourful pixel?
    row_hits = colorful_mask.any(axis=1)            # (h,) bool per row
    col_colorful_rows = colorful_mask.sum(axis=0)   # (w,) int — count per col

    active = col_colorful_rows >= min_colorful_rows  # (w,) bool

    # Merge small gaps so a dashed line doesn't split into many tiny blocks
    active_merged = active.copy()
    for i in range(w_img - merge_gap):
        if active[i] and active[i + merge_gap]:
            active_merged[i:i + merge_gap] = True

    # Collect contiguous blocks
    blocks: List[Tuple[int, int]] = []
    in_block = False
    bstart = 0
    for i in range(w_img):
        if active_merged[i] and not in_block:
            bstart, in_block = i, True
        elif not active_merged[i] and in_block:
            blocks.append((bstart, i))
            in_block = False
    if in_block:
        blocks.append((bstart, w_img))

    if not blocks:
        print("  [COL] No colourful column band found — using full width")
        return 0, w_img

    # Pick the widest block (most X columns with consistently colourful rows)
    blocks.sort(key=lambda b: b[1] - b[0], reverse=True)
    best_x1, best_x2 = blocks[0]
    print(f"  [COL] Colour-sample column detected: x={best_x1}-{best_x2} "
          f"({best_x2 - best_x1}px wide,  {int(col_colorful_rows[best_x1:best_x2].max())} peak rows)")
    return best_x1, best_x2


def dominant_color_in_row(
    img_rgb: np.ndarray,
    row_y1: int, row_y2: int,
    exclude_x1: int = 0,
    exclude_x2: int = 0,
    min_count: int = 6,
    min_density: float = 0.01,
) -> Optional[Tuple]:
    """
    Find the ISO line colour sample in a legend row by scanning the FULL
    row width.

    WHY no X exclusion:
      The legend is a table where the colored line sample is a GRAPHICAL
      element in a dedicated column.  The surrounding text is black, which
      is filtered out by _colorful_pixels_in_region (brightness and
      color_diff checks).  Restricting the X search range to outside the
      text bbox silently excludes the color sample when the OCR text bbox
      spans the full row width (which happens whenever the row text fills
      the image width).  Scanning the full width and relying on the
      colorfulness filter is simpler and more reliable.

    WHY tight Y bounds (no expansion):
      The legend rows are densely packed.  Expanding Y bleeds colorful
      pixels from adjacent rows into the current row's scan, causing
      colour swaps between entries (e.g. CPP-03 ↔ CPP-13).  Y bounds
      are kept strict: the caller passes (y1, y2) of the matched text
      line with zero expansion.
    """
    h_img, w_img = img_rgb.shape[:2]
    row_y1 = max(0, row_y1)
    row_y2 = min(h_img, row_y2)
    if row_y2 <= row_y1:
        return None

    cp = _colorful_pixels_in_region(img_rgb, row_y1, row_y2, 0, w_img)
    if len(cp) < min_count:
        return None

    total   = (row_y2 - row_y1) * w_img
    density = len(cp) / total if total > 0 else 0.0
    if density < min_density:
        return None

    color = _hue_cluster_median(cp, min_count=min_count)
    if color is not None:
        print(f"    [LEGEND] row y={row_y1}-{row_y2}  "
              f"colorful_px={len(cp)}  density={density:.3f}  "
              f"hex={rgb_to_hex(color)}")
    return color


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

            # Estimate x2 of the matched ISO name only (not the full line).
            # OCR text-line bboxes often span the ENTIRE table row
            # (ISO name + description + colour sample + Yes/No column), so
            # x2 ends up near the image midpoint and all four right-side
            # search regions start AFTER the colour-sample column.
            # Scaling x2 by the character-count ratio of the matched name
            # puts region-1 (x2_est+5 … x2_est+150) right over the sample.
            if len(text) > len(iso_name):
                char_end_ratio = (match.start() + len(iso_name)) / len(text)
                x2_est = int(x1 + char_end_ratio * (x2 - x1))
            else:
                x2_est = x2

            iso_boxes.append((iso_name, (x1, y1, x2_est, y2)))

    print(f"  Matched {len(iso_boxes)} ISO identifiers in legend\n")

    iso_color_map = {}
    if debug:
        debug_img = img.copy()

    for iso_name, (x1, y1, x2, y2) in iso_boxes:
        # ── Colour extraction: v4-style right-side directional search ─────
        #
        # For each ISO text bbox, try several search regions to the RIGHT
        # of the text and pick the one with the MOST colorful pixels.
        # This is exactly what iso-legends-extractor-v4 does and it reliably
        # finds the colour sample column regardless of legend layout.
        #
        # WHY not column-restriction:
        #   The OCR text bbox often spans the full row width, so a fixed
        #   column boundary derived from the "Node Legend" header can fall
        #   inside (or outside) the actual colour sample.  Scanning outward
        #   from x2 of the ISO text and picking the richest region is layout-
        #   agnostic and matches v4's proven approach.
        #
        # WHY strict Y bounds (zero expansion):
        #   Adjacent legend rows are densely packed.  Even ±6 px of Y
        #   expansion causes colourful pixels from the row above/below to
        #   bleed into the current scan.  The wide search region (w*0.5 →
        #   w-10) in particular spans both CPP-03 and CPP-13 rows when Y is
        #   loose, swapping their extracted colours.  Strict bounds (exactly
        #   y1..y2 from the OCR bbox) prevent this.
        # Search regions: 6 candidates covering the full width.
        # Region 1-2 start from the proportionally-estimated x2 of the ISO
        # name text, putting them right over the colour-sample column.
        # Regions 3-4 sweep the left/center (for entries whose sample is at
        # x < w/2 — e.g. CPP-09 whose crop shows blank in x(1920-3830)).
        # Regions 5-6 are the original right-half sweeps from v4.
        search_regions = [
            (x2 + 5,        min(x2 + 200, w)),   # just right of ISO name
            (x2 + 150,      min(x2 + 600, w)),   # wider right of ISO name
            (int(w * 0.25), int(w * 0.55)),       # left-center sweep
            (int(w * 0.35), int(w * 0.65)),       # center sweep
            (int(w * 0.50), w - 10),              # right half (v4)
            (int(w * 0.60), w - 10),              # rightmost 40% (v4)
        ]

        best_cp       = np.empty((0, 3), dtype=np.uint8)
        best_vivid_cp = np.empty((0, 3), dtype=np.uint8)
        best_region   = search_regions[0]
        best_score    = -1

        for sx1, sx2 in search_regions:
            sx1 = max(0, sx1)
            sx2 = min(w, sx2)
            if sx1 >= sx2:
                continue
            # Strict Y bounds: exactly y1..y2 — no expansion
            cp = _colorful_pixels_in_region(img_rgb, y1, y2, sx1, sx2)
            if len(cp) == 0:
                continue

            # Score by VIVID pixels (HSV S > 70).
            # HAZOP table backgrounds are pale (S ≈ 30-55) and score 0,
            # while actual ISO line samples (S ≥ 80-255) score high.
            # This prevents the wide sweep from winning due to many pale
            # background pixels when the colour sample is in a narrow band.
            cp_arr   = cp.reshape(1, -1, 3).astype(np.uint8)
            cp_hsv_r = cv2.cvtColor(cp_arr, cv2.COLOR_RGB2HSV).reshape(-1, 3)
            vivid_mask  = cp_hsv_r[:, 1] > 70
            vivid_count = int(np.sum(vivid_mask))
            # Primary sort: vivid pixels; tiebreaker: total colorful pixels
            score = vivid_count * 100_000 + len(cp)
            if score > best_score:
                best_score    = score
                best_cp       = cp
                best_vivid_cp = cp[vivid_mask]
                best_region   = (sx1, sx2)

        # ── Fallback pass: y-expanded search when strict bounds yield vivid=0 ──
        # Cause: OCR sometimes returns a bbox for a continuation/wrap line that
        # is BELOW the row containing the colour sample.  (CPP-09 example: the
        # description "Condensate from 1st Stage Separator to FSO2 export
        # pipeline" wraps across 2 table rows; the colour sample is in the top
        # row but the OCR detection y-range can fall on the lower continuation.)
        # Fix: when vivid=0 for ALL strict-bound regions, search the right-side
        # column (Node Legend) with gradually expanded y-bounds (upward first,
        # then ±N rows) until vivid pixels are found.
        if best_score < 5 * 100_000:   # vivid_count = 0 in every strict region
            row_h = max(y2 - y1, 20)
            right_side = [(int(w * 0.50), w - 10), (int(w * 0.60), w - 10)]
            for expand_rows in [1, 2, 3]:
                improved = False
                # Try above first (most common mismatch direction), then ±
                for exp_y1, exp_y2 in [
                    (max(0, y1 - expand_rows * row_h),  y2),          # extend up
                    (y1, min(h, y2 + expand_rows * row_h)),            # extend down
                    (max(0, y1 - expand_rows * row_h),
                     min(h, y2 + expand_rows * row_h)),                # both
                ]:
                    if exp_y2 <= exp_y1:
                        continue
                    for sx1_r, sx2_r in right_side:
                        sx1_r = max(0, sx1_r)
                        sx2_r = min(w, sx2_r)
                        if sx1_r >= sx2_r:
                            continue
                        cp_exp = _colorful_pixels_in_region(
                            img_rgb, exp_y1, exp_y2, sx1_r, sx2_r)
                        if len(cp_exp) == 0:
                            continue
                        cp_arr_exp  = cp_exp.reshape(1, -1, 3).astype(np.uint8)
                        cp_hsv_exp  = cv2.cvtColor(
                            cp_arr_exp, cv2.COLOR_RGB2HSV).reshape(-1, 3)
                        vivid_exp   = cp_hsv_exp[:, 1] > 70
                        vivid_cnt_e = int(np.sum(vivid_exp))
                        score_exp   = vivid_cnt_e * 100_000 + len(cp_exp)
                        if score_exp > best_score:
                            best_score    = score_exp
                            best_cp       = cp_exp
                            best_vivid_cp = cp_exp[vivid_exp]
                            best_region   = (sx1_r, sx2_r)
                            improved = True
                            print(f"    [LEGEND] {iso_name} y-expand ±{expand_rows}row "
                                  f"y({exp_y1}-{exp_y2}) x({sx1_r}-{sx2_r}) "
                                  f"vivid={vivid_cnt_e} px={len(cp_exp)}")
                if improved and best_score >= 5 * 100_000:
                    break   # found vivid pixels — stop expanding

        # Pass vivid pixels to hue clustering; fall back to all colorful if
        # the entry uses a genuinely pale colour (e.g. very light lavenders).
        pixels_for_color = best_vivid_cp if len(best_vivid_cp) >= 5 else best_cp
        avg_color_tuple  = _hue_cluster_median(pixels_for_color, min_count=5)

        if avg_color_tuple is not None:
            bx1, bx2 = best_region
            print(f"    [LEGEND] {iso_name:<15}  "
                  f"hex={rgb_to_hex(avg_color_tuple)}  "
                  f"vivid={len(best_vivid_cp)}  px={len(best_cp)}  "
                  f"region=x({bx1}-{bx2})")
            hsv_color      = rgb_to_hsv(avg_color_tuple)
            iso_normalized = re.sub(r'\s+', ' ', iso_name.upper())

            # ── Pattern detection on VIVID-pixel-narrowed crop ───────────────
            # Problem: the best_region can be 1900+ px wide (e.g. x1920-3830).
            # When detect_line_pattern samples rows across this full width, the
            # dashed-line segment (a few hundred pixels in one cell) is flanked
            # by 1000+ px of white space on each side.  Sample rows that land
            # outside the dash segment see only white → recorded as coverage=0
            # and are discarded, leaving too few samples to call 'dashed'.
            # Fix: narrow the crop to the x-range that actually contains vivid
            # (highly saturated) pixels — that's where the colour sample lives.
            full_crop = img_rgb[y1:y2, bx1:bx2]
            if len(best_vivid_cp) >= 5 and (bx2 - bx1) > 100:
                vsub_hsv       = cv2.cvtColor(full_crop, cv2.COLOR_RGB2HSV)
                vivid_col_mask = (vsub_hsv[:, :, 1] > 70).any(axis=0)
                vivid_row_mask = (vsub_hsv[:, :, 1] > 70).any(axis=1)
                if vivid_col_mask.any() and vivid_row_mask.any():
                    col_idxs   = np.where(vivid_col_mask)[0]
                    row_idxs   = np.where(vivid_row_mask)[0]
                    narrow_clo = max(0,                  col_idxs[0]  - 20)
                    narrow_chi = min(bx2 - bx1,          col_idxs[-1] + 20)
                    narrow_rlo = max(0,                  row_idxs[0]  - 5)
                    narrow_rhi = min(full_crop.shape[0], row_idxs[-1] + 5)
                    pattern_crop = full_crop[narrow_rlo:narrow_rhi + 1,
                                             narrow_clo:narrow_chi + 1]
                else:
                    pattern_crop = full_crop
            else:
                pattern_crop = full_crop
            pattern = detect_line_pattern(pattern_crop, iso_name)

            if save_color_regions:
                os.makedirs('color_regions', exist_ok=True)
                region_path = (
                    f'color_regions/'
                    f'{iso_normalized.replace("/", "_").replace(" ", "_")}.png'
                )
                cv2.imwrite(region_path,
                            cv2.cvtColor(full_crop, cv2.COLOR_RGB2BGR))

            iso_color_map[iso_normalized] = {
                'rgb':     avg_color_tuple,
                'hsv':     hsv_color,
                'hex':     rgb_to_hex(avg_color_tuple),
                'pattern': pattern,
                'vivid':   len(best_vivid_cp),
            }
        else:
            print(f"  ⚠ Could not extract colour for {iso_name} "
                  f"— row had no distinct colorful region")

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
        color_h_tol: int = 12,
        color_s_min_ratio: float = 0.40,
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
        # Identifiers vetoed by VERT-THROUGH in any patch of the current page.
        # A VERT-THROUGH rejection means the coloured pipe passes vertically
        # through the identifier bbox — the identifier doesn't label that pipe.
        # Populated during per-patch processing; reset at the start of each page.
        self._vert_through_vetoes: Set[str] = set()

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
        # Saturation ceiling: prevents a pale legend colour from matching vivid
        # lines that belong to a more saturated legend entry with the same hue.
        # e.g. CPP-13 (S=41) must NOT match the vivid OW lines (S≈150) whose
        # correct legend entry is CPP-03 (S=116).
        # Formula: allow up to max(target_s*1.4, target_s+50) so that:
        #   • pale entries (S≤64) have tight ceilings that exclude vivid pixels
        #   • vivid entries (S≥120) have generous ceilings (no effective cap)
        s_max = min(255, max(int(target_s * 1.4), target_s + 50))
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
                np.array([h_high, s_max, v_max]),
            )
        else:
            # Hue wraps around 0/180
            m1 = cv2.inRange(
                img_hsv,
                np.array([h_low, s_min, v_min]),
                np.array([179,   s_max, v_max]),
            )
            m2 = cv2.inRange(
                img_hsv,
                np.array([0,      s_min, v_min]),
                np.array([h_high, s_max, v_max]),
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

    # ------------------------------------------------------------------
    # Association logic  —  local-colour approach
    # ------------------------------------------------------------------

    def _find_iso_line_by_local_color(
        self,
        bbox:       tuple,
        img_rgb:    np.ndarray,
        img_h:      int,
        img_w:      int,
        polygon:    Optional[list] = None,
        identifier: Optional[str]  = None,
    ) -> Optional[str]:
        """
        Associate an ISO identifier with a legend entry by sampling the
        pixels in a small window centred on the identifier bbox, extracting
        the dominant color, and matching it to the closest legend entry by
        HSV distance.

        WHY this replaces the mask-based approach:
          The previous approach built patch-wide HSV masks for every legend
          color and found which mask was nearest to the identifier.  When
          multiple legend entries share the same hue family (e.g. CPP-01/03/13
          are all reddish), their masks overlap on the SAME line pixels →
          all equidistant → wrong alphabetical winner.

          Direct color measurement at the identifier location naturally
          discriminates CPP-03 (vivid coral, S≈116) from CPP-01 (pale salmon,
          S≈64) from CPP-13 (very pale peach, S≈41) because the HSV distance
          formula explicitly penalises saturation mismatches.

        Returns None if no colorful pixels are found near the identifier or
        the best legend match is too dissimilar.
        """
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(img_w, int(bbox[2]))
        y2 = min(img_h, int(bbox[3]))

        if x2 <= x1 or y2 <= y1:
            return None

        # ── Step 1: sample VIVID pixels in an expanding window ───────────
        # Only vivid pixels (HSV S > 70) are used for colour extraction.
        #
        # WHY vivid-only (not all colorful):
        #   The window around an identifier picks up BOTH line pixels AND pale
        #   white/grey background pixels.  Using all colorful pixels biases the
        #   hue-cluster median toward the pale background, lowering the apparent
        #   saturation.  For entries that differ primarily in saturation
        #   (e.g. CPP-01 S=81 vs CPP-03 S=116), a diluted S=82 median would
        #   incorrectly match CPP-01 even when a vivid CPP-03 line is adjacent.
        #   Using only vivid pixels isolates the actual line colour.
        #
        # WHY this also rejects false positives (e.g. 18"-NG-8113-D48):
        #   Identifiers that are NOT adjacent to any coloured line produce only
        #   pale background colorful pixels at every margin.  These have S < 70
        #   and therefore contribute no vivid pixels.  Requiring ≥ 8 vivid pixels
        #   to accept a match means far-away identifiers are correctly rejected.
        extracted_color: Optional[tuple] = None
        used_margin = 0
        dark_pipe_rejected = False
        bw_bbox = x2 - x1
        bh_bbox = y2 - y1
        # Determine true text orientation.
        # Prefer the polygon over the axis-aligned bbox: Surya OCR sometimes
        # normalises a 90°-rotated text label to a landscape axis-aligned bbox
        # even though the physical text region is tall and narrow.  The polygon
        # corner-points always describe the actual image footprint of the text.
        is_portrait = bh_bbox > bw_bbox * 2.0  # fallback: axis-aligned bbox
        if polygon is not None and len(polygon) >= 4:
            try:
                pts = np.array(polygon, dtype=float)
                # Average opposite edge pairs to get the true w/h of the text box
                edge01 = float(np.linalg.norm(pts[1] - pts[0]))
                edge12 = float(np.linalg.norm(pts[2] - pts[1]))
                edge23 = float(np.linalg.norm(pts[3] - pts[2]))
                edge30 = float(np.linalg.norm(pts[0] - pts[3]))
                true_w = (edge01 + edge23) / 2.0
                true_h = (edge12 + edge30) / 2.0
                is_portrait = true_h > true_w * 2.0
            except Exception:
                pass  # keep axis-aligned estimate on any error
        if is_portrait:
            print(f"      [LOCAL] portrait bbox ({x1},{y1})-({x2},{y2})"
                  f"  bw={bw_bbox}  bh={bh_bbox}"
                  f"  poly={'yes' if polygon else 'no'}")
        # Portrait bboxes use a stricter vivid-saturation threshold (90 vs 70).
        # Reason: a vertical text label sitting beside a CPP-03 line (S=116) can
        # pick up anti-aliased edge pixels (S≈82) at wider margins. S=82 is closer
        # to CPP-01 (S=81) than CPP-03 (S=116), causing a wrong match.
        # Raising the threshold to 90 excludes those edge pixels while keeping the
        # core CPP-03 pixels (S=116) and other well-saturated lines.
        s_thresh = 90 if is_portrait else 70
        p0_spurious = False  # set True when Phase 0 spurious-bypass fires → skip Phase 0b

        # ── Phase 0: arrow-based rejection ──────────────────────────────────
        # Run BEFORE Phase 1 so that identifiers whose leader arrow points to an
        # uncoloured pipe are excluded even when a coloured ISO line happens to
        # fall within the Phase-1 proximity window.
        #
        # Key differences from Phase 2:
        #   beyond_phase1_dist=0  — don't skip any vivid pixels; a short arrow to
        #                           a coloured line (< 25 px) must still be found.
        #   min_dark=10           — slightly tighter than Phase-2's 5 to reduce
        #                           false triggers from scattered instrument marks.
        #
        # This call is REJECTION-ONLY: if the arrow tip is coloured we still let
        # Phase 1 extract the precise legend colour; we only early-exit here when
        # the arrow clearly leads to an uncoloured region.
        p0_color, p0_arrow, p0_vivid_any = self._detect_and_follow_arrow(
            img_rgb, x1, y1, x2, y2, img_h, img_w,
            min_vivid_directed=(50 if is_portrait else 8),
            beyond_phase1_dist=0,
            # Portrait keeps min_dark=10 to avoid the ~28-vivid false-positive
            # documented in the Phase-2 comment.  Landscape uses 5 to catch
            # small arrowheads (e.g. a short leader with only 5-9 dark pixels).
            min_dark=(10 if is_portrait else 5),
            # Bidirectional check: for horizontal landscape arrows, require
            # vivid colored pixels in the OPPOSITE direction too — an on-line
            # identifier has the colored line extending both ways, but a branch
            # identifier at a junction has it only on the junction side.
            bidirectional_check=(not is_portrait),
        )
        # Track whether Phase 0 detected a directional dark-pixel cluster before
        # the bypass code can reset p0_arrow.  Used to gate portrait Phase 2:
        # when Phase 0 found an arrow (dark_px ≥ min_dark=10), the "arrow" is
        # usually from a nearby horizontal pipe, not a real leader — Phase 2 with
        # its looser min_dark=5 would re-detect the same spurious direction and
        # produce a false positive.  When Phase 0 found NO arrow (very few dark
        # pixels around the bbox), Phase 2 can legitimately find a genuine weak
        # short leader arrow (e.g. 8"-OW-8104-D03 with dark_px=8).
        p0_found_arrow = p0_arrow
        if p0_arrow and p0_color is None:
            if is_portrait:
                # For portrait identifiers Phase 0 can mis-fire: the dark pixels
                # that form the "arrow" are often from a nearby horizontal pipe,
                # not a real leader arrow.  Don't reject here — let Phase 1 handle it.
                # Phase 2 is disabled for portrait (see below) so only Phase 1 matters.
                print(f"      [ARROW-P0] portrait + arrow→uncoloured → skip rejection, let Phase 1 handle")
                p0_arrow = False  # reset so PIPE-EDGE / LOOP-INSIDE don't gate on it
            elif p0_vivid_any:
                # Vivid pixels exist within ARROW_MAX but are NOT in the detected
                # arrow direction.  This happens when the "arrow" is actually a
                # dark pipe segment near the bbox (not a real leader arrow) — the
                # centroid points toward the pipe, but the colored ISO line is
                # accessible from a different direction via Phase 1.
                # Example: 24"-NG-8191-D48 has dark pipe above → centroid UP, but
                # CPP-08 orange is to the left/right within Phase-1 range.
                print(f"      [ARROW-P0] landscape vivid-elsewhere → likely spurious arrow, let Phase 1 decide")
                p0_arrow = False    # reset so guards don't gate on it
                p0_spurious = True  # skip Phase 0b (same dark pixels, same false result)
            else:
                print(f"      [ARROW-P0] arrow→uncoloured → reject")
                return None

        # Phase 0b: re-try with a tighter ring (2–18 px, min_dark=3) to catch very
        # short leader arrows (2–4 px from bbox boundary) that sit inside the
        # standard ring_inner=5 exclusion zone and therefore escape Phase 0.
        # Only run when Phase 0 found no arrow and the identifier is landscape.
        if not p0_arrow and not is_portrait and not p0_spurious:
            p0b_color, p0b_arrow, _ = self._detect_and_follow_arrow(
                img_rgb, x1, y1, x2, y2, img_h, img_w,
                min_vivid_directed=8,
                beyond_phase1_dist=0,
                min_dark=3,
                ring_inner=2,
                bidirectional_check=True,
            )
            if p0b_arrow and p0b_color is None:
                print(f"      [ARROW-P0b] short arrow→uncoloured → reject")
                return None
            if p0b_arrow:
                p0_arrow = True   # arrow confirmed colored; let Phase 1 determine precise color

        # When p0_spurious=True (arrow detected toward uncoloured, vivid elsewhere),
        # Phase 0 already confirmed the real arrow direction is NOT toward the
        # coloured line.  The coloured line is "elsewhere" but only reachable at
        # margin=15/25 — meaning the identifier is not directly adjacent to it.
        # Restrict to margin=8 only so we don't pick up a distant background line.
        # (24"-NG-8191-D48, which IS on the coloured line, still finds it at margin=8.)
        phase1_margins = [8] if p0_spurious else [8, 15, 25]
        for margin in phase1_margins:   # Phase 1: tight proximity search
            if is_portrait:
                # Vertical text beside a vertical line → expand only L/R.
                # Horizontal lines above/below the text are NOT the target line.
                rx1 = max(0, x1 - margin)
                rx2 = min(img_w, x2 + margin)
                ry1 = y1
                ry2 = y2
            else:
                rx1 = max(0, x1 - margin)
                rx2 = min(img_w, x2 + margin)
                ry1 = max(0, y1 - margin)
                ry2 = min(img_h, y2 + margin)
            cp = _colorful_pixels_in_region(img_rgb, ry1, ry2, rx1, rx2)
            if len(cp) < 8:
                continue
            # Filter to vivid pixels only
            cp_arr  = cp.reshape(1, -1, 3).astype(np.uint8)
            cp_hsv  = cv2.cvtColor(cp_arr, cv2.COLOR_RGB2HSV).reshape(-1, 3)
            vivid   = cp[cp_hsv[:, 1] > s_thresh]
            if len(vivid) < 8:
                continue   # not enough vivid pixels yet — keep expanding
            # For portrait bboxes, verify that vivid pixels span vertically
            # across at least 30% of the bbox height.  A vertical line running
            # alongside the text will produce vivid pixels in most rows.  A
            # horizontal line that merely clips through the window produces vivid
            # pixels in only ~1–5 rows regardless of margin — reject that case
            # so that identifiers near horizontal lines (e.g. 24"-NG-8831-D48
            # near a horizontal CPP-08) are never falsely associated.
            if is_portrait:
                region      = img_rgb[ry1:ry2, rx1:rx2]
                region_hsv  = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
                row_has_vivid   = (region_hsv[:, :, 1] > s_thresh).any(axis=1)
                vivid_row_span  = int(row_has_vivid.sum())
                bbox_h          = y2 - y1
                row_ratio       = vivid_row_span / max(1, bbox_h)
                print(f"      [LOCAL] spread@margin={margin}: "
                      f"vivid_rows={vivid_row_span}/{bbox_h} ({row_ratio:.2f})")
                if row_ratio < 0.5:
                    continue  # horizontal line clip only — not a vertical adjacency
            color = _hue_cluster_median(vivid, min_count=8)
            if color is not None:
                extracted_color = color
                used_margin     = margin
                break

        # ── Dark-pipe proximity guard ─────────────────────────────────────
        # If a dark (uncoloured spur) pipe is closer to this bbox than
        # used_margin/2, the identifier labels that spur, not the coloured
        # ISO line found by Phase 1.  Applies to landscape always; for portrait
        # also apply but only when Phase 1 found the colour (used_margin ≤ 25),
        # not when Phase 2 matched a distant horizontal band (used_margin = 50).
        if (extracted_color is not None and used_margin >= 8
                and (not is_portrait or used_margin <= 25)):
            pad  = 20
            gx1  = max(0, x1 - pad);  gx2 = min(img_w, x2 + pad)
            gy1  = max(0, y1 - pad);  gy2 = min(img_h, y2 + pad)
            g_rgb = img_rgb[gy1:gy2, gx1:gx2]
            g_hsv = cv2.cvtColor(g_rgb, cv2.COLOR_RGB2HSV)
            dark_mask = (g_hsv[:, :, 2] < 50) & (g_hsv[:, :, 1] < 40)
            # Exclude bbox interior
            ry1b = y1 - gy1;  ry2b = y2 - gy1
            rx1b = x1 - gx1;  rx2b = x2 - gx1
            dark_mask[ry1b:ry2b, rx1b:rx2b] = False
            if dark_mask.any():
                ys_dp, xs_dp = np.where(dark_mask)
                dr_dp = np.maximum(0, np.maximum(ry1b - ys_dp, ys_dp - (ry2b - 1)))
                dc_dp = np.maximum(0, np.maximum(rx1b - xs_dp, xs_dp - (rx2b - 1)))
                dist_dp = np.sqrt(dr_dp.astype(float) ** 2 + dc_dp.astype(float) ** 2)
                far_enough = dist_dp >= 3.0   # skip text-bleed pixels
                if far_enough.any():
                    min_dp_dist = float(dist_dp[far_enough].min())
                    threshold   = used_margin / 2.0
                    print(f"      [LOCAL] dark_proxy: min_dp={min_dp_dist:.1f}px"
                          f"  used_margin={used_margin}  threshold={threshold:.1f}")
                    if min_dp_dist < threshold:
                        print(f"      [LOCAL] dark pipe closer than coloured line"
                              f" → spur label → reject")
                        extracted_color = None
                        dark_pipe_rejected = True

        # ── Vertical-pipe-through check ──────────────────────────────────────
        # For a landscape (horizontal) identifier, if the coloured ISO line runs
        # VERTICALLY and passes THROUGH the text bbox, Phase 1 finds vivid pixels
        # both ABOVE and BELOW the identifier in a narrow column range.  That
        # geometry means the pipe crosses the text — the identifier does not label
        # that pipe.  Detect: vivid pixels present above AND below the bbox, and
        # their column span is less than half the identifier width.
        if (extracted_color is not None and not is_portrait and used_margin == 8):
            pad_vt = 8
            gx1_vt = max(0, x1 - pad_vt); gx2_vt = min(img_w, x2 + pad_vt)
            gy1_vt = max(0, y1 - pad_vt); gy2_vt = min(img_h, y2 + pad_vt)
            vt_rgb = img_rgb[gy1_vt:gy2_vt, gx1_vt:gx2_vt]
            vt_hsv = cv2.cvtColor(vt_rgb, cv2.COLOR_RGB2HSV)
            vt_vivid = (vt_hsv[:, :, 1] > s_thresh) & (vt_hsv[:, :, 2] >= 60)
            ry1_vt = y1 - gy1_vt; ry2_vt = y2 - gy1_vt
            rx1_vt = x1 - gx1_vt; rx2_vt = x2 - gx1_vt
            vt_vivid[ry1_vt:ry2_vt, rx1_vt:rx2_vt] = False
            if vt_vivid.any():
                ys_vt, xs_vt = np.where(vt_vivid)
                vivid_above_vt = int((ys_vt < ry1_vt).sum())
                vivid_below_vt = int((ys_vt >= ry2_vt).sum())
                if vivid_above_vt > 20 and vivid_below_vt > 20:
                    col_span_vt = int(xs_vt.max()) - int(xs_vt.min())
                    id_w = x2 - x1
                    print(f"      [VERT-THROUGH?] above={vivid_above_vt} below={vivid_below_vt}"
                          f" col_span={col_span_vt} id_w={id_w}")
                    if col_span_vt < id_w * 0.5:
                        print(f"      [VERT-THROUGH] vertical ISO line through landscape"
                              f" identifier → not relevant → reject")
                        if identifier:
                            self._vert_through_vetoes.add(identifier)
                        extracted_color = None
                        dark_pipe_rejected = True

        # ── Left/right edge pipe scan ────────────────────────────────────────
        # After Phase 1 at margin=8: a dark spur pipe that starts at 1px from
        # the bbox left/right edge would produce many dark pixels in a thin
        # edge strip (proportional to identifier height).  Text edge bleed
        # produces < 3 scattered pixels.  A genuine pipe crossing the full
        # height would produce ≥ identifier_height/3 dark pixels.
        # Only applies to landscape identifiers where no Phase 0/0b arrow found.
        if (extracted_color is not None and not is_portrait
                and used_margin == 8 and not p0_arrow):
            identifier_h  = y2 - y1
            min_pipe_px   = max(8, identifier_h // 3)
            strip_w       = 2

            def _count_dark_strip(arr: np.ndarray) -> int:
                if arr.size == 0:
                    return 0
                hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
                return int(((hsv[:, :, 2] < 50) & (hsv[:, :, 1] < 40)).sum())

            identifier_w = x2 - x1
            lx1        = max(0, x1 - strip_w)
            left_dark  = _count_dark_strip(img_rgb[y1:y2, lx1:x1])
            rx2        = min(img_w, x2 + strip_w)
            right_dark = _count_dark_strip(img_rgb[y1:y2, x2:rx2])
            ty1        = max(0, y1 - strip_w)
            top_dark   = _count_dark_strip(img_rgb[ty1:y1, x1:x2])
            by2        = min(img_h, y2 + strip_w)
            bot_dark   = _count_dark_strip(img_rgb[y2:by2, x1:x2])

            if left_dark >= min_pipe_px or right_dark >= min_pipe_px:
                print(f"      [PIPE-EDGE] L={left_dark} R={right_dark}"
                      f" min_pipe_px={min_pipe_px} → dark pipe at edge → spur → reject")
                return None

            # Vivid-pixel directional check: if vivid pixels exist ONLY above the
            # bbox (none below, none within 4px), the label is sitting below a
            # colored line (inside a loop or spur junction) — not labeling it.
            pad_v = 8
            gx1v = max(0, x1 - pad_v); gx2v = min(img_w, x2 + pad_v)
            gy1v = max(0, y1 - pad_v); gy2v = min(img_h, y2 + pad_v)
            v_rgb = img_rgb[gy1v:gy2v, gx1v:gx2v]
            v_hsv = cv2.cvtColor(v_rgb, cv2.COLOR_RGB2HSV)
            v_vivid = (v_hsv[:, :, 1] > s_thresh) & (v_hsv[:, :, 2] >= 60)
            ry1v = y1 - gy1v; ry2v = y2 - gy1v
            rx1v = x1 - gx1v; rx2v = x2 - gx1v
            v_vivid[ry1v:ry2v, rx1v:rx2v] = False
            if v_vivid.any():
                ys_v, xs_v = np.where(v_vivid)
                dr_v = np.maximum(0, np.maximum(ry1v - ys_v, ys_v - (ry2v - 1)))
                dc_v = np.maximum(0, np.maximum(rx1v - xs_v, xs_v - (rx2v - 1)))
                dist_v = np.sqrt(dr_v.astype(float) ** 2 + dc_v.astype(float) ** 2)
                vivid_count_close = int((dist_v < 4).sum())
                vivid_above = int((ys_v < ry1v).sum())
                vivid_below = int((ys_v >= ry2v).sum())
                if vivid_above > 100 and vivid_below == 0 and vivid_count_close == 0:
                    print(f"      [LOOP-INSIDE] vivid only above ({vivid_above}px)"
                          f" → label inside loop / spur → reject")
                    return None

        if extracted_color is None and not dark_pipe_rejected and not (is_portrait and p0_found_arrow):
            # ── Phase 2: leader-arrow detection ──────────────────────────
            # Phase 1 found no coloured pixels within 25 px.  Try to detect a
            # leader arrow departing from the text bbox and sample the coloured
            # ISO line at its tip (up to 50 px away).
            # Portrait guard: skip Phase 2 when Phase 0 already detected a dark-
            # pixel directional cluster (p0_found_arrow=True).  That cluster is
            # typically from a nearby horizontal pipe, not a real leader — Phase 2
            # with its looser min_dark=5 would re-detect the same spurious direction
            # and produce a false positive.  If Phase 0 found NO arrow (p0_found_arrow
            # =False), Phase 2 may still find a genuine weak short leader (8"-OW-8104).
            arrow_color, arrow_detected, _ = self._detect_and_follow_arrow(
                img_rgb, x1, y1, x2, y2, img_h, img_w,
                min_vivid_directed=(50 if is_portrait else 8),
                arrow_max=50,
            )
            if not arrow_detected:
                return None   # no arrow found and Phase 1 also failed
            if arrow_color is None:
                return None   # arrow found but tip is uncoloured → exclude
            extracted_color = arrow_color
            used_margin = 50  # indicative: colour found via leader arrow

        if extracted_color is None:
            return None   # dark_pipe_rejected with no Phase 2 fallback

        ex_h, ex_s, ex_v = rgb_to_hsv(extracted_color)

        # ── Step 2: HSV distance to every legend entry ───────────────────
        # Hue: 3× weight — primary colour identity (0-90 range).
        # Saturation: 40/255 weight per unit — distinguishes pale vs vivid
        #   same-hue entries (CPP-01 S=64 vs CPP-03 S=116 vs CPP-13 S=41).
        # Value: 10/255 weight — minor brightness discriminator.
        candidates: List[Tuple[str, float]] = []
        for iso_name, info in self.iso_color_map.items():
            lg_h, lg_s, lg_v = info['hsv']
            dh = min(abs(int(ex_h) - int(lg_h)),
                     180 - abs(int(ex_h) - int(lg_h)))
            ds = abs(int(ex_s) - int(lg_s)) / 255.0
            dv = abs(int(ex_v) - int(lg_v)) / 255.0
            dist = dh * 3.0 + ds * 40.0 + dv * 10.0
            candidates.append((iso_name, dist))

        candidates.sort(key=lambda c: c[1])
        best_name, best_dist = candidates[0]

        top5 = "  ".join(f"{n}={d:.1f}" for n, d in candidates[:5])
        print(f"      [LOCAL] margin={used_margin}px  "
              f"hex={rgb_to_hex(extracted_color)}  "
              f"hsv=({ex_h},{ex_s},{ex_v})  top5: {top5}")

        # Reject if even the closest entry is too dissimilar
        if best_dist > 45.0:
            print(f"      [LOCAL] → no match (best dist {best_dist:.1f} > 45)")
            return None

        # ── Step 3: pattern + vivid tiebreaker ───────────────────────────
        # When top-2 entries are within 8 distance units (same-color entries
        # that differ only in pattern, e.g. CPP-02 dashed vs CPP-08 solid),
        # detect the local line pattern from a horizontal strip around the
        # identifier and prefer the entry whose legend pattern matches.
        # If pattern is ambiguous (multiple candidates share the same pattern
        # or pattern detection returns 'unknown'), use the vivid pixel count
        # from legend extraction as a secondary tiebreaker — more vivid pixels
        # means the legend entry had a larger/denser color sample, which is a
        # stronger signal for the "main" line vs a minor variant.
        if len(candidates) >= 2 and (candidates[1][1] - best_dist) < 25.0:
            cy_mid = (y1 + y2) // 2
            # Use a 40 px tall strip (±20 px) so the sample spans multiple
            # dash/gap cycles of dashed lines.  A 16 px strip (±8 px) can land
            # entirely on a solid segment between dashes and wrongly return
            # 'solid', causing the correct dashed legend entry to be bypassed.
            strip = img_rgb[
                max(0, cy_mid - 20) : min(img_h, cy_mid + 20),
                max(0, x1 - 80)    : min(img_w, x2 + 80),
            ]
            local_pattern = (detect_line_pattern(strip)
                             if strip.size > 0 else 'unknown')
            print(f"      [LOCAL] pattern_strip local='{local_pattern}'  "
                  + "  ".join(
                      f"{n}={self.iso_color_map.get(n,{}).get('pattern','?')}"
                      for n, _ in candidates[:5]
                  ))

            tied = [(name, dist) for name, dist in candidates
                    if dist - best_dist < 25.0]

            # When all tied candidates share the exact same HSV distance (spread
            # < 0.5 units), their colours are literally identical in the legend.
            # Pattern detection on the local strip is unreliable here because any
            # dashed pattern reading will arbitrarily prefer dashed-legend entries
            # even when the correct entry (CPP-08 solid, larger legend sample) is
            # present.  Use vivid count directly — a larger legend sample is a
            # stronger signal for the dominant line variant.
            tied_dists = [d for _, d in tied]
            dist_spread = (max(tied_dists) - min(tied_dists)) if tied_dists else 0.0
            if dist_spread < 0.5:
                winner = max(
                    tied,
                    key=lambda nc: self.iso_color_map.get(nc[0], {}).get('vivid', 0)
                )[0]
                if winner != best_name:
                    print(f"      [LOCAL] identical-color spread={dist_spread:.2f}"
                          f" → vivid tiebreaker → prefer {winner} over {best_name}")
                best_name = winner
            elif local_pattern != 'unknown':
                pattern_matches = [
                    (name, dist) for name, dist in tied
                    if self.iso_color_map.get(name, {}).get('pattern') == local_pattern
                ]
                if len(pattern_matches) == 1:
                    # Unambiguous — exactly one candidate matches local pattern
                    winner = pattern_matches[0][0]
                    if winner != best_name:
                        print(f"      [LOCAL] pattern '{local_pattern}'"
                              f" → prefer {winner} over {best_name}")
                    best_name = winner
                elif len(pattern_matches) > 1:
                    # Multiple candidates match same pattern → vivid tiebreaker
                    winner = max(
                        pattern_matches,
                        key=lambda nc: self.iso_color_map.get(nc[0], {}).get('vivid', 0)
                    )[0]
                    if winner != best_name:
                        print(f"      [LOCAL] pattern '{local_pattern}' tie"
                              f" → vivid tiebreaker → prefer {winner} over {best_name}")
                    best_name = winner
                # else: no candidate matches local pattern → keep color-dist winner
            else:
                # Pattern unknown → vivid tiebreaker among all tied candidates
                winner = max(
                    tied,
                    key=lambda nc: self.iso_color_map.get(nc[0], {}).get('vivid', 0)
                )[0]
                if winner != best_name:
                    print(f"      [LOCAL] pattern unknown"
                          f" → vivid tiebreaker → prefer {winner} over {best_name}")
                best_name = winner

        return best_name

    # ------------------------------------------------------------------
    # Leader-arrow detection  (Phase 2 of local-colour association)
    # ------------------------------------------------------------------

    def _detect_and_follow_arrow(
        self,
        img_rgb: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        img_h:  int,
        img_w:  int,
        min_vivid_directed: int = 8,
        beyond_phase1_dist: int = 25,
        min_dark: int = 5,
        bidirectional_check: bool = False,
        ring_inner: int = 5,
        arrow_max: int = 50,
    ) -> Tuple[Optional[tuple], bool]:
        """
        Detect a leader arrow departing from the identifier text bbox and
        return the median RGB of the coloured ISO line at its tip.

        Context
        -------
        Some P&ID identifiers are displaced from their ISO line and connected
        to it by a thin dark arrow (straight or L-shaped) that always points
        FROM the text TOWARD the line.  The arrowhead (small triangle) touches
        the coloured line.  Arrow length is typically 20–50 px; it can depart
        from any point on the text boundary.

        Algorithm
        ---------
        A. Scan an annular ring 5–18 px outside the text bbox for near-black
           pixels (V < 80, S < 60).  Their centroid relative to the bbox centre
           gives an estimate of the arrow departure direction.
        B. Expand to 50 px and keep only vivid (S > 70) pixels whose direction
           from the bbox centre is within ±60° (cos ≥ 0.5) of that arrow
           direction AND are more than 25 px from the bbox boundary (pixels
           inside Phase-1 range were already handled).
        C. If ≥ min_vivid_directed such vivid pixels exist, return their
           hue-cluster median RGB so the caller can match against the legend.
           If no vivid pixels lie in the arrow direction, the arrow points to an
           uncoloured line — return None so the identifier is excluded.

        Returns
        -------
        (color, True)  — arrow detected, tip is coloured; color is the RGB tuple
        (None,  True)  — arrow detected, tip is uncoloured → caller should reject
        (None,  False) — no arrow detected → caller should fall through to Phase 1
        """
        RING_INNER = ring_inner  # px outside bbox where ring starts (skip anti-alias bleed)
        RING_OUTER = 18    # px outside bbox where ring ends
        ARROW_MAX  = arrow_max  # px: maximum arrow length to search
        MIN_DARK   = min_dark   # minimum near-black pixels required to declare an arrow
        COS_THRESH = 0.50  # cos(60°) — vivid pixel must be within 60° of arrow dir

        # ── A: find near-black pixels in the ring ────────────────────────
        ry1 = max(0, y1 - RING_OUTER)
        ry2 = min(img_h, y2 + RING_OUTER)
        rx1 = max(0, x1 - RING_OUTER)
        rx2 = min(img_w, x2 + RING_OUTER)

        ring = img_rgb[ry1:ry2, rx1:rx2]
        if ring.size == 0:
            return None, False, False

        ring_hsv = cv2.cvtColor(ring, cv2.COLOR_RGB2HSV)
        # Near-black: dark value AND low saturation (avoids dark-coloured lines)
        near_black = (ring_hsv[:, :, 2] < 80) & (ring_hsv[:, :, 1] < 60)

        # Exclude the inner zone (bbox + RING_INNER border) to skip text pixels
        # and immediate anti-aliasing bleed
        ey1 = max(0,            y1 - RING_INNER - ry1)
        ey2 = min(ring.shape[0], y2 + RING_INNER - ry1)
        ex1 = max(0,            x1 - RING_INNER - rx1)
        ex2 = min(ring.shape[1], x2 + RING_INNER - rx1)
        near_black[ey1:ey2, ex1:ex2] = False

        dark_coords = np.argwhere(near_black)   # (row, col) in ring coords
        if len(dark_coords) < MIN_DARK:
            return None, False, False   # no arrow detected

        # ── B: arrow direction from dark-pixel centroid ──────────────────
        bbox_cy_r = (y1 + y2) / 2.0 - ry1   # bbox centre in ring coords
        bbox_cx_c = (x1 + x2) / 2.0 - rx1

        dir_r = float(dark_coords[:, 0].mean()) - bbox_cy_r
        dir_c = float(dark_coords[:, 1].mean()) - bbox_cx_c

        mag = (dir_r ** 2 + dir_c ** 2) ** 0.5
        if mag < 2.0:
            return None, False, False   # no clear directional bias

        arrow_dy = dir_r / mag   # row component (↓ positive)
        arrow_dx = dir_c / mag   # col component (→ positive)
        print(f"      [ARROW] detected dir=({arrow_dx:+.2f},{arrow_dy:+.2f})"
              f"  dark_px={len(dark_coords)}")

        # ── C: vivid pixels within ARROW_MAX in the arrow direction ──────
        ay1 = max(0, y1 - ARROW_MAX)
        ay2 = min(img_h, y2 + ARROW_MAX)
        ax1 = max(0, x1 - ARROW_MAX)
        ax2 = min(img_w, x2 + ARROW_MAX)

        arrow_region = img_rgb[ay1:ay2, ax1:ax2]
        if arrow_region.size == 0:
            return None, True, False   # arrow detected but region empty → treat as uncoloured

        ar_hsv    = cv2.cvtColor(arrow_region, cv2.COLOR_RGB2HSV)
        vivid_mask = ar_hsv[:, :, 1] > 70
        vivid_coords = np.argwhere(vivid_mask)   # (row, col) in arrow-region coords

        if len(vivid_coords) < 8:
            print(f"      [ARROW] no vivid pixels within {ARROW_MAX}px → reject")
            return None, True, False   # arrow detected, no vivid anywhere

        # Absolute coordinates of each vivid pixel
        pix_r = vivid_coords[:, 0].astype(float) + ay1
        pix_c = vivid_coords[:, 1].astype(float) + ax1

        # Direction vector from bbox centre to each vivid pixel
        bcr = (y1 + y2) / 2.0
        bcc = (x1 + x2) / 2.0
        vec_r = pix_r - bcr
        vec_c = pix_c - bcc
        pix_dist = np.sqrt(vec_r ** 2 + vec_c ** 2)

        # Distance from the nearest point on the bbox BOUNDARY (not centre).
        # Pixels inside beyond_phase1_dist px from the boundary are skipped
        # when called from Phase 2 (avoids double-counting Phase-1 range).
        # When called from Phase 0 (before Phase 1), pass 0 so all pixels count.
        nearest_r = np.clip(pix_r, y1, y2)
        nearest_c = np.clip(pix_c, x1, x2)
        dist_from_bbox = np.sqrt((pix_r - nearest_r) ** 2 + (pix_c - nearest_c) ** 2)
        beyond_phase1 = dist_from_bbox > beyond_phase1_dist

        # Cosine of angle between arrow direction and each pixel's direction
        cos_angle = (vec_r * arrow_dy + vec_c * arrow_dx) / np.maximum(pix_dist, 1.0)

        in_arrow_dir = beyond_phase1 & (cos_angle >= COS_THRESH)

        if not in_arrow_dir.any():
            print(f"      [ARROW] vivid pixels present but none in arrow direction"
                  f" → arrow points to uncoloured line → reject")
            return None, True, True   # arrow detected, vivid exists but not in direction

        directed_pixels = arrow_region[
            vivid_coords[in_arrow_dir, 0],
            vivid_coords[in_arrow_dir, 1],
        ]   # shape (N, 3) RGB

        if len(directed_pixels) < min_vivid_directed:
            return None, True, True   # too few directed vivid pixels → vivid exists

        # ── Bidirectional check (Phase-0 landscape only) ──────────────────
        # A horizontal arrow on an identifier that is ON a colored line points
        # along the line — vivid colored pixels exist in BOTH the arrow direction
        # AND the opposite direction (the line continues both ways).
        # A horizontal arrow on a BRANCH identifier points toward the junction;
        # the pipe in the opposite direction is uncolored → vivid pixels exist
        # ONLY in the arrow direction.  Detecting this asymmetry lets us reject
        # the branch case even though the arrow tip appears colored.
        if bidirectional_check and abs(arrow_dx) > 0.7:
            opp_cos = (vec_r * (-arrow_dy) + vec_c * (-arrow_dx)) / np.maximum(pix_dist, 1.0)
            # Adaptive limit: only look for opposite-direction vivid pixels within
            # (min_forward_dist + 10) px of the bbox boundary.  This prevents distant
            # incidental colored elements (from a different ISO line elsewhere in the
            # patch) from incorrectly "passing" the bidirectional check — they would
            # be much farther than the forward-direction pixels that hug the junction.
            if in_arrow_dir.any():
                min_fwd_dist  = float(dist_from_bbox[in_arrow_dir].min())
                opp_dist_limit = max(min_fwd_dist + 10.0, 15.0)
            else:
                opp_dist_limit = 20.0
            in_opp_dir = beyond_phase1 & (opp_cos >= COS_THRESH) & (dist_from_bbox <= opp_dist_limit)
            if not in_opp_dir.any():
                print(f"      [ARROW] horizontal arrow, no vivid opposite within"
                      f" {opp_dist_limit:.0f}px → branch at junction → reject")
                return None, True, True   # vivid exists in fwd direction at least

        color = _hue_cluster_median(directed_pixels, min_count=8)
        if color is not None:
            print(f"      [ARROW] tip hex={rgb_to_hex(color)}"
                  f"  vivid_in_dir={len(directed_pixels)}")
        return color, True, True

    # ------------------------------------------------------------------
    # OCR with bbox tracking
    # ------------------------------------------------------------------

    def run_ocr_with_bbox_map(self, img: np.ndarray):
        """
        Run Surya OCR on a BGR patch image.

        Returns:
            identifier_bbox_map    : {complete_identifier_str: bbox}
            identifier_polygon_map : {complete_identifier_str: polygon | None}
            texts_with_bboxes      : raw [(text, bbox), ...] from OCR
            bbox_polygon_map       : {tuple(bbox): polygon | None}
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        predictions = self.rec_predictor([img_pil], det_predictor=self.det_predictor)
        texts_with_bboxes = self.extract_text_from_predictions(predictions)

        # Build bbox → polygon lookup from raw predictions.
        # Surya provides a `polygon` attribute (list of 4 corner points) on each
        # TextLine.  This reflects the TRUE text region orientation even when the
        # axis-aligned bbox has been normalised to landscape for rotated text.
        bbox_polygon_map: Dict[tuple, Optional[list]] = {}
        for line in predictions[0].text_lines:
            ltext = line.text.strip()
            if ltext:
                poly = getattr(line, 'polygon', None)
                bbox_polygon_map[tuple(line.bbox)] = poly

        identifier_bbox_map:     Dict[str, tuple]          = {}
        identifier_polygon_map:  Dict[str, Optional[list]] = {}
        for text, bbox in texts_with_bboxes:
            match = self.complete_pattern.search(self._clean_identifier_text(text))
            if match:
                formatted = self.normalize_identifier(
                    match.group(1), match.group(2), match.group(3), match.group(4)
                )
                if formatted and formatted not in identifier_bbox_map:
                    identifier_bbox_map[formatted]    = bbox
                    identifier_polygon_map[formatted] = bbox_polygon_map.get(tuple(bbox))

        # ── Pairwise merging: recover identifiers split across adjacent OCR boxes
        # Surya OCR sometimes splits a single identifier into two adjacent text
        # boxes (e.g. "6"-OW-8601-D" + "03").  Sort boxes left-to-right and try
        # joining each nearby pair that shares the same text row.
        # Only merge boxes that are NOT already complete identifiers on their own
        # (prevents "6"-OW-8601-D03" + "0" → spurious "6"-OW-8601-D030").
        sorted_twb = sorted(texts_with_bboxes, key=lambda t: t[1][0])
        for i, (ti, bi) in enumerate(sorted_twb):
            # Skip if ti is itself a complete identifier — no need to extend it
            if self.complete_pattern.search(ti):
                continue
            for j in range(i + 1, min(i + 6, len(sorted_twb))):
                tj, bj = sorted_twb[j]
                # Must share the same row: vertical overlap ≥ 30% of taller box
                v_overlap = min(bi[3], bj[3]) - max(bi[1], bj[1])
                row_h     = max(bi[3] - bi[1], bj[3] - bj[1])
                if row_h == 0 or v_overlap < row_h * 0.3:
                    continue
                # Must be horizontally close (gap ≤ 50px, left box first)
                h_gap = bj[0] - bi[2]
                if h_gap < -10 or h_gap > 50:
                    continue
                combined_text = ti + tj
                m = self.complete_pattern.search(self._clean_identifier_text(combined_text))
                if m:
                    formatted = self.normalize_identifier(
                        m.group(1), m.group(2), m.group(3), m.group(4)
                    )
                    if formatted and formatted not in identifier_bbox_map:
                        merged_bbox = (
                            min(bi[0], bj[0]), min(bi[1], bj[1]),
                            max(bi[2], bj[2]), max(bi[3], bj[3]),
                        )
                        identifier_bbox_map[formatted]    = merged_bbox
                        identifier_polygon_map[formatted] = None  # no single polygon

        # ── Vertical pairwise merging: recover portrait identifiers split into
        # two stacked OCR boxes (common when text is rotated/vertical).
        # Collect boxes taller than wide (portrait-ish) that aren't yet complete.
        portrait_twb = [
            (t, b) for t, b in texts_with_bboxes
            if (b[3] - b[1]) > (b[2] - b[0]) * 1.5
            and not self.complete_pattern.search(t)
        ]
        portrait_twb.sort(key=lambda tb: ((tb[1][0] + tb[1][2]) / 2, tb[1][1]))
        for i, (ti, bi) in enumerate(portrait_twb):
            xi_ctr = (bi[0] + bi[2]) / 2
            for j in range(i + 1, min(i + 4, len(portrait_twb))):
                tj, bj = portrait_twb[j]
                # Must be in the same x-column (within 30px)
                if abs((bj[0] + bj[2]) / 2 - xi_ctr) > 30:
                    continue
                # Must be vertically close (gap ≤ 60px, no overlap)
                v_gap = bj[1] - bi[3]
                if v_gap < -10 or v_gap > 60:
                    continue
                for combined in (ti + tj, tj + ti):
                    m = self.complete_pattern.search(self._clean_identifier_text(combined))
                    if m:
                        formatted = self.normalize_identifier(
                            m.group(1), m.group(2), m.group(3), m.group(4)
                        )
                        if formatted and formatted not in identifier_bbox_map:
                            merged_bbox = (
                                min(bi[0], bj[0]), min(bi[1], bj[1]),
                                max(bi[2], bj[2]), max(bi[3], bj[3]),
                            )
                            identifier_bbox_map[formatted]    = merged_bbox
                            identifier_polygon_map[formatted] = None
                        break

        return (identifier_bbox_map, identifier_polygon_map,
                texts_with_bboxes, bbox_polygon_map)

    # ------------------------------------------------------------------
    # Extension handling with association
    # ------------------------------------------------------------------

    def process_extensions_with_associations(
        self,
        patch_path: Path,
        current_patch: np.ndarray,
        img_rgb: np.ndarray,
        partials_info: List[dict],
        img_h: int,
        img_w: int,
    ) -> List[Tuple[str, str, dict]]:
        """
        For partial identifiers found near patch edges, stitch in a strip from
        the neighbouring patch to recover the complete identifier, then
        associate via _find_iso_line_by_local_color using the partial's
        ORIGINAL bbox position in the current patch (the coloured line is
        always in the current patch, adjacent to where the text starts).

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
                complete_ids, ext_texts = self.run_ocr_on_image(extended_region)
            except Exception as e:
                print(f"    ✗ OCR error on extension: {e}")
                continue

            # Fallback: when extension OCR doesn't produce a complete identifier
            # by itself, try combining the original partial text with each text
            # line returned by the extension OCR.  This handles the common case
            # where the combined region is split across two boxes
            # (e.g. partial "3"-NG" in current patch + "-8980-D48" in extension
            # both appear as separate OCR lines instead of one merged string).
            if not complete_ids:
                # Try partial text combined with each ext_text
                for ext_text, _ in ext_texts:
                    for combined in (text + ext_text, ext_text + text):
                        m = self.complete_pattern.search(self._clean_identifier_text(combined))
                        if m:
                            f = self.normalize_identifier(
                                m.group(1), m.group(2), m.group(3), m.group(4)
                            )
                            if f:
                                complete_ids.add(f)
                                break

            if not complete_ids and len(ext_texts) >= 2:
                # Try all pairs of ext_texts against each other — handles the case
                # where the extension OCR splits the identifier across two boxes
                # (e.g. '24"-NG-8131' + '-D48' → '24"-NG-8131-D48')
                for i, (et_i, _) in enumerate(ext_texts):
                    for et_j, _ in ext_texts[i + 1:]:
                        for combined in (et_i + et_j, et_j + et_i):
                            m = self.complete_pattern.search(self._clean_identifier_text(combined))
                            if m:
                                f = self.normalize_identifier(
                                    m.group(1), m.group(2), m.group(3), m.group(4)
                                )
                                if f:
                                    complete_ids.add(f)
                                    break

            for identifier in complete_ids:
                # Associate using local color at the partial's original bbox
                # in the current patch (the line is here, not in the extension)
                iso_line = self._find_iso_line_by_local_color(
                    bbox, img_rgb, img_h, img_w,
                    polygon=partial.get('polygon'),
                    identifier=identifier,
                )
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
          1. Fast color scan  → skip patches with no coloured lines (saves OCR)
          2. OCR to detect ISO identifiers with their bounding boxes
          3. For each identifier: sample a small window of pixels around its
             bbox, extract the dominant colour, match to closest legend entry
             by HSV distance  (see _find_iso_line_by_local_color)
          4. Handle partial identifiers at patch edges via neighbour-extension
             OCR, then associate using the same local-colour approach

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

        # ── Step 2: OCR ──────────────────────────────────────────────────
        (identifier_bbox_map, identifier_polygon_map,
         texts_with_bboxes, bbox_polygon_map) = self.run_ocr_with_bbox_map(patch)

        associations: List[Tuple[str, str, dict]] = []
        seen_ids: Set[str] = set()

        # ── Step 3: associate complete identifiers ────────────────────────
        for identifier, bbox in identifier_bbox_map.items():
            polygon = identifier_polygon_map.get(identifier)
            iso_line = self._find_iso_line_by_local_color(
                bbox, img_rgb, img_h, img_w, polygon=polygon, identifier=identifier
            )
            if iso_line:
                color_info = self.iso_color_map[iso_line]
                associations.append((identifier, iso_line, color_info))
                seen_ids.add(identifier)
                print(
                    f"    ✓ {identifier} → {iso_line} "
                    f"({color_info.get('pattern', '?')}, {color_info['hex']})"
                )
            else:
                print(f"    ✗ {identifier} → no match")

        # ── Step 4: partial identifiers at patch edges ───────────────────
        partials_info: List[dict] = []
        for text, bbox in texts_with_bboxes:
            if any(text in cid for cid in seen_ids):
                continue
            if self.has_partial_pattern(text):
                edge, direction = self.determine_edge_and_direction(
                    bbox, img_w, img_h
                )
                if edge and direction:
                    partials_info.append({
                        'text':      text,
                        'bbox':      bbox,
                        'edge':      edge,
                        'direction': direction,
                        'polygon':   bbox_polygon_map.get(tuple(bbox)),
                    })

        if partials_info:
            ext_assoc = self.process_extensions_with_associations(
                patch_path, patch, img_rgb, partials_info, img_h, img_w,
            )
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

        # Reset per-page VERT-THROUGH vetoes before processing this page's patches
        self._vert_through_vetoes = set()

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

        # Apply cross-patch VERT-THROUGH vetoes: if VERT-THROUGH fired for an
        # identifier in ANY patch of this page, remove its association everywhere.
        # This overrides an earlier-patch Phase-1 acceptance that happened before
        # VERT-THROUGH could fire (the golden pipe wasn't visible in that patch).
        if self._vert_through_vetoes:
            before = len(unique_associations)
            unique_associations = [
                (ident, iso_line, color_info)
                for ident, iso_line, color_info in unique_associations
                if ident not in self._vert_through_vetoes
            ]
            removed = before - len(unique_associations)
            if removed:
                print(f"\n  [VETO] VERT-THROUGH cross-patch veto removed {removed} association(s):"
                      f" {sorted(self._vert_through_vetoes)}")

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
    import re
    import csv

    legend_path  = input("Enter path to ISO legend image: ").strip().strip("'\"")
    patch_folder = input("Enter path to multi-page patch folder: ").strip().strip("'\"")

    patch_folder_path = Path(patch_folder)
    if not patch_folder_path.is_dir():
        print(f"\nERROR: Folder not found: {patch_folder}")
        return

    # Auto-detect all page names from patch filenames (ERCP_{N}_r{row}_c{col}.png)
    page_names = set()
    for f in patch_folder_path.glob("*.png"):
        m = re.match(r'^(ERCP_\d+)_r\d+_c\d+\.png$', f.name)
        if m:
            page_names.add(m.group(1))

    if not page_names:
        print("\nERROR: No patch files matching ERCP_*_r*_c*.png found in folder.")
        return

    page_names = sorted(page_names, key=lambda x: int(x.split('_')[1]))
    print(f"\nDetected {len(page_names)} pages: {', '.join(page_names)}")

    associator = ISOLineAssociator(patch_folder)
    associator.load_legend(legend_path)

    if not associator.iso_color_map:
        print("\nERROR: No legend colors loaded. Cannot proceed.")
        return

    # Process each page and collect all associations with page info
    # Each entry: (iso_line, page_name, identifier, pattern, hex_color)
    all_entries: List[Tuple[str, str, str, str, str]] = []
    total_patches = 0
    total_processed = 0
    total_skipped = 0

    for page_name in page_names:
        results = associator.process_page_with_associations(page_name)
        total_patches   += results['total_patches']
        total_processed += results['processed_patches']
        total_skipped   += results['skipped_patches']
        for identifier, iso_line, color_info in results['associations']:
            pattern   = color_info.get('pattern', 'unknown')
            hex_color = color_info['hex']
            all_entries.append((iso_line, page_name, identifier, pattern, hex_color))

    # Deduplicate across pages: keep first occurrence of (iso_line, page, identifier)
    seen: set = set()
    unique_entries: List[Tuple[str, str, str, str, str]] = []
    for entry in all_entries:
        key = (entry[0], entry[1], entry[2])
        if key not in seen:
            seen.add(key)
            unique_entries.append(entry)

    # Sort: by ISO line, then by page number, then by identifier
    unique_entries.sort(key=lambda e: (e[0], int(e[1].split('_')[1]), e[2]))

    # ── Print results ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL RESULTS  —  ISO IDENTIFIER → ISO LINE ASSOCIATIONS")
    print("=" * 80)
    print(f"\nPages processed  : {len(page_names)}  ({page_names[0]} → {page_names[-1]})")
    print(f"Total patches    : {total_patches}")
    print(f"  Processed      : {total_processed}  (had colored lines)")
    print(f"  Skipped        : {total_skipped}  (no colored lines — OCR skipped)")
    print(f"\nTotal associations: {len(unique_entries)}\n")

    if unique_entries:
        col_w = 27
        print(f"{'ISO Line':<22} {'Page':<10} {'ISO Identifier':<{col_w}} {'Pattern':<10} Color")
        print("-" * 80)
        current_iso = None
        for iso_line, page_name, identifier, pattern, hex_color in unique_entries:
            if iso_line != current_iso:
                if current_iso is not None:
                    print()
                current_iso = iso_line
            print(f"{iso_line:<22} {page_name:<10} {identifier:<{col_w}} {pattern:<10} {hex_color}")
    else:
        print("  No associations found.")

    print("\n" + "=" * 80)

    # ── Write CSV ────────────────────────────────────────────────────────
    script_dir = Path(__file__).parent
    csv_path   = script_dir / "iso_associations.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["ISO Line", "Page", "ISO Identifier", "Pattern", "Color (Hex)"])
        for iso_line, page_name, identifier, pattern, hex_color in unique_entries:
            writer.writerow([iso_line, page_name, identifier, pattern, hex_color])

    print(f"\nCSV saved to: {csv_path}")
    return unique_entries


if __name__ == "__main__":
    check_cuda()
    main_with_legend()