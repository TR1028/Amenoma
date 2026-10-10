"""Pixel heuristics for the artifact detail panel: lock state, Astral Mark and the 6.0 "unactivated"
(待激活) substat line.

All coordinates are in the panel reference frame used by ``ocr.Config`` (game resolution
2560x1440, i.e. ``scale_ratio == 1``); they are multiplied by ``scale_ratio`` like the OCR crops.

Pure numpy + PIL so that it can be unit tested without TensorFlow or a game window.

Colour references
-----------------
* active substat text: navy (73, 83, 102)  (the legacy presence check, +-25 per channel)
* locked: dark button (70, 80, 100) with a red padlock glyph (255, 137, 117)
  (Inventory_Kamera / Genshin Optimizer scanners)
* Astral Mark (5.2+): golden star on the active button; star colour ~ (255, 204, 50)
* unactivated substat (6.0+): neutral grey text, e.g. "暴击率+3.9%（待激活）"
* set name below the substats: green (93, 178, 87)
"""
import numpy as np

NAVY_TEXT = (73, 83, 102)
NAVY_TOLERANCE = 25


def scale_box(coords, scale_ratio):
    return [c * scale_ratio for c in coords]


def crop(img, coords, scale_ratio):
    return img.crop(scale_box(coords, scale_ratio))


def to_rgb_array(img):
    """(h, w, 3) float32 array from a PIL image or an array (alpha / extra channels dropped)."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return np.asarray(arr[..., :3], dtype=np.float32)


def min_pixels(scale_ratio, base):
    """Pixel-count threshold that scales with the captured area."""
    return max(6, int(round(base * float(scale_ratio) ** 2)))


# ----------------------------------------------------------------------------------- colour masks
def navy_text_mask(arr, tol=NAVY_TOLERANCE):
    return np.abs(arr - np.asarray(NAVY_TEXT, dtype=np.float32)).max(axis=-1) <= tol


def red_lock_mask(arr):
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return (r >= 190) & (g >= 90) & (g <= 185) & (b >= 60) & (b <= 170) & (r - g >= 55) & (r - b >= 65)


def golden_mask(arr):
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return (r >= 190) & (g >= 140) & (b <= 125) & (r - b >= 90) & (g - b >= 55)


def grey_text_mask(arr):
    spread = arr.max(axis=-1) - arr.min(axis=-1)
    mean = arr.mean(axis=-1)
    return (spread <= 28) & (mean >= 90) & (mean <= 200)


def green_text_mask(arr):
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return (g >= 120) & (g - r >= 35) & (g - b >= 35)


# ------------------------------------------------------------------------------------- detectors
def has_navy_text(line_img):
    """Legacy substat-line presence check (any pixel within +-25 of the navy text colour)."""
    return bool(navy_text_mask(to_rgb_array(line_img)).any())


def count_red_lock_pixels(img):
    return int(red_lock_mask(to_rgb_array(img)).sum())


def count_golden_pixels(img):
    return int(golden_mask(to_rgb_array(img)).sum())


def detect_lock(panel_img, coords, scale_ratio=1.0):
    """True if the red padlock glyph of the *locked* state is inside ``coords``."""
    region = crop(panel_img, coords, scale_ratio)
    return count_red_lock_pixels(region) >= min_pixels(scale_ratio, 40)


def detect_astral_mark(panel_img, coords, scale_ratio=1.0):
    """True if the golden star of an active Astral Mark is inside ``coords``."""
    region = crop(panel_img, coords, scale_ratio)
    return count_golden_pixels(region) >= min_pixels(scale_ratio, 40)


def is_unactivated_line(line_img, scale_ratio=1.0):
    """A substat slot without navy text that holds grey text (and is not the green set-name line)
    is the 6.0 "unactivated" substat."""
    arr = to_rgb_array(line_img)
    grey = int(grey_text_mask(arr).sum())
    green = int(green_text_mask(arr).sum())
    return grey >= min_pixels(scale_ratio, 60) and green < min_pixels(scale_ratio, 20)


def split_substat_lines(line_imgs, scale_ratio=1.0):
    """Decide which of the (up to 4) substat line crops are present.

    Lines are kept up to the first one without navy text (same as the v3.4 logic).  That first line
    is kept as well, and reported as unactivated, if it holds grey text.

    :return: (present, unactivated) - lists of 1-based line numbers
    """
    present, unactivated = [], []
    for number, line in enumerate(line_imgs, 1):
        if has_navy_text(line):
            present.append(number)
            continue
        if is_unactivated_line(line, scale_ratio):
            present.append(number)
            unactivated.append(number)
        break
    return present, unactivated
