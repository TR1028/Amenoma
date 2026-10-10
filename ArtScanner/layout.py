"""Inventory grid geometry for the artifact and the material scanner (pure python + numpy).

Everything here is in client-area pixels of the game window.  The reference numbers are
measured at 2560x1440 (``scale_ratio == 1``) and scale with
``scale_ratio = min(w / 2560, h / 1440)``, exactly like ``GameInfo.calculateCoordinates``.

Two grids share the same card size / gaps / columns but differ vertically:

* **material grid** - unchanged since 2.5.  ``GridLayout(w, h, 'material')`` reproduces the
  legacy ``GameInfo`` fields bit for bit (``first_art_y``, ``art_rows``, ``scroll_fin_keypt_*``,
  ``lastrow_offset``, ...), which the material scanner (and the ocr-fix debug tools) keep using.
* **artifact grid** - since 5.2 the artifact inventory has a filter bar above the grid and the
  whole grid moved down by 48.5 px @1600x900 (= ``ARTIFACT_GRID_OFFSET_Y`` = 77.6 px @2560x1440).
  Only fully visible rows are scanned (4 rows on 16:9, 5 on 16:10): with the old
  ``round()`` the half visible 5th row would be clicked.

Calibration: an optional JSON file (``artifact_layout.json`` in the working directory, or the
path in ``$AMENOMA_ARTIFACT_LAYOUT``) can override ``offset_y``, ``lastrow_offset``,
``keypoint_dy`` (reference px @2560x1440) and ``rows`` of the artifact grid.  Use
``AMENOMA_DEBUG_ARTIFACT_CROP=1`` / ``Tools/artifact_layout_overlay.py`` to check the result.
"""
import json
import math
import os

import numpy as np

REF_WIDTH, REF_HEIGHT = 2560, 1440

CARD_WIDTH = 164.39
CARD_HEIGHT = 203.21
CARD_EXPAND = 6
GAP_X = 30.89
GAP_Y = 30.35

MATERIAL_FIRST_Y = 161
# bottom of the visible grid on a 1440 px high window; anchored to the bottom of the window
GRID_BOTTOM = 1270
# beige level strip at the bottom of a card (233, 229, 220); sampled to detect row alignment
KEYPOINT_DX = 10
KEYPOINT_DY = 174  # == 335 - 161, relative to the top of the first row
LABEL_COLOR = (233, 229, 220)
LABEL_TOLERANCE = 5
# on the last page the list stops scrolling between two rows: rows sit this much higher
MATERIAL_LASTROW_OFFSET = 125.5

# 5.2+: filter bar above the artifact grid (+48.5 px @1600x900)
ARTIFACT_GRID_OFFSET_Y = 77.6
# Estimate only (the end of the list is anchored to the same bottom edge for both inventories, so
# the extra 77.6 px shift adds to the material value).  The artifact scanner measures the real
# offset on the last page from the label strips (detect_row_offset) and only falls back to this.
ARTIFACT_LASTROW_OFFSET = (MATERIAL_LASTROW_OFFSET + ARTIFACT_GRID_OFFSET_Y) % (CARD_HEIGHT + GAP_Y)

OVERRIDE_ENV = 'AMENOMA_ARTIFACT_LAYOUT'
OVERRIDE_FILE = 'artifact_layout.json'
OVERRIDE_KEYS = ('offset_y', 'lastrow_offset', 'keypoint_dy', 'rows')


def load_overrides(path=None):
    """Read the optional artifact layout calibration file; returns {} if there is none."""
    path = path or os.environ.get(OVERRIDE_ENV) or OVERRIDE_FILE
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    unknown = sorted(set(data) - set(OVERRIDE_KEYS) - {'comment'})
    if unknown:
        raise ValueError('unknown keys in %s: %s (allowed: %s)' % (path, unknown, list(OVERRIDE_KEYS)))
    return {k: data[k] for k in OVERRIDE_KEYS if data.get(k) is not None}


class GridLayout(object):
    def __init__(self, width, height, kind='artifact', offset_y=None, lastrow_offset=None,
                 keypoint_dy=None, rows=None):
        if kind not in ('artifact', 'material'):
            raise ValueError('kind must be "artifact" or "material", got %r' % (kind,))
        self.kind = kind
        self.width, self.height = width, height
        s = self.scale_ratio = min(width / REF_WIDTH, height / REF_HEIGHT)

        self.card_width = CARD_WIDTH * s
        self.card_height = CARD_HEIGHT * s
        self.card_expand = CARD_EXPAND * s
        self.gap_x = GAP_X * s
        self.gap_y = GAP_Y * s
        self.pitch_x = self.card_width + self.gap_x
        self.pitch_y = self.card_height + self.gap_y

        # == updated in 2.5 == (same expressions as GameInfo.calculateCoordinates)
        self.left_margin = (155 if width < 2 * height else 295.33) * s
        self.right_margin = (871.33 if width < 2 * height else 967.33) * s
        self.cols = int(math.floor((width - self.left_margin - self.right_margin + self.gap_x) / self.pitch_x))
        self.shift_x = ((width - self.left_margin - self.right_margin + self.gap_x)
                        - self.cols * self.pitch_x) / 2
        self.first_x = self.left_margin + self.shift_x

        if offset_y is None:
            offset_y = ARTIFACT_GRID_OFFSET_Y if kind == 'artifact' else 0
        if keypoint_dy is None:
            keypoint_dy = KEYPOINT_DY
        if lastrow_offset is None:
            lastrow_offset = ARTIFACT_LASTROW_OFFSET if kind == 'artifact' else MATERIAL_LASTROW_OFFSET
        self.offset_y_ref = offset_y
        self.offset_y = offset_y * s
        self.first_y = MATERIAL_FIRST_Y * s + self.offset_y
        self.keypoint_x = self.first_x + KEYPOINT_DX * s
        self.keypoint_y = (MATERIAL_FIRST_Y + keypoint_dy) * s + self.offset_y
        self.lastrow_offset = lastrow_offset * s

        self.grid_bottom = GRID_BOTTOM * s + height - REF_HEIGHT * s
        # number of (card + gap) units between the first row and the bottom of the grid
        self.rows_float = (self.grid_bottom - self.first_y + self.gap_y) / self.pitch_y
        # fully visible rows only (clicking a cut-off card auto-scrolls the game and cuts off amount labels)
        self.rows = int(math.floor(self.rows_float + 1e-6))
        self.incomplete_lastrow = False
        if rows:
            self.rows = int(rows)

    @classmethod
    def for_artifacts(cls, width, height, overrides=None):
        return cls(width, height, 'artifact', **(overrides or {}))

    @classmethod
    def for_materials(cls, width, height):
        return cls(width, height, 'material')

    def __repr__(self):
        return ('GridLayout(%s, %dx%d, s=%.4f, %d rows x %d cols, first=(%.1f, %.1f), keypoint=(%.1f, %.1f), '
                'lastrow_offset=%.1f)' % (self.kind, self.width, self.height, self.scale_ratio, self.rows,
                                          self.cols, self.first_x, self.first_y, self.keypoint_x,
                                          self.keypoint_y, self.lastrow_offset))

    # ------------------------------------------------------------------ coordinates
    def row_top(self, row, offset_y=0):
        return self.first_y + self.pitch_y * row - offset_y

    def click_point(self, row, col, offset_y=0):
        """Where the scanner clicks an item: horizontal centre, 1/5 below the top of the card."""
        x = self.first_x + self.pitch_x * col + self.card_width / 2
        y = self.first_y + self.pitch_y * row + self.card_height / 5
        if offset_y:
            y -= offset_y
        return x, y

    def item_box(self, row, col, offset_y=0):
        x1 = self.first_x + self.pitch_x * col
        y1 = self.row_top(row, offset_y)
        return x1, y1, x1 + self.card_width, y1 + self.card_height

    def selection_probe(self, center_x, center_y):
        """1.5 px box left of the card that turns bright when the card is selected."""
        x = center_x - self.card_width / 2 - self.card_expand
        return x, center_y, x + 1.5, center_y + 1.5

    def keypoint_box(self):
        return self.keypoint_x, self.keypoint_y, self.keypoint_x + 1.5, self.keypoint_y + 1.5

    def first_item_probe(self):
        x = self.first_x + self.card_width / 2
        return x - 1, self.first_y + self.card_height / 2, x + 1, self.first_y + self.card_height

    def label_column_box(self):
        """Integer client box of the pixel column through the label strips of the first column."""
        x = int(round(self.keypoint_x))
        y0 = int(max(0, math.floor(self.first_y - self.gap_y)))
        y1 = int(min(self.height, math.ceil(self.grid_bottom + self.gap_y)))
        return x, y0, x + 2, y1

    def visible_rows(self, offset_y=0):
        """Rows (in the frame shifted up by ``offset_y``) whose card is completely inside the grid."""
        tol = 2 * self.scale_ratio
        rows, row = [], 0
        while True:
            top = self.row_top(row, offset_y)
            if top + self.card_height > self.grid_bottom + tol:
                break
            if top >= self.first_y - tol:
                rows.append(row)
            row += 1
        return rows


def is_label_pixel(pixel, tol=LABEL_TOLERANCE):
    return all(abs(int(p) - c) <= tol for p, c in zip(pixel[:3], LABEL_COLOR))


def find_label_runs(column, y0, tol=LABEL_TOLERANCE, min_len=3):
    """Runs of label coloured pixels in an (N, 3[+1]) pixel column starting at client y ``y0``.

    :return: list of (start_y, end_y) client coordinates, end exclusive
    """
    col = np.asarray(column, dtype=np.int32).reshape(len(column), -1)[:, :3]
    mask = np.all(np.abs(col - np.array(LABEL_COLOR)) <= tol, axis=1)
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                runs.append((y0 + start, y0 + i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        runs.append((y0 + start, y0 + len(mask)))
    return runs


def detect_row_offset(grid, column, y0, keypoint_is_label=None, tol=LABEL_TOLERANCE):
    """Measure how far the grid is scrolled away from the row aligned layout.

    The bottom edge of each card's beige level strip is the bottom edge of the card.  From the
    label runs found in the pixel column (see ``GridLayout.label_column_box``) this returns
    ``(offset_y, rows)``: row ``j`` currently has its top at ``grid.row_top(j, offset_y)`` and
    ``rows`` are the fully visible row indices in that frame.  ``None`` if no label was found.

    ``offset_y`` is in ``[-0.15 pitch, 0.85 pitch)``; when the keypoint is known to be on a label
    (= the scanner counted the last row passage) offsets close to a full pitch are taken as small
    negative ones, matching how ``ArtScannerLogic.scrollToRow`` counts rows.
    """
    s = grid.scale_ratio
    column = np.asarray(column)
    runs = find_label_runs(column, y0, tol=tol, min_len=max(3, int(round(12 * s))))
    column_end = y0 + len(column)
    # a label strip is a short band at the bottom of the card; drop longer beige areas and runs
    # truncated by the end of the captured column (their bottom edge is unknown)
    runs = [(a, b) for a, b in runs if b - a <= 0.5 * grid.card_height and b < column_end]
    if not runs:
        return None
    # labels clipped by the top / bottom edge of the grid are shorter than the others
    full_len = max(b - a for a, b in runs)
    runs = [(a, b) for a, b in runs if b - a >= 0.7 * full_len]

    pitch = grid.pitch_y
    tops = [b - grid.card_height for _, b in runs]
    rel = [(grid.first_y - top) % pitch for top in tops]
    ref = rel[0]
    unwrapped = [ref + ((r - ref + pitch / 2) % pitch) - pitch / 2 for r in rel]
    offset = float(np.median(unwrapped)) % pitch
    if keypoint_is_label is not None:
        if keypoint_is_label and offset > pitch / 2:
            offset -= pitch
    elif offset > 0.85 * pitch:
        offset -= pitch

    # rows that have items (a label strip) and whose card is completely inside the grid
    exists = set(int(round((top + offset - grid.first_y) / pitch)) for top in tops)
    rows = [row for row in grid.visible_rows(offset) if row in exists]
    return offset, rows


def detail_panel_box(width, height):
    """Client box of the item detail panel on the right (same as ``GameInfo.art_info_*``).

    On aspect ratios taller than 16:9 (16:10, 4:3), the right panel stretches vertically to the
    bottom of the window: ``extra_h = max(0, height - REF_HEIGHT * s)``.
    """
    s = min(width / REF_WIDTH, height / REF_HEIGHT)
    extra_h = max(0.0, float(height) - REF_HEIGHT * s)
    info_margin = 0 if width < 2 * height else 96 * s
    left = width - 816 * s - info_margin
    top = 160 * s
    return left, top, left + 656 * s, top + 1119 * s + extra_h


def panel_ocr_boxes(panel_left, panel_top, scale_ratio, panel_height=None, coords_dict=None):
    """Calculate client pixel bounding boxes for panel OCR crops, anchoring bottom fields (equipped)."""
    s = float(scale_ratio)
    ref_panel_h = 1119.0 * s
    extra_h = max(0.0, float(panel_height) - ref_panel_h) if panel_height is not None else 0.0
    boxes = {}
    if coords_dict:
        for name, coords in coords_dict.items():
            c = list(coords)
            if name == 'equipped' and extra_h > 0:
                boxes[name] = [panel_left + c[0] * s, panel_top + c[1] * s + extra_h,
                               panel_left + c[2] * s, panel_top + c[3] * s + extra_h]
            else:
                boxes[name] = [panel_left + c[0] * s, panel_top + c[1] * s,
                               panel_left + c[2] * s, panel_top + c[3] * s]
    return boxes


# ---------------------------------------------------------------------------------- debug drawing
GRID_COLOR = (0, 255, 255)
CARD_COLOR = (0, 255, 0)
CLICK_COLOR = (255, 0, 0)
KEYPOINT_COLOR = (255, 0, 255)
LABEL_COLUMN_COLOR = (255, 255, 0)
LABEL_RUN_COLOR = (255, 128, 0)
PANEL_COLOR = (0, 128, 255)


def draw_grid_overlay(img, grid, offset_y=0, rows=None, panel_box=None, label_runs=None):
    """Copy of a window capture with the scanner geometry drawn on it (debug dump / calibration).

    green: cards that will be clicked, red cross: click points, magenta: keypoint, yellow: label
    column (orange: label strips found in it), cyan: first row top / grid bottom, blue: panel.
    """
    from PIL import ImageDraw
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    r = max(2.0, 5 * grid.scale_ratio)
    if rows is None:
        rows = grid.visible_rows(offset_y)
    draw.line([(0, grid.first_y), (img.width, grid.first_y)], fill=GRID_COLOR)
    draw.line([(0, grid.grid_bottom), (img.width, grid.grid_bottom)], fill=GRID_COLOR)
    for row in rows:
        for col in range(grid.cols):
            draw.rectangle(grid.item_box(row, col, offset_y), outline=CARD_COLOR)
            x, y = grid.click_point(row, col, offset_y)
            draw.line([(x - r, y), (x + r, y)], fill=CLICK_COLOR, width=2)
            draw.line([(x, y - r), (x, y + r)], fill=CLICK_COLOR, width=2)
        x1, y1, _, _ = grid.item_box(row, 0, offset_y)
        draw.text((x1 + 3, y1 + 3), 'r%d' % row, fill=CARD_COLOR)
    kx, ky = grid.keypoint_x, grid.keypoint_y
    draw.ellipse([kx - r, ky - r, kx + r, ky + r], outline=KEYPOINT_COLOR, width=2)
    x0, y0, _, y1 = grid.label_column_box()
    draw.line([(x0, y0), (x0, y1)], fill=LABEL_COLUMN_COLOR)
    for a, b in label_runs or []:
        draw.line([(x0 + 3, a), (x0 + 3, b)], fill=LABEL_RUN_COLOR, width=3)
    if panel_box is not None:
        draw.rectangle(panel_box, outline=PANEL_COLOR, width=2)
    return img


def draw_boxes(img, boxes, color=(255, 0, 0), colors=None):
    """Copy of ``img`` with labelled boxes ``{name: (x1, y1, x2, y2)}`` drawn on it."""
    from PIL import ImageDraw
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    for name, box in boxes.items():
        c = (colors or {}).get(name, color)
        draw.rectangle(list(box), outline=c, width=2)
        draw.text((box[0] + 2, box[1] + 1), name, fill=c)
    return img

