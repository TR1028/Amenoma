"""Draw the scanner's artifact grid geometry on a screenshot to check / calibrate it.

Take a screenshot of the game's client area (windowed mode, inventory -> artifacts, scrolled to the
top; or use a ``page_NNN.png`` of the ``AMENOMA_DEBUG_ARTIFACT_CROP=1`` dump) and run, from ArtScanner/:

    python Tools/artifact_layout_overlay.py screenshot.png [-o overlay.png] [--layout artifact_layout.json]

The overlay shows the cards that would be clicked (green) with their click points (red), the row
alignment keypoint (magenta), the label column (yellow) with the detected level label strips (orange),
the first row / bottom of the grid (cyan) and the detail panel with the OCR crop boxes (blue / red).
The detected row offset is printed: on a screenshot scrolled to the top it should be ~0; if the cards
do not line up, put the corrections into ``artifact_layout.json`` (keys: offset_y, lastrow_offset,
keypoint_dy, rows; reference pixels at 2560x1440), next to the exe or via $AMENOMA_ARTIFACT_LAYOUT.

Needs only Pillow and numpy (the OCR crop boxes are read from ocr.py without importing TensorFlow).
"""
import argparse
import ast
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import layout  # noqa: E402


def read_ocr_boxes(ocr_py):
    """{field: [x1, y1, x2, y2]} of the ``*_coords`` in ``class Config`` of ocr.py / ocr_EN.py (AST, no TF)."""
    with open(ocr_py, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    boxes = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'Config':
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    name = stmt.targets[0].id
                    if name.endswith('_coords'):
                        boxes[name[:-len('_coords')]] = list(ast.literal_eval(stmt.value))
    return boxes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('screenshot', help='client area screenshot of the game window')
    p.add_argument('-o', '--output', help='output image (default: <screenshot>_overlay.png)')
    p.add_argument('--layout', help='artifact_layout.json overrides (default: $%s or ./%s)'
                                    % (layout.OVERRIDE_ENV, layout.OVERRIDE_FILE))
    p.add_argument('--kind', choices=('artifact', 'material'), default='artifact')
    p.add_argument('--offset', type=float, default=None,
                   help='draw the rows shifted up by this many client pixels (default: detected / 0)')
    p.add_argument('--ocr', default=os.path.join(os.path.dirname(HERE), 'ocr.py'),
                   help='ocr.py / ocr_EN.py to read the panel crop boxes from')
    args = p.parse_args(argv)

    img = Image.open(args.screenshot).convert('RGB')
    w, h = img.size
    if args.kind == 'artifact':
        overrides = layout.load_overrides(args.layout)
        grid = layout.GridLayout.for_artifacts(w, h, overrides)
    else:
        overrides = {}
        grid = layout.GridLayout.for_materials(w, h)
    print(grid)
    if overrides:
        print('overrides:', overrides)

    x, y0, _, y1 = grid.label_column_box()
    column = np.asarray(img)[y0:y1, x, :3]
    runs = layout.find_label_runs(column, y0)
    detected = layout.detect_row_offset(grid, column, y0)
    print('label strips at y:', runs)
    if detected is None:
        print('no level label strip found in the label column (x=%d): wrong screen or layout' % x)
        offset, rows = 0.0, grid.visible_rows(0)
    else:
        offset, rows = detected
        print('detected row offset: %.1f px (%.1f px @2560x1440), fully visible rows: %s'
              % (offset, offset / grid.scale_ratio, rows))
    if args.offset is not None:
        offset, rows = args.offset, grid.visible_rows(args.offset)

    panel = layout.detail_panel_box(w, h)
    out = layout.draw_grid_overlay(img, grid, offset, rows, panel_box=panel, label_runs=runs)
    s = grid.scale_ratio
    boxes = {k: [panel[0 if i % 2 == 0 else 1] + c * s for i, c in enumerate(v)]
             for k, v in read_ocr_boxes(args.ocr).items()}
    out = layout.draw_boxes(out, boxes, color=layout.PANEL_COLOR,
                            colors={'lock': (255, 64, 64), 'astral_mark': (255, 200, 0)})
    output = args.output or os.path.splitext(args.screenshot)[0] + '_overlay.png'
    out.save(output)
    print('wrote', output)
    print(json.dumps({'width': w, 'height': h, 'rows': grid.rows, 'cols': grid.cols,
                      'first_y': round(grid.first_y, 1), 'detected_offset': None if detected is None
                      else round(detected[0], 1)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
