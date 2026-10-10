# -*- coding: utf-8 -*-
"""Comprehensive verification and layout overlay CLI for Genshin screenshots and detail panels.

Usage:
    python ArtScanner/Tools/verify_screenshot.py <screenshot.png> [options]

Options:
    --kind auto|artifact|material   Scanner grid type (default: auto)
    --lang CHS|EN                   Language for OCR and postprocessing (default: CHS)
    --out-dir DIR                   Output directory for overlay and JSON results (default: same as input)
    --layout PATH                   Path to custom artifact_layout.json overrides
    --offset OFFSET                 Manual row offset in client pixels (overrides auto-detection)
    --no-ocr                        Skip model loading and OCR inference (geometry overlay only)

Output:
    * <out-dir>/<stem>_overlay.png  Visual overlay with cards, clicks, keypoint, labels, and OCR boxes
    * <out-dir>/<stem>_result.json  Structured JSON containing geometry metadata and recognized item data
"""
import argparse
import ast
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ARTSCANNER_DIR = os.path.dirname(HERE)
if ARTSCANNER_DIR not in sys.path:
    sys.path.insert(0, ARTSCANNER_DIR)

import layout  # noqa: E402
import panel_detect  # noqa: E402
import postprocess  # noqa: E402


def _read_config_boxes(ocr_file: str) -> Dict[str, List[float]]:
    """Parse *_coords in class Config from ocr.py / ocr_EN.py via AST (no TF dependency)."""
    with open(ocr_file, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())
    boxes = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and ('Config' in node.name):
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    name = stmt.targets[0].id
                    if name.endswith('_coords'):
                        clean_name = name[:-len('_coords')]
                        boxes[clean_name] = list(ast.literal_eval(stmt.value))
    return boxes


def is_detail_panel_only(width: int, height: int) -> bool:
    """Check if the image is an isolated detail panel crop rather than a full game window."""
    aspect = width / float(height)
    # Detail panel reference aspect: 656 / 1119 ~ 0.586 (always tall, aspect < 0.8)
    return aspect < 0.85 and width < 1200


def analyze_screenshot(img_path: str,
                       kind: str = 'auto',
                       lang: str = 'CHS',
                       out_dir: Optional[str] = None,
                       layout_file: Optional[str] = None,
                       manual_offset: Optional[float] = None,
                       run_ocr: bool = True) -> Dict:
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    lang = lang.upper()
    stem = os.path.splitext(os.path.basename(img_path))[0]
    out_dir = out_dir or os.path.dirname(os.path.abspath(img_path)) or '.'
    os.makedirs(out_dir, exist_ok=True)

    overlay_path = os.path.join(out_dir, f"{stem}_overlay.png")
    result_path = os.path.join(out_dir, f"{stem}_result.json")

    is_panel = is_detail_panel_only(w, h)
    result = {
        "file": os.path.abspath(img_path),
        "width": w,
        "height": h,
        "is_detail_panel": is_panel,
        "lang": lang,
    }

    if is_panel:
        # Direct detail panel analysis
        s = w / 656.0
        extra_h = max(0.0, float(h) - 1119.0 * s)
        result["scale_ratio"] = round(s, 4)
        result["extra_height"] = round(extra_h, 2)

        ocr_file = os.path.join(ARTSCANNER_DIR, 'ocr.py' if lang == 'CHS' else 'ocr_EN.py')
        raw_boxes = _read_config_boxes(ocr_file)
        panel_boxes = layout.panel_ocr_boxes(0, 0, s, panel_height=h, coords_dict=raw_boxes)

        overlay_img = img.copy()
        draw_boxes_map = {k: v for k, v in panel_boxes.items() if not k.endswith('_coords')}
        overlay_img = layout.draw_boxes(
            overlay_img, draw_boxes_map,
            color=layout.PANEL_COLOR,
            colors={'lock': (255, 64, 64), 'astral_mark': (255, 200, 0), 'equipped': (0, 200, 255)}
        )
        overlay_img.save(overlay_path)

        panel_info = {
            "locked": panel_detect.detect_lock(img, raw_boxes.get('lock', [490, 480, 560, 560]), s),
            "astral_mark": panel_detect.detect_astral_mark(img, raw_boxes.get('astral_mark', [490, 480, 560, 560]), s),
        }
        result["panel_features"] = panel_info
        result["boxes"] = {k: [round(x, 1) for x in v] for k, v in panel_boxes.items()}

        if run_ocr:
            try:
                ocr_module = __import__('ocr' if lang == 'CHS' else 'ocr_EN')
                model_cls = getattr(ocr_module, 'OCR')
                model = model_cls()
                model.setScaleRatio(s)
                raw_data = model.detect_info(img)
                corrected_data = dict(raw_data)
                postprocess.correct_artifact(corrected_data, lang)
                result["ocr_raw"] = raw_data
                result["artifact"] = corrected_data
            except Exception as e:
                result["ocr_status"] = f"OCR skipped/failed: {e}"

    else:
        # Full game window analysis
        s = min(w / float(layout.REF_WIDTH), h / float(layout.REF_HEIGHT))
        aspect_ratio = f"{w}:{h}"
        extra_h = max(0.0, float(h) - layout.REF_HEIGHT * s)

        resolved_kind = kind
        if resolved_kind == 'auto':
            # Default to artifact scanner unless material is explicitly detected
            resolved_kind = 'artifact'

        overrides = layout.load_overrides(layout_file) if resolved_kind == 'artifact' else {}
        grid = layout.GridLayout(w, h, kind=resolved_kind, **overrides)
        panel_box = layout.detail_panel_box(w, h)

        # Detect label runs and row offset
        col_x, y0, _, y1 = grid.label_column_box()
        col_pixels = np.asarray(img)[y0:y1, col_x, :3]
        label_runs = layout.find_label_runs(col_pixels, y0)
        detected = layout.detect_row_offset(grid, col_pixels, y0)

        offset_y = manual_offset if manual_offset is not None else (detected[0] if detected is not None else 0.0)
        visible_rows = grid.visible_rows(offset_y)

        result.update({
            "kind": resolved_kind,
            "scale_ratio": round(s, 4),
            "aspect_ratio": aspect_ratio,
            "extra_height": round(extra_h, 2),
            "grid": {
                "rows": grid.rows,
                "cols": grid.cols,
                "first_x": round(grid.first_x, 1),
                "first_y": round(grid.first_y, 1),
                "card_width": round(grid.card_width, 1),
                "card_height": round(grid.card_height, 1),
                "pitch_x": round(grid.pitch_x, 1),
                "pitch_y": round(grid.pitch_y, 1),
                "keypoint": [round(grid.keypoint_x, 1), round(grid.keypoint_y, 1)],
                "detected_offset": round(detected[0], 2) if detected is not None else None,
                "active_offset": round(offset_y, 2),
                "visible_rows": visible_rows,
            },
            "panel_box": [round(x, 1) for x in panel_box],
            "label_runs": label_runs,
        })

        # Generate overlay image
        overlay_img = layout.draw_grid_overlay(
            img, grid, offset_y=offset_y, rows=visible_rows,
            panel_box=panel_box, label_runs=label_runs
        )

        ocr_file = os.path.join(ARTSCANNER_DIR, 'ocr.py' if lang == 'CHS' else 'ocr_EN.py')
        raw_boxes = _read_config_boxes(ocr_file)
        panel_h = panel_box[3] - panel_box[1]
        panel_boxes = layout.panel_ocr_boxes(panel_box[0], panel_box[1], s, panel_height=panel_h, coords_dict=raw_boxes)

        draw_boxes_map = {k: v for k, v in panel_boxes.items() if not k.endswith('_coords')}
        overlay_img = layout.draw_boxes(
            overlay_img, draw_boxes_map,
            color=layout.PANEL_COLOR,
            colors={'lock': (255, 64, 64), 'astral_mark': (255, 200, 0), 'equipped': (0, 200, 255)}
        )

        # If material kind, also draw amount boxes on each visible card
        if resolved_kind == 'material':
            amt_coords = [10, 175, 150, 203]  # relative to card top-left
            amt_boxes = {}
            for r in visible_rows:
                for c in range(grid.cols):
                    card_box = grid.item_box(r, c, offset_y)
                    amt_box = [card_box[0] + amt_coords[0] * s, card_box[1] + amt_coords[1] * s,
                               card_box[0] + amt_coords[2] * s, card_box[1] + amt_coords[3] * s]
                    amt_boxes[f"amt_r{r}c{c}"] = amt_box
            overlay_img = layout.draw_boxes(overlay_img, amt_boxes, color=(255, 180, 0))

        overlay_img.save(overlay_path)

        # Crop detail panel for feature detection
        panel_img = img.crop(panel_box)
        panel_info = {
            "locked": panel_detect.detect_lock(panel_img, raw_boxes.get('lock', [490, 480, 560, 560]), s),
            "astral_mark": panel_detect.detect_astral_mark(panel_img, raw_boxes.get('astral_mark', [490, 480, 560, 560]), s),
        }
        result["panel_features"] = panel_info

        if run_ocr and resolved_kind == 'artifact':
            try:
                ocr_module = __import__('ocr' if lang == 'CHS' else 'ocr_EN')
                model_cls = getattr(ocr_module, 'OCR')
                model = model_cls()
                model.setScaleRatio(s)
                raw_data = model.detect_info(panel_img)
                corrected_data = dict(raw_data)
                postprocess.correct_artifact(corrected_data, lang)
                result["ocr_raw"] = raw_data
                result["artifact"] = corrected_data
            except Exception as e:
                result["ocr_status"] = f"OCR skipped/failed: {e}"

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('screenshot', help='Path to game screenshot or detail panel image')
    p.add_argument('--kind', choices=('auto', 'artifact', 'material'), default='auto',
                   help='Inventory type (default: auto)')
    p.add_argument('--lang', choices=('CHS', 'EN'), default='CHS',
                   help='Language for OCR and postprocessing (default: CHS)')
    p.add_argument('--out-dir', default=None,
                   help='Output directory for overlay and JSON (default: same as input)')
    p.add_argument('--layout', default=None,
                   help='Path to custom artifact_layout.json overrides')
    p.add_argument('--offset', type=float, default=None,
                   help='Manual row offset in client pixels')
    p.add_argument('--no-ocr', action='store_true',
                   help='Skip OCR inference, only generate layout overlay')
    args = p.parse_args(argv)

    if not os.path.isfile(args.screenshot):
        sys.stderr.write(f"Error: Screenshot file not found: {args.screenshot}\n")
        return 1

    res = analyze_screenshot(
        args.screenshot,
        kind=args.kind,
        lang=args.lang,
        out_dir=args.out_dir,
        layout_file=args.layout,
        manual_offset=args.offset,
        run_ocr=not args.no_ocr,
    )

    print("=" * 60)
    print("AMENOMA SCREENSHOT VERIFICATION REPORT")
    print("=" * 60)
    print(f"File:        {res['file']}")
    print(f"Resolution:  {res['width']}x{res['height']} (Scale: {res.get('scale_ratio', 1.0)})")
    if res.get('is_detail_panel'):
        print("Type:        Detail Panel Crop")
    else:
        print(f"Type:        Full Game Window ({res.get('kind', 'artifact')})")
        grid = res.get('grid', {})
        print(f"Grid Layout: {grid.get('rows')} rows x {grid.get('cols')} cols")
        print(f"Row Offset:  {grid.get('active_offset')} px (Detected: {grid.get('detected_offset')})")
        print(f"Visible:     Rows {grid.get('visible_rows')}")

    pf = res.get('panel_features', {})
    print(f"Panel State: Locked={pf.get('locked')}, AstralMark={pf.get('astral_mark')}")

    if 'artifact' in res:
        art = res['artifact']
        print("-" * 60)
        print(f"Artifact:    {art.get('name')} (+{art.get('level')}) [{art.get('star')}★]")
        print(f"Set:         {art.get('set_name')} ({art.get('set_key')})")
        print(f"Slot:        {art.get('type')}")
        print(f"Main Stat:   {art.get('main_attr_name')} {art.get('main_attr_value')}")
        print(f"Equipped:    {art.get('equipped') or '(none)'}")
        sub_list = [f"{art.get(f'subattr_{i}')}" for i in range(1, 5) if art.get(f'subattr_{i}')]
        if sub_list:
            print(f"Substats:    {', '.join(sub_list)}")
        if art.get('unactivated_substats'):
            print(f"Unactivated: {', '.join(art['unactivated_substats'])}")

    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
