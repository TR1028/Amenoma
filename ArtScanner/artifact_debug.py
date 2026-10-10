"""Debug dump of an artifact scan (``AMENOMA_DEBUG_ARTIFACT_CROP=1``), like the material crop dump.

Writes to ``artifacts/debug_artifact_scan/<timestamp>/``:

* ``layout.json``           window size, scale ratio, the artifact grid geometry, the detail panel box
                            and the OCR crop boxes (to check a calibration / ``artifact_layout.json``)
* ``page_NNN.png``          full window capture before each page is scanned, and
  ``page_NNN_overlay.png``  the same with the cards / click points / keypoint / label strips drawn
* ``NNNNNN_panel.png``      detail panel of every artifact, ``NNNNNN_overlay.png`` with the OCR crop
                            boxes (incl. lock / Astral Mark search boxes), ``NNNNNN_<field>.png`` the
                            crops fed to the OCR model
* ``debug_artifact_scan.jsonl``  one record per page / artifact: grid position, click point, raw and
                            corrected OCR result, status, warnings / errors

No Windows / TensorFlow imports here: the window capture function and the OCR model are passed in.
"""
import json
import os
import time

import numpy as np

import layout

ENV = 'AMENOMA_DEBUG_ARTIFACT_CROP'

STATUS_NAMES = {0: 'init', 1: 'skipped', 2: 'saved', 3: 'failed'}

BOX_COLORS = {'lock': (255, 64, 64), 'astral_mark': (255, 200, 0)}


def enabled():
    return os.environ.get(ENV) == '1'


def ocr_boxes(config, scale_ratio):
    """{field: box} of every ``*_coords`` attribute of an ``ocr.Config`` class, scaled."""
    boxes = {}
    for attr in sorted(vars(config)):
        if attr.endswith('_coords'):
            boxes[attr[:-len('_coords')]] = [c * scale_ratio for c in getattr(config, attr)]
    return boxes


def grid_dict(grid):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sorted(vars(grid).items())}


class ArtifactScanDebugDump(object):
    def __init__(self, root, game_info, grid, model, ocr_config, capture=None, logger=None):
        """
        :param root: base directory ('artifacts')
        :param game_info: art_scanner_logic.GameInfo (w, h, left, top, hwnd)
        :param grid: layout.GridLayout of the artifact inventory
        :param model: ocr.OCR / ocr_EN.OCR (analyze_panel, scale_ratio)
        :param ocr_config: ocr.Config / ocr_EN.Config
        :param capture: capture() -> PIL image of the whole client area (None: no page captures)
        """
        self.dir = os.path.join(root, 'debug_artifact_scan', time.strftime('%Y%m%d-%H%M%S'))
        os.makedirs(self.dir, exist_ok=True)
        self.jsonl = os.path.join(self.dir, 'debug_artifact_scan.jsonl')
        with open(self.jsonl, 'w', encoding='utf-8'):
            pass
        self.game_info = game_info
        self.grid = grid
        self.model = model
        self.ocr_config = ocr_config
        self.capture = capture
        self.logger = logger
        self.panel_box = layout.detail_panel_box(game_info.w, game_info.h)
        self.write_layout()

    def _log_exception(self, e):
        if self.logger is not None:
            self.logger.exception(e)

    def _append(self, record):
        with open(self.jsonl, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')

    def _path(self, name):
        return os.path.join(self.dir, name)

    def write_layout(self):
        try:
            s = self.model.scale_ratio
            data = {
                'window': {'w': self.game_info.w, 'h': self.game_info.h,
                           'left': getattr(self.game_info, 'left', None), 'top': getattr(self.game_info, 'top', None)},
                'scale_ratio': s,
                'grid': grid_dict(self.grid),
                'visible_rows': self.grid.visible_rows(0),
                'detail_panel_box': [round(v, 2) for v in self.panel_box],
                'ocr_boxes': {k: [round(v, 2) for v in box] for k, box in ocr_boxes(self.ocr_config, s).items()},
                'overrides': {'env': os.environ.get(layout.OVERRIDE_ENV), 'file': layout.OVERRIDE_FILE,
                              'values': layout.load_overrides()},
            }
            with open(self._path('layout.json'), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            self._log_exception(e)

    def page(self, page, rows, offset_y, last_page_info):
        """Called before a page is scanned (ArtScannerLogic.scanArtifacts(on_page=...))."""
        record = {'type': 'page', 'page': page, 'rows': list(rows), 'offset_y': round(offset_y, 2),
                  'last_page': last_page_info}
        try:
            if self.capture is not None:
                img = self.capture()
                name = 'page_%03d' % page
                img.save(self._path(name + '.png'))
                x, y0, _, y1 = self.grid.label_column_box()
                column = np.asarray(img.convert('RGB'))[y0:y1, x, :3]
                runs = layout.find_label_runs(column, y0)
                record['label_runs'] = runs
                record['detected'] = layout.detect_row_offset(self.grid, column, y0)
                overlay = layout.draw_grid_overlay(img, self.grid, offset_y, rows, self.panel_box, runs)
                overlay.save(self._path(name + '_overlay.png'))
                record['paths'] = {'page': name + '.png', 'overlay': name + '_overlay.png'}
        except Exception as e:
            self._log_exception(e)
            record['error'] = repr(e)
        self._append(record)

    def artifact(self, art_id, art_img, raw_info, final_info, scanner_info, status, warnings=None, error=None):
        prefix = '%06d' % art_id
        record = {'type': 'artifact', 'art_id': art_id, 'status': STATUS_NAMES.get(status, status),
                  'scanner': scanner_info, 'raw_detected_info': raw_info, 'final_detected_info': final_info,
                  'warnings': list(warnings or []), 'error': error}
        try:
            art_img.save(self._path(prefix + '_panel.png'))
            crops, unactivated = self.model.analyze_panel(art_img)
            for key, crop in sorted(crops.items()):
                crop.save(self._path('%s_%s.png' % (prefix, key)))
            boxes = ocr_boxes(self.ocr_config, self.model.scale_ratio)
            layout.draw_boxes(art_img, boxes, colors=BOX_COLORS).save(self._path(prefix + '_overlay.png'))
            record['crops'] = sorted(crops)
            record['unactivated_lines'] = unactivated
            record['scale_ratio'] = self.model.scale_ratio
        except Exception as e:
            self._log_exception(e)
            record['dump_error'] = repr(e)
        self._append(record)
