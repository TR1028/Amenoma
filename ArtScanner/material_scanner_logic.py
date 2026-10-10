# -*- coding: utf-8 -*-
import os
import time
from typing import Tuple

import art_scanner_logic
import layout
import mouse
import numpy as np
from utils import captureWindow


def getMaterialItemCoord(game_info, row: int, col: int, lastrow=False, offset_y: float = 0) -> Tuple[int, int, int, int]:
    """Client coordinates of the material item card at (row, col).

    Backward-compatible with legacy callers and debug_crop_from_saved_image.py.
    """
    grid = getattr(game_info, 'material_grid', None) or layout.GridLayout.for_materials(game_info.w, game_info.h)
    off_y = offset_y if offset_y != 0 else (grid.lastrow_offset if lastrow else 0)
    box = grid.item_box(row, col, off_y)
    return int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3]))


class MaterialScannerLogic:
    def __init__(self, game_info: art_scanner_logic.GameInfo):
        # The material inventory keeps the legacy geometry (no filter bar above the grid)
        grid = getattr(game_info, 'material_grid', None) or layout.GridLayout.for_materials(game_info.w, game_info.h)
        self.scanner = art_scanner_logic.ArtScannerLogic(game_info, grid=grid)
        self.page = 0

    def interrupt(self):
        self.scanner.interrupt()

    def getItemCenter(self, row: int, col: int, offset_y: float = 0):
        return self.scanner.getArtCenter(row, col, offset_y)

    def scanMora(self):
        """scan Mora and Primogems"""
        pass

    def clickCDIButton(self):
        """click Character Development Items button"""
        mouse.move(*self.scanner.game_info.cdi_button)
        mouse.click()
        time.sleep(0.5)

    def clickMaterialButton(self):
        """click Material button"""
        mouse.move(*self.scanner.game_info.m_button)
        mouse.click()
        time.sleep(0.5)

    def scanRows(self, rows, callback, offset_y: float = 0, lastrow: bool = False, debug: bool = False) -> bool:
        """Scan a set of rows in the material inventory.

        Crucial fix: offset_y is applied synchronously to BOTH the click point (getItemCenter)
        and the item crop box (getItemCoord), preventing click/crop desynchronization on the last page.
        """
        debug_material_crop = debug or os.environ.get("AMENOMA_DEBUG_MATERIAL_CROP") == "1"

        if lastrow and offset_y == 0:
            offset_y = self.scanner.grid.lastrow_offset

        rows = list(rows)
        if len(rows) < 1:
            return True

        art_center_x, art_center_y = self.getItemCenter(rows[0], 0, offset_y)
        mouse.move(self.scanner.game_info.left + art_center_x,
                   self.scanner.game_info.top + art_center_y)
        mouse.click()

        if offset_y != 0:
            time.sleep(0.5)

        for art_row in rows:
            for art_col in range(self.scanner.game_info.art_cols):
                if self.scanner.stopped:
                    return False

                click = self.getItemCenter(art_row, art_col, offset_y)
                if self.scanner.waitSwitched(*click, min_wait=0.1, max_wait=3):
                    detail_img = captureWindow(self.scanner.game_info.hwnd, (
                        self.scanner.game_info.art_info_left,
                        self.scanner.game_info.art_info_top,
                        self.scanner.game_info.art_info_left + self.scanner.game_info.art_info_width,
                        self.scanner.game_info.art_info_top + self.scanner.game_info.art_info_height))
                    item_coord = getMaterialItemCoord(self.scanner.game_info, art_row, art_col, offset_y=offset_y)
                    item_img = captureWindow(self.scanner.game_info.hwnd, item_coord)

                    if debug_material_crop:
                        debug_info = {
                            "page": self.page,
                            "row": art_row,
                            "col": art_col,
                            "rows": rows,
                            "offset_y": round(offset_y, 2),
                            "item_coord": item_coord,
                            "click": [round(v, 1) for v in click],
                            "last_row": offset_y != 0,
                            "lastrow_offset": self.scanner.game_info.lastrow_offset,
                            "lastrow_offset_applied": offset_y != 0,
                        }

                    if art_col == self.scanner.game_info.art_cols - 1:
                        art_row += 1
                        art_col = 0
                    else:
                        art_col += 1

                    if art_row in rows:
                        next_center_x, next_center_y = self.getItemCenter(art_row, art_col, offset_y)
                        mouse.move(self.scanner.game_info.left + next_center_x,
                                   self.scanner.game_info.top + next_center_y)
                        mouse.click()

                    if debug_material_crop:
                        callback(detail_img, item_img, debug_info)
                    else:
                        callback(detail_img, item_img)
                else:
                    return False
        return True

    def is_between_rows(self) -> bool:
        return not self.scanner.isKeypointOnLabel()

    def alignFirstRow(self):
        return self.scanner.alignFirstRow()

    def scrollToRow(self, target_row, max_scrolls=20, extra_scroll=0, interval=0.05) -> int:
        return self.scanner.scrollToRow(target_row, max_scrolls, extra_scroll, interval)

    def scanMaterials(self, callback, interval=0.05, debug=False, on_page=None):
        """Scan the entire material inventory with full pagination and dynamic row offset detection."""
        rows = self.scanner.grid.rows
        self.page = 0
        self.scanner.page = 0
        self.scanner.last_page_info = None

        while True:
            if self.scanner.stopped:
                return "stopped"
            if on_page is not None:
                on_page(self.page, list(range(rows)), 0, None)
            if not self.scanRows(range(rows), callback, offset_y=0, debug=debug):
                return "stopped" if self.scanner.stopped else "no_item"

            scrolled = self.scrollToRow(rows, max_scrolls=20, extra_scroll=int(rows > 5), interval=interval)
            self.page += 1
            self.scanner.page += 1

            if scrolled >= rows:
                continue

            last_rows, offset_y, info = self.scanner.lastPageRows(scrolled)
            self.scanner.last_page_info = info
            if on_page is not None:
                on_page(self.page, last_rows, offset_y, info)
            if last_rows and not self.scanner.stopped:
                if not self.scanRows(last_rows, callback, offset_y=offset_y, debug=debug):
                    return "stopped" if self.scanner.stopped else "no_item"
            return "stopped" if self.scanner.stopped else "end"
