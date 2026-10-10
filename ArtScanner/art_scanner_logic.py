import math
import time

import mouse
import numpy as np
import win32gui

import layout
from utils import captureWindow
from utils import logger


class GameInfo:
    def __init__(self, hwnd):
        self.hwnd = hwnd
        self.w, self.h = win32gui.GetClientRect(self.hwnd)[2:]
        self.left, self.top = win32gui.ClientToScreen(self.hwnd, (0, 0))

    def calculateCoordinates(self):
        self.scale_ratio = min(self.w / 2560, self.h / 1440)

        self.art_width = 164.39 * self.scale_ratio
        self.art_height = 203.21 * self.scale_ratio
        self.art_expand = 6 * self.scale_ratio

        self.art_gap_x = 30.89 * self.scale_ratio
        self.art_gap_y = 30.35 * self.scale_ratio

        self.art_info_width = 656 * self.scale_ratio
        self.art_info_height = 1119 * self.scale_ratio

        # == in <=2.4 ==
        # self.left_margin = (199.33 if self.w < 2 * self.h else 295.33) * self.scale_ratio
        # == updated in 2.5 ==
        self.left_margin = (155 if self.w < 2 * self.h else 295.33) * self.scale_ratio
        self.right_margin = (871.33 if self.w < 2 * self.h else 967.33) * self.scale_ratio
        self.info_margin = 0 if self.w < 2 * self.h else 96 * self.scale_ratio

        self.art_cols = int(math.floor(
            (self.w - self.left_margin - self.right_margin + self.art_gap_x) /
            (self.art_width + self.art_gap_x)))

        self.art_shift = ((self.w - self.left_margin - self.right_margin + self.art_gap_x)
                          - self.art_cols * (self.art_width + self.art_gap_x)) / 2

        self.first_art_x = self.left_margin + self.art_shift
        self.first_art_y = 161 * self.scale_ratio

        self.art_info_top = 160 * self.scale_ratio
        # == in <=2.4 ==
        # self.art_info_left = self.w - 837 * self.scale_ratio - self.info_margin
        # == updated in 2.5 ==
        self.art_info_left = self.w - 816 * self.scale_ratio - self.info_margin
        # scroll_keypt_x = left_margin + art_shift + 158*scale_ratio
        # scroll_keypt_y = 1270*scale_ratio+h-1440*scale_ratio
        self.scroll_fin_keypt_x = self.left_margin + self.art_shift + 10 * self.scale_ratio
        self.scroll_fin_keypt_y = 335 * self.scale_ratio

        self.art_rows = round((1270 * self.scale_ratio + self.h - 1440 * self.scale_ratio
                               - self.first_art_y + self.art_gap_y) /
                              (self.art_height + self.art_gap_y))
        self.incomplete_lastrow = ((1270 * self.scale_ratio + self.h
                                    - 1440 * self.scale_ratio
                                    - self.first_art_y + self.art_gap_y) /
                                   (self.art_height + self.art_gap_y) - self.art_rows + 1) < 0.7

        # == in <=2.4 ==
        # self.lastrow_offset = 34.5 * self.scale_ratio
        # == updated in 2.5 ==
        self.lastrow_offset = 125.5 * self.scale_ratio
        # Character Development Item
        self.cdi_button = (self.left + 1025 * self.scale_ratio,
                           self.top + 70 * self.scale_ratio)
        # Materials
        self.m_button = (self.left + 1270 * self.scale_ratio,
                           self.top + 70 * self.scale_ratio)

        # The attributes above describe the material inventory (unchanged since 2.5, used by
        # MaterialScannerLogic).  Since 5.2 the artifact inventory has a filter bar above the grid, so
        # the artifact scanner uses its own geometry, see layout.py.
        self.material_grid = layout.GridLayout.for_materials(self.w, self.h)
        try:
            overrides = layout.load_overrides()
        except Exception as e:
            logger.exception(e)
            overrides = {}
        if overrides:
            logger.info(f"artifact layout overrides: {overrides}")
        self.artifact_grid = layout.GridLayout.for_artifacts(self.w, self.h, overrides)
        logger.info(f"{self.material_grid}")
        logger.info(f"{self.artifact_grid}")


class ArtScannerLogic:
    def __init__(self, game_info, grid=None):
        self.game_info = game_info
        # geometry of the grid to scan (artifact inventory unless told otherwise)
        if grid is None:
            grid = getattr(game_info, 'artifact_grid', None) or layout.GridLayout.for_artifacts(game_info.w, game_info.h)
        self.grid = grid
        self.stopped = False
        self.avg_response_time = 1 / 60
        # set by scanArtifacts, reported in the debug info of scanRows
        self.page = 0
        self.last_page_info = None

    def interrupt(self):
        self.stopped = True

    def waitSwitched(self, art_center_x, art_center_y, min_wait=0.1, max_wait=3,
                     condition=lambda pix: sum(pix) / 3 > 200):
        start = time.time()
        total_wait = 0
        count = 0
        while True:
            count += 1
            mouse.move(self.game_info.left + art_center_x, self.game_info.top + art_center_y)
            mouse.click()
            pix = captureWindow(self.game_info.hwnd, self.grid.selection_probe(art_center_x, art_center_y))
            if condition(pix.getpixel((0, 0))):
                self.avg_response_time = 0.5 * self.avg_response_time + 0.5 * (time.time() - start)
                logger.info("waitSwitched: %d times, avg response time: %f" % (count, self.avg_response_time))
                return True
            else:
                time.sleep(min_wait)
                total_wait += min_wait
            if total_wait > max_wait:
                return False

    def getArtCenter(self, row, col, offset_y=0):
        return self.grid.click_point(row, col, offset_y)

    def scanRows(self, rows, callback, offset_y=0, debug=False):
        '''
        callback: function to take in artifact image and do what ever you want
        offset_y: the grid is scrolled up by this many pixels (last page, see lastPageRows)
        debug: call callback(art_img, debug_info) with the grid position of the item
        '''
        rows = list(rows)
        if len(rows) < 1:
            return True
        art_center_x, art_center_y = self.getArtCenter(rows[0], 0, offset_y)
        mouse.move(self.game_info.left + art_center_x, self.game_info.top + art_center_y)
        mouse.click()
        for art_row in rows:
            for art_col in range(self.grid.cols):
                if self.stopped:
                    return False
                click = self.getArtCenter(art_row, art_col, offset_y)
                if self.waitSwitched(*click, min_wait=0.1, max_wait=3):
                    art_img = captureWindow(self.game_info.hwnd, (
                        self.game_info.art_info_left,
                        self.game_info.art_info_top,
                        self.game_info.art_info_left + self.game_info.art_info_width,
                        self.game_info.art_info_top + self.game_info.art_info_height))
                    debug_info = {"page": self.page, "row": art_row, "col": art_col, "rows": rows,
                                  "offset_y": round(offset_y, 2),
                                  "click": [round(v, 1) for v in click],
                                  "item_box": [round(v, 1) for v in self.grid.item_box(art_row, art_col, offset_y)],
                                  "last_page": self.last_page_info}
                    if art_col == self.grid.cols - 1:
                        art_row += 1
                        art_col = 0
                    else:
                        art_col += 1
                    if art_row in rows:
                        art_center_x, art_center_y = self.getArtCenter(art_row, art_col, offset_y)
                        mouse.move(self.game_info.left + art_center_x, self.game_info.top + art_center_y)
                        mouse.click()
                    if debug:
                        callback(art_img, debug_info)
                    else:
                        callback(art_img)
                else:
                    return False
        return True

    def isKeypointOnLabel(self):
        pix = captureWindow(self.game_info.hwnd, self.grid.keypoint_box())
        return layout.is_label_pixel(pix.getpixel((0, 0)))

    def alignFirstRow(self):
        mouse.move(self.game_info.left + self.grid.first_x, self.game_info.top + self.grid.first_y)
        mouse.click()
        if not self.isKeypointOnLabel():
            for _ in range(3):
                mouse.wheel(1)
            time.sleep(0.1)
            self.scrollToRow(0)

    def scrollToRow(self, target_row, max_scrolls=20, extra_scroll=0, interval=0.05):
        in_between_row = False
        rows_scrolled = 0
        lines_scrolled = 0
        while True:
            if not self.isKeypointOnLabel():
                # if in_between_row==False:
                #     print('到行之间了')
                in_between_row = True
            elif in_between_row:
                in_between_row = False
                rows_scrolled += 1
                lines_scrolled = 0
                # print(f'已翻{rows_scrolled}行')
                if rows_scrolled >= target_row:
                    for _ in range(extra_scroll):
                        mouse.wheel(-1)
                    return rows_scrolled
            if lines_scrolled > max_scrolls:
                return rows_scrolled
            get_first_art = lambda: np.array(captureWindow(self.game_info.hwnd, self.grid.first_item_probe()))
            first_art = get_first_art()
            for _ in range(7 if lines_scrolled == 0 and target_row > 0 else 1):
                mouse.wheel(-1)
                lines_scrolled += 1
                # print('翻一下')
            time.sleep(self.avg_response_time)
            total_waited = 0
            while True:
                if total_waited > 5:
                    break
                if np.max(np.abs(get_first_art() - first_art)) > 5:
                    break
                time.sleep(interval)
                total_waited += interval

    def locateRows(self):
        """Measure the current row offset from the label strips, see layout.detect_row_offset."""
        box = self.grid.label_column_box()
        column = np.array(captureWindow(self.game_info.hwnd, box))[:, 0, :3]
        return layout.detect_row_offset(self.grid, column, box[1], keypoint_is_label=self.isKeypointOnLabel())

    def lastPageRows(self, rows_scrolled):
        """The list stopped scrolling after ``rows_scrolled`` rows (< grid.rows): end of the inventory.

        The last scroll usually stops between two rows, so the remaining rows are not where
        getArtCenter expects them, and one more row may have become visible below the page.
        :return: (rows to scan, offset_y for scanRows, info dict for logging)
        """
        first_new = self.grid.rows - rows_scrolled
        detected = self.locateRows()
        if detected is not None:
            offset_y, visible = detected
            source = "detected"
        else:
            # no label strip found: aligned if the keypoint is on a label, else use the estimate
            offset_y = 0 if self.isKeypointOnLabel() else self.grid.lastrow_offset
            visible = self.grid.visible_rows(offset_y)
            source = "estimated"
        rows = [r for r in visible if r >= first_new]
        info = {"rows_scrolled": rows_scrolled, "first_new_row": first_new, "offset_y": round(offset_y, 1),
                "visible_rows": visible, "rows": rows, "source": source}
        logger.info(f"last page: {info}")
        return rows, offset_y, info

    def scanArtifacts(self, callback, interval=0.05, debug=False, on_page=None):
        """Scan the whole inventory, page by page (call alignFirstRow first).

        Every page scans the ``grid.rows`` fully visible rows and scrolls that many rows down.  When
        the list stops early (end of the inventory) the remaining rows are located with
        lastPageRows and scanned once.

        :param callback: callback(art_img) or, with ``debug``, callback(art_img, debug_info)
        :param interval: polling interval while scrolling (the "delay" setting)
        :param on_page: optional on_page(page, rows, offset_y, last_page_info) called before a page
                        is scanned (debug dump)
        :return: why the scan ended: "stopped" (interrupted), "no_item" (a click did not select an
                 item, i.e. an empty slot) or "end" (last page done)
        """
        rows = self.grid.rows
        self.page = 0
        self.last_page_info = None
        while True:
            if self.stopped:
                return "stopped"
            if on_page is not None:
                on_page(self.page, list(range(rows)), 0, None)
            if not self.scanRows(range(rows), callback, debug=debug):
                return "stopped" if self.stopped else "no_item"
            scrolled = self.scrollToRow(rows, max_scrolls=20, extra_scroll=int(rows > 5), interval=interval)
            self.page += 1
            if scrolled >= rows:
                continue
            last_rows, offset_y, info = self.lastPageRows(scrolled)
            self.last_page_info = info
            if on_page is not None:
                on_page(self.page, last_rows, offset_y, info)
            if last_rows and not self.stopped:
                if not self.scanRows(last_rows, callback, offset_y=offset_y, debug=debug):
                    return "stopped" if self.stopped else "no_item"
            return "stopped" if self.stopped else "end"

