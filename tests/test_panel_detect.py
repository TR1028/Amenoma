# -*- coding: utf-8 -*-
"""Unit tests for ArtScanner.panel_detect: lock, astral mark, and unactivated line detection."""
import numpy as np
from PIL import Image
import pytest

import panel_detect


def test_navy_text_detection():
    # Navy text: (73, 83, 102) +- 25
    navy_img = np.full((20, 100, 3), (73, 83, 102), dtype=np.uint8)
    assert panel_detect.has_navy_text(navy_img)

    # Off-navy background (e.g. white/light grey background (235, 230, 220))
    bg_img = np.full((20, 100, 3), (235, 230, 220), dtype=np.uint8)
    assert not panel_detect.has_navy_text(bg_img)


def test_lock_detection():
    # Red padlock glyph: r>=190, g in [90, 185], b in [60, 170], r-g>=55, r-b>=65 (e.g. 255, 137, 117)
    base = np.full((100, 100, 3), (70, 80, 100), dtype=np.uint8)  # dark button
    # Add red padlock patch
    base[40:60, 40:60] = (255, 137, 117)
    img = Image.fromarray(base)

    coords = [30, 30, 70, 70]
    assert panel_detect.detect_lock(img, coords, scale_ratio=1.0)

    # Empty region without red padlock
    empty_coords = [0, 0, 30, 30]
    assert not panel_detect.detect_lock(img, empty_coords, scale_ratio=1.0)


def test_astral_mark_detection():
    # Golden star: r>=190, g>=140, b<=125, r-b>=90, g-b>=55 (e.g. 255, 204, 50)
    base = np.full((100, 100, 3), (70, 80, 100), dtype=np.uint8)
    base[40:60, 40:60] = (255, 204, 50)
    img = Image.fromarray(base)

    coords = [30, 30, 70, 70]
    assert panel_detect.detect_astral_mark(img, coords, scale_ratio=1.0)

    empty_coords = [0, 0, 30, 30]
    assert not panel_detect.detect_astral_mark(img, empty_coords, scale_ratio=1.0)


def test_unactivated_grey_line_detection():
    # Grey text line: spread <= 28, mean in [90, 200]
    grey_img = np.full((20, 100, 3), (140, 140, 140), dtype=np.uint8)
    assert panel_detect.is_unactivated_line(grey_img, scale_ratio=1.0)

    # Green text line: g >= 120, g-r >= 35, g-b >= 35 (e.g. 93, 178, 87)
    green_img = np.full((20, 100, 3), (93, 178, 87), dtype=np.uint8)
    assert not panel_detect.is_unactivated_line(green_img, scale_ratio=1.0)


def test_split_substat_lines():
    # 4 lines: 3 navy lines, 1 grey unactivated line
    navy_line = np.full((20, 100, 3), (73, 83, 102), dtype=np.uint8)
    grey_line = np.full((20, 100, 3), (140, 140, 140), dtype=np.uint8)
    blank_line = np.full((20, 100, 3), (240, 240, 240), dtype=np.uint8)

    lines = [navy_line, navy_line, navy_line, grey_line]
    present, unactivated = panel_detect.split_substat_lines(lines, scale_ratio=1.0)
    assert present == [1, 2, 3, 4]
    assert unactivated == [4]

    # 2 navy lines, then blank
    lines_2 = [navy_line, navy_line, blank_line, blank_line]
    present_2, unactivated_2 = panel_detect.split_substat_lines(lines_2, scale_ratio=1.0)
    assert present_2 == [1, 2]
    assert unactivated_2 == []
