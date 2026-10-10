import json
import math

import numpy as np
import pytest

import layout
from art_scanner_logic import GameInfo

SIZES_16_9 = [(1280, 720), (1600, 900), (1920, 1080), (2560, 1440), (3840, 2160)]
SIZES = SIZES_16_9 + [(1680, 1050), (1920, 1200), (2560, 1600), (2560, 1080), (3440, 1440)]


def legacy_game_info(w, h):
    gi = GameInfo.__new__(GameInfo)  # no window: skip win32gui in __init__
    gi.hwnd, gi.w, gi.h, gi.left, gi.top = 0, w, h, 0, 0
    gi.calculateCoordinates()
    return gi


@pytest.fixture(autouse=True)
def no_overrides(monkeypatch, tmp_path):
    monkeypatch.delenv(layout.OVERRIDE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)  # no artifact_layout.json in the working directory


@pytest.mark.parametrize('w, h', SIZES)
def test_material_grid_matches_legacy_game_info(w, h):
    gi = legacy_game_info(w, h)
    g = gi.material_grid
    assert g.scale_ratio == gi.scale_ratio
    assert g.cols == gi.art_cols
    assert g.rows == gi.art_rows
    assert g.incomplete_lastrow == gi.incomplete_lastrow
    assert g.first_x == pytest.approx(gi.first_art_x)
    assert g.first_y == pytest.approx(gi.first_art_y)
    assert g.keypoint_x == pytest.approx(gi.scroll_fin_keypt_x)
    assert g.keypoint_y == pytest.approx(gi.scroll_fin_keypt_y)
    assert g.lastrow_offset == pytest.approx(gi.lastrow_offset)
    assert g.card_width == pytest.approx(gi.art_width)
    assert g.card_height == pytest.approx(gi.art_height)
    for row, col in ((0, 0), (1, 2), (gi.art_rows - 1, gi.art_cols - 1)):
        # == the legacy ArtScannerLogic.getArtCenter
        x = gi.first_art_x + (gi.art_width + gi.art_gap_x) * col + gi.art_width / 2
        y = gi.first_art_y + (gi.art_height + gi.art_gap_y) * row + gi.art_height / 5
        assert g.click_point(row, col) == (pytest.approx(x), pytest.approx(y))
    panel = layout.detail_panel_box(w, h)
    assert panel == (pytest.approx(gi.art_info_left), pytest.approx(gi.art_info_top),
                     pytest.approx(gi.art_info_left + gi.art_info_width),
                     pytest.approx(gi.art_info_top + gi.art_info_height))


@pytest.mark.parametrize('w, h', SIZES)
def test_artifact_grid_geometry(w, h):
    gi = legacy_game_info(w, h)
    g = gi.artifact_grid
    s = g.scale_ratio
    assert g.kind == 'artifact'
    assert g.first_y == pytest.approx((161 + 77.6) * s)
    assert g.keypoint_y == pytest.approx((335 + 77.6) * s)
    assert g.cols == gi.material_grid.cols
    expected_rows = 5 if h / w > 0.6 else 4  # 16:10 shows one more row
    assert g.rows == expected_rows
    assert g.incomplete_lastrow is False
    # every scanned row is fully visible, the next one is not
    assert g.row_top(g.rows - 1) + g.card_height <= g.grid_bottom + 1e-6
    assert g.row_top(g.rows) + g.card_height > g.grid_bottom
    assert g.visible_rows(0) == list(range(g.rows))


def test_artifact_rows_at_1080p():
    g = layout.GridLayout.for_artifacts(1920, 1080)
    assert (g.rows, g.cols) == (4, 8)
    assert g.rows_float == pytest.approx(4.546, abs=0.01)
    legacy = layout.GridLayout.for_materials(1920, 1080)
    assert (legacy.rows, legacy.cols) == (5, 8)


def test_overrides(tmp_path, monkeypatch):
    path = tmp_path / 'artifact_layout.json'
    path.write_text(json.dumps({'offset_y': 80, 'rows': 3, 'lastrow_offset': 150, 'keypoint_dy': 170,
                                'comment': 'measured on my PC'}), encoding='utf-8')
    overrides = layout.load_overrides()
    assert overrides == {'offset_y': 80, 'rows': 3, 'lastrow_offset': 150, 'keypoint_dy': 170}
    g = layout.GridLayout.for_artifacts(2560, 1440, overrides)
    assert (g.first_y, g.rows, g.lastrow_offset, g.keypoint_y) == (241, 3, 150, 161 + 170 + 80)
    gi = legacy_game_info(2560, 1440)
    assert gi.artifact_grid.rows == 3  # GameInfo picks the file up from the working directory

    monkeypatch.setenv(layout.OVERRIDE_ENV, str(tmp_path / 'missing.json'))
    assert layout.load_overrides() == {}


def test_overrides_reject_unknown_keys(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps({'offsety': 80}), encoding='utf-8')
    with pytest.raises(ValueError):
        layout.load_overrides(str(path))


def test_bad_override_file_does_not_break_game_info(tmp_path):
    (tmp_path / 'artifact_layout.json').write_text('{not json', encoding='utf-8')
    gi = legacy_game_info(1920, 1080)
    assert gi.artifact_grid.rows == 4


def test_visible_rows_with_offset():
    g = layout.GridLayout.for_artifacts(2560, 1440)
    assert g.visible_rows(0) == [0, 1, 2, 3]
    # scrolled up by 100 px: row 0 is cut at the top, row 4 is not fully visible yet (needs ~106 px)
    assert g.visible_rows(100) == [1, 2, 3]
    assert g.visible_rows(110) == [1, 2, 3, 4]
    assert g.visible_rows(-1) == [0, 1, 2, 3]


# ------------------------------------------------------------------------- label strip detection
def synthetic_column(g, offset_y, n_rows, strip=35.0, height=None, noise=None):
    """Pixel column through the first card column: cards of n_rows rows scrolled up by offset_y."""
    s = g.scale_ratio
    x, y0, _, y1 = g.label_column_box()
    col = np.zeros((y1 - y0, 3), np.uint8)
    col[:] = (40, 40, 50)
    for row in range(-1, n_rows):
        top = g.row_top(row, offset_y)
        a, b = top + g.card_height - strip * s, top + g.card_height
        for y in range(y0, y1):
            if top <= y < top + g.card_height and row >= 0:
                col[y - y0] = layout.LABEL_COLOR if a <= y < b else (90, 70, 120)
    if noise is not None:
        rng = np.random.RandomState(0)
        col = np.clip(col.astype(int) + rng.randint(-noise, noise + 1, col.shape), 0, 255).astype(np.uint8)
    return col, y0


@pytest.mark.parametrize('w, h', [(1600, 900), (1920, 1080), (2560, 1440), (1920, 1200)])
@pytest.mark.parametrize('offset_ref', [0, 3, 30, 77.6, 125.5, 203.1, 225])
def test_detect_row_offset(w, h, offset_ref):
    g = layout.GridLayout.for_artifacts(w, h)
    s = g.scale_ratio
    offset = offset_ref * s
    column, y0 = synthetic_column(g, offset, n_rows=12)
    keypoint_is_label = None
    detected, rows = layout.detect_row_offset(g, column, y0, keypoint_is_label)
    expected = offset if offset <= 0.85 * g.pitch_y else offset - g.pitch_y
    assert detected == pytest.approx(expected, abs=1.5)
    assert rows == g.visible_rows(detected)


def test_detect_row_offset_keypoint_hint():
    g = layout.GridLayout.for_artifacts(1920, 1080)
    # list stopped 5 px *before* the aligned position: raw offset is pitch - 5
    column, y0 = synthetic_column(g, -5, n_rows=8)
    off, rows = layout.detect_row_offset(g, column, y0, keypoint_is_label=True)
    assert off == pytest.approx(-5, abs=1.5)
    # same picture with the keypoint between rows means the offset is large and positive
    column, y0 = synthetic_column(g, 0.8 * g.pitch_y, n_rows=8)
    off, rows = layout.detect_row_offset(g, column, y0, keypoint_is_label=False)
    assert off == pytest.approx(0.8 * g.pitch_y, abs=1.5)
    assert rows[0] == 1


def test_detect_row_offset_partial_last_rows():
    g = layout.GridLayout.for_artifacts(1920, 1080)
    column, y0 = synthetic_column(g, 40, n_rows=3)  # only 3 rows of items exist
    off, rows = layout.detect_row_offset(g, column, y0)
    assert off == pytest.approx(40, abs=1.5)
    assert rows == [1, 2]  # row 0 is cut off at the top


def test_detect_row_offset_none_without_labels():
    g = layout.GridLayout.for_artifacts(1920, 1080)
    x, y0, _, y1 = g.label_column_box()
    column = np.full((y1 - y0, 3), 30, np.uint8)
    assert layout.detect_row_offset(g, column, y0) is None


def test_detect_row_offset_tolerates_noise():
    g = layout.GridLayout.for_artifacts(2560, 1440)
    column, y0 = synthetic_column(g, 50, n_rows=10, noise=3)
    off, _ = layout.detect_row_offset(g, column, y0)
    assert off == pytest.approx(50, abs=2)


def test_lastrow_offset_estimate():
    assert layout.ARTIFACT_LASTROW_OFFSET == pytest.approx((125.5 + 77.6) % (203.21 + 30.35))
    g = layout.GridLayout.for_artifacts(1920, 1080)
    assert g.lastrow_offset == pytest.approx(layout.ARTIFACT_LASTROW_OFFSET * 0.75)


def test_label_column_box_is_integer_and_inside_window():
    for w, h in SIZES:
        g = layout.GridLayout.for_artifacts(w, h)
        x, y0, x1, y1 = g.label_column_box()
        assert all(isinstance(v, int) for v in (x, y0, x1, y1))
        assert 0 <= y0 < y1 <= h and x1 == x + 2
        assert math.isclose(x, round(g.keypoint_x))


def test_draw_overlays():
    from PIL import Image
    g = layout.GridLayout.for_artifacts(1600, 900)
    img = Image.new('RGB', (1600, 900), (10, 10, 10))
    out = layout.draw_grid_overlay(img, g, 0, None, layout.detail_panel_box(1600, 900), [(300, 320)])
    assert out.size == img.size and out.getpixel((int(g.first_x) + 1, int(g.first_y))) != (10, 10, 10)
    boxed = layout.draw_boxes(img, {'name': (10, 10, 100, 40)})
    assert boxed.getpixel((10, 20)) != (10, 10, 10)
