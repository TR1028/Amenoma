import json
import math
import os

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
    assert (legacy.rows, legacy.cols) == (4, 8)


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


# ---------------------------------------------------------------- multi-aspect ratio yas parity
YAS_RESOLUTIONS = [
    (1600, 900, "16:9", 5, 8, 1090.0, 100.0, 700.0, 762.6),
    (1440, 900, "16:10", 6, 8, 981.0, 90.0, 720.0, 776.0),
    (1280, 960, "4:3", 7, 8, 872.0, 80.0, 800.0, 849.8),
    (2100, 900, "21:9", 5, 11, 1531.0, 100.0, 700.0, 762.6),
    (3440, 1440, "21:9", 5, 11, 2528.0, 160.0, 1119.0, 1220.0),
]


@pytest.mark.parametrize('w, h, aspect, expected_rows, expected_cols, panel_left, panel_top, panel_h, equip_top', YAS_RESOLUTIONS)
def test_yas_multi_aspect_ratio_parity(w, h, aspect, expected_rows, expected_cols, panel_left, panel_top, panel_h, equip_top):
    gi = legacy_game_info(w, h)
    s = gi.scale_ratio

    # 1. Panel dimensions and position
    panel = layout.detail_panel_box(w, h)
    assert panel[0] == pytest.approx(panel_left, abs=2.0)
    assert panel[1] == pytest.approx(panel_top, abs=2.0)
    assert (panel[3] - panel[1]) == pytest.approx(panel_h, abs=2.0)

    # 2. Equipped rect anchoring across aspect ratios
    # Equipped ref coords: [118, 1060, 520, 1100] relative to panel
    ocr_boxes = layout.panel_ocr_boxes(panel[0], panel[1], s, panel_height=panel[3] - panel[1],
                                       coords_dict={'equipped': [118, 1060, 520, 1100]})
    equipped_client_top = ocr_boxes['equipped'][1]
    assert equipped_client_top == pytest.approx(equip_top, abs=2.5)

    # 3. Top bar buttons centered relative to w/2
    assert gi.cdi_button[0] == pytest.approx(w / 2 - 255 * s)
    assert gi.m_button[0] == pytest.approx(w / 2 - 10 * s)


def test_material_scanner_offset_sync():
    import material_scanner_logic
    gi = legacy_game_info(1920, 1080)
    msl = material_scanner_logic.MaterialScannerLogic(gi)

    # Verify getItemCenter and getMaterialItemCoord apply the same offset_y
    offset_y = 50.0
    click_x, click_y = msl.getItemCenter(2, 3, offset_y=offset_y)
    item_box = material_scanner_logic.getMaterialItemCoord(gi, 2, 3, offset_y=offset_y)

    # Item box top should match the grid row_top shifted by offset_y
    expected_top = gi.material_grid.row_top(2, offset_y=offset_y)
    assert item_box[1] == pytest.approx(expected_top, abs=1.0)
    assert click_y == pytest.approx(expected_top + gi.material_grid.card_height / 5, abs=1.0)

    # Legacy lastrow=True matches passing grid.lastrow_offset
    box_legacy = material_scanner_logic.getMaterialItemCoord(gi, 4, 0, lastrow=True)
    box_explicit = material_scanner_logic.getMaterialItemCoord(gi, 4, 0, offset_y=gi.material_grid.lastrow_offset)
    assert box_legacy == box_explicit


@pytest.mark.parametrize('w, h, aspect, expected_rows, expected_cols, panel_left, panel_top, panel_h, equip_top', YAS_RESOLUTIONS)
def test_every_scanned_card_within_viewport(w, h, aspect, expected_rows, expected_cols, panel_left, panel_top, panel_h, equip_top):
    """Verify that every scanned card (and amount label) sits 100% inside the viewport at all resolutions."""
    for kind in ('artifact', 'material'):
        grid = layout.GridLayout(w, h, kind=kind)
        s = grid.scale_ratio
        for r in range(grid.rows):
            for c in range(grid.cols):
                box = grid.item_box(r, c, offset_y=0)
                card_top, card_bottom = box[1], box[3]
                assert card_top >= grid.first_y - 1.0, f"Card ({r}, {c}) top ({card_top}) above first_y ({grid.first_y})"
                assert card_bottom <= grid.grid_bottom + 1.0, f"Card ({r}, {c}) bottom ({card_bottom}) below grid_bottom ({grid.grid_bottom})"
                if kind == 'material':
                    amt_bottom = card_top + 203.0 * s
                    assert amt_bottom <= grid.grid_bottom + 1.0, f"Material amount label ({r}, {c}) cut off by grid_bottom"


def test_verify_screenshot_cli(tmp_path):
    from PIL import Image
    from Tools import verify_screenshot

    # 1. Detail panel mode
    panel_img = Image.new('RGB', (656, 1119), (240, 235, 225))
    panel_path = str(tmp_path / 'test_panel.png')
    panel_img.save(panel_path)

    res = verify_screenshot.analyze_screenshot(panel_path, out_dir=str(tmp_path), run_ocr=True)
    assert res['is_detail_panel'] is True
    assert 'boxes' in res
    assert 'panel_features' in res
    assert os.path.isfile(str(tmp_path / 'test_panel_overlay.png'))
    assert os.path.isfile(str(tmp_path / 'test_panel_result.json'))
    # OCR inference should succeed and populate ocr_raw with fields
    assert 'ocr_raw' in res, f"Expected ocr_raw in result, got: {res.get('ocr_status')}"
    assert 'name' in res['ocr_raw']
    assert 'type' in res['ocr_raw']
    assert 'level' in res['ocr_raw']

    # 2. Full window screenshot mode (no-ocr mode)
    win_img = Image.new('RGB', (1920, 1080), (30, 30, 35))
    win_path = str(tmp_path / 'test_win.png')
    win_img.save(win_path)

    res_win = verify_screenshot.analyze_screenshot(win_path, kind='artifact', out_dir=str(tmp_path), run_ocr=False)
    assert res_win['is_detail_panel'] is False
    assert res_win['grid']['rows'] == 4
    assert res_win['grid']['cols'] == 8
    assert os.path.isfile(str(tmp_path / 'test_win_overlay.png'))
    assert os.path.isfile(str(tmp_path / 'test_win_result.json'))
