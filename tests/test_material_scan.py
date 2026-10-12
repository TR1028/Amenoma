import json
import logging
import sys
import numpy as np
import pytest
from PIL import Image

import config
import material_saver
import utils


def test_ensure_standard_streams_replaces_none_and_repairs_handlers(monkeypatch):
    bad_handler = logging.StreamHandler(stream=sys.stderr)
    bad_handler.stream = None
    test_logger = logging.getLogger("amenoma_test_null_stream")
    test_logger.addHandler(bad_handler)

    try:
        monkeypatch.setattr(sys, "stdout", None)
        monkeypatch.setattr(sys, "stderr", None)
        utils.ensure_standard_streams()

        assert sys.stdout is not None
        assert sys.stderr is not None
        sys.stdout.write("hello")
        sys.stdout.flush()
        sys.stderr.write("world")
        sys.stderr.flush()
        assert bad_handler.stream is not None
        bad_handler.flush()
    finally:
        test_logger.removeHandler(bad_handler)


@pytest.mark.parametrize("raw, expected", [
    ("1", 1),
    ("l", 1),
    ("I", 1),
    ("|", 1),
    ("O", 0),
    ("1O", 10),
    ("1,234", 1234),
    ("+20", 20),
    ("46.6%", 0.466),
    (5, 5),
])
def test_decode_value_handles_ocr_digit_confusions(raw, expected):
    assert utils.decodeValue(raw) == pytest.approx(expected)


def test_material_name_auto_correct_billet_variants():
    assert utils.material_name_auto_correct("北陆单手剑原胚") == "北陆单手剑原胚"
    assert utils.material_name_auto_correct("北陆单手剑原坯") == "北陆单手剑原胚"
    assert utils.material_name_auto_correct("央陆单手剑原坯") == "央陆单手剑原胚"
    assert utils.material_name_auto_correct("垠土单手剑原坯") == "垠土单手剑原胚"
    assert utils.material_name_auto_correct("异的剑柄") == "裂断的剑柄"
    assert utils.material_name_auto_correct("万石") == "万相石"
    assert utils.material_name_auto_correct("燃铃木") == "悬铃木"


def test_material_database_export_good_json(tmp_path):
    db = material_saver.MaterialDatabase()
    try:
        assert db.add({"name": "北陆单手剑原胚", "amount": 1}, None, None)
        assert db.add({"name": "今昔剧画之恶尉", "amount": 15}, None, None)
        assert db.add({"name": "未知测试材料名称", "amount": 3}, None, None)
        out = tmp_path / "materials.GOOD.json"
        db.exportGOODJSON(str(out))
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["materials"]["NorthlanderSwordBillet"] == 1
        assert data["materials"]["MaskOfTheWickedLieutenant"] == 15
        assert "未知测试材料名称" not in data["materials"]
        assert data["背包素材"]["北陆单手剑原胚"] == 1
        assert data["背包素材"]["今昔剧画之恶尉"] == 15
        assert data["背包素材"]["今昔剧画的恶尉"] == 15
    finally:
        del db


def test_ocr_m_with_null_streams_and_thin_crops(monkeypatch):
    import os
    ocr_m = pytest.importorskip("ocr_m")
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    utils.ensure_standard_streams()

    bundle_dir = os.path.dirname(os.path.abspath(config.__file__))
    model = ocr_m.OCR(model=config.model_path(bundle_dir, "CHS", "material"))
    # 1-pixel wide vertical line in amount crop (simulating digit '1' on Northlander Sword Billet)
    detail = Image.new("RGB", (650, 1119), (245, 241, 235))
    item = Image.new("RGB", (160, 210), (245, 241, 235))
    arr = np.array(item)
    arr[180:198, 80:81] = [50, 50, 50]
    item = Image.fromarray(arr)

    res = model.detect_info(detail, item)
    assert "name" in res and "amount" in res and "star" in res


def test_adaptive_material_amount_crop_and_empty_slot_detection():
    import layout

    # 1. Empty slot (blurred warm background, no beige strip and no dark digit ink)
    empty = np.full((203, 164, 3), (225, 205, 195), dtype=np.uint8)
    empty[:50, :, :] = (235, 215, 205)
    empty_img = Image.fromarray(empty)
    assert layout.find_card_label_strip(empty_img, 1.0) is None
    assert layout.is_empty_material_slot(empty_img, 1.0) is True

    # 2. Shifted-up card: beige strip at y=154..190, white border at 191..194, dark gap at 195..202
    card = np.full((203, 164, 3), (105, 95, 153), dtype=np.uint8)
    card[154:191, :, :] = (233, 229, 220)
    card[191:195, :, :] = (255, 255, 255)
    card[195:203, :, :] = (80, 70, 80)
    # Digit at y=163..183 (would be half cut off by legacy y=175..203 crop)
    card[163:184, 76:88, :] = (73, 83, 102)
    card_img = Image.fromarray(card)

    assert layout.is_empty_material_slot(card_img, 1.0) is False
    assert layout.find_card_label_strip(card_img, 1.0) == (154, 190)

    ocr_m = pytest.importorskip("ocr_m")
    cropped = np.asarray(ocr_m.crop_amount(card_img, 1.0))
    # Bottom-right corner of cropped amount must be beige (not the dark gap at y>=195)
    assert tuple(cropped[-1, -1, :3]) == (233, 229, 220)
    # Full digit height (21 rows) must be inside the crop
    digit_rows = np.flatnonzero((cropped[:, :, 0] < 100).any(axis=1))
    assert len(digit_rows) == 21

    # 3. Bottom-clipped card: beige strip at y=176..202 with digit ink touching y=202
    clipped = np.full((203, 164, 3), (96, 110, 148), dtype=np.uint8)
    clipped[176:203, :, :] = (233, 229, 220)
    clipped[185:203, 78:86, :] = (73, 83, 102)
    clipped_img = Image.fromarray(clipped)
    cropped_clipped = np.asarray(ocr_m.crop_amount(clipped_img, 1.0))
    # Height is extended by 2 rows (from 203 - 182 = 21 to 23) to restore clipped stem
    assert cropped_clipped.shape[0] == 23

