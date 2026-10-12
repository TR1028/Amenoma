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


def test_material_database_export_good_json(tmp_path):
    db = material_saver.MaterialDatabase()
    try:
        assert db.add({"name": "北陆单手剑原胚", "amount": 1}, None, None)
        assert db.add({"name": "未知测试材料名称", "amount": 3}, None, None)
        out = tmp_path / "materials.GOOD.json"
        db.exportGOODJSON(str(out))
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["materials"]["NorthlanderSwordBillet"] == 1
        assert "未知测试材料名称" not in data["materials"]
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
