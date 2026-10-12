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
    import biliwiki_export

    db = material_saver.MaterialDatabase()
    try:
        assert db.add({"name": "北陆单手剑原胚", "amount": 1}, None, None)
        assert db.add({"name": "今昔剧画之恶尉", "amount": 15}, None, None)
        assert db.add({"name": "「角逐」的教导", "amount": 61}, None, None)
        assert db.add({"name": "未知测试材料名称", "amount": 3}, None, None)
        out = tmp_path / "materials.GOOD.json"
        db.exportGOODJSON(str(out))
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["materials"]["NorthlanderSwordBillet"] == 1
        assert data["materials"]["MaskOfTheWickedLieutenant"] == 15
        assert "未知测试材料名称" not in data["materials"]

        # BiliWiki 消耗计算器: exactly its own keys, in its own order (「导入计算器数据」 replaces
        # the whole backpack object and its panel iterates the object)
        bag = data["背包素材"]
        assert list(bag) == biliwiki_export.BILIWIKI_BAG_KEYS
        assert bag["今昔剧画的恶尉"] == 15  # BiliWiki spells it 的, Amenoma 之
        assert "今昔剧画之恶尉" not in bag
        assert bag["「角逐」的教导"] == 61  # 4.1+ items are what 「导入背包素材」 cannot import
        assert bag["摩拉"] == 0
        assert "北陆单手剑原胚" not in bag  # not tracked by the calculator
    finally:
        del db


def test_biliwiki_bag_keys_are_known_materials():
    import biliwiki_export
    import MaterialInfo

    keys = biliwiki_export.BILIWIKI_BAG_KEYS
    assert len(keys) == len(set(keys))
    known = set(MaterialInfo.MaterialsNameCHS) | set(biliwiki_export.NAME_ALIASES.values()) | {"摩拉"}
    assert [k for k in keys if k not in known] == []
    assert all(name in MaterialInfo.MaterialsNameCHS for name in biliwiki_export.NAME_ALIASES)


def _fake_material_ocr(ocr_m, pred_by_row, n_classes, blank):
    """An ocr_m.OCR whose model returns fixed predictions ({'amount': ..., 'name': ...})."""
    import charset_util

    class FakeModel:
        def predict(self, x, verbose=0):
            assert x.shape[0] == 2  # rows sorted: amount, name
            return np.stack([pred_by_row["amount"], pred_by_row["name"]]).astype(np.float32)

    chars = [str(d) for d in range(10)] + ["的", "剑", "柄", "裂", "断"]
    assert len(chars) + 2 == n_classes and blank == n_classes - 1
    model = ocr_m.OCR.__new__(ocr_m.OCR)
    model.scale_ratio = 1
    model.model = FakeModel()
    model.charset = charset_util.Charset(chars, width=320, height=16, max_length=15)
    model.characters = model.charset.characters
    model.width, model.height, model.max_length = 320, 16, 15
    return model


def _ctc_rows(emissions, timesteps, n_classes, blank):
    """(timesteps, n_classes) prediction: blank everywhere except {t: (class, prob)}."""
    pred = np.zeros((timesteps, n_classes), dtype=np.float32)
    pred[:, blank] = 1.0
    for t, (cls, prob) in emissions.items():
        pred[t, :] = 0.0
        pred[t, cls] = prob
        pred[t, blank] = 1.0 - prob
    return pred


def test_detect_info_never_masks_the_name_row():
    """Regression (4e171ac): the zero-padding mask truncated the last character of names."""
    ocr_m = pytest.importorskip("ocr_m")
    n_classes, blank, timesteps = 17, 16, 40
    cls = {c: i + 1 for i, c in enumerate([str(d) for d in range(10)] + ["的", "剑", "柄", "裂", "断"])}
    # the model emits name characters far right of the ink (t 20..38 here)
    name = _ctc_rows({20: (cls["裂"], 0.99), 24: (cls["断"], 0.99), 28: (cls["的"], 0.99),
                      33: (cls["剑"], 0.99), 38: (cls["柄"], 0.99)}, timesteps, n_classes, blank)
    amount = _ctc_rows({0: (cls["3"], 1.0), 2: (cls["2"], 1.0)}, timesteps, n_classes, blank)
    model = _fake_material_ocr(ocr_m, {"name": name, "amount": amount}, n_classes, blank)

    detail = np.full((1119, 650, 3), (245, 241, 235), dtype=np.uint8)
    detail[20:60, 40:140] = (30, 30, 30)  # short dark "name" block: ~40 px of ink after resize
    card = np.full((203, 164, 3), (105, 95, 153), dtype=np.uint8)
    card[160:197, :, :] = (233, 229, 220)
    card[170:190, 70:90, :] = (60, 60, 60)
    res = model.detect_info(Image.fromarray(detail), Image.fromarray(card))
    assert res["name"] == "裂断的剑柄"
    assert res["amount"] == "32"


def test_mask_amount_padding_removes_only_unsure_late_echo():
    ocr_m = pytest.importorskip("ocr_m")
    n_classes, blank, timesteps = 17, 16, 40
    one = 2  # class of '1'
    ink = np.zeros((16, 320), dtype=np.float32)
    ink[:, :13] = 1.0  # '11' is 13 px wide -> boundary timestep ceil(13 / 4) = 4

    echo = _ctc_rows({0: (one, 1.0), 2: (one, 1.0), 4: (one, 0.6)}, timesteps, n_classes, blank)
    ocr_m.mask_amount_padding(echo, ink, blank)
    assert [int(t) for t in np.flatnonzero(echo.argmax(-1) != blank)] == [0, 2]

    # a confident late digit is a real digit and must survive
    late = _ctc_rows({0: (one, 1.0), 2: (one, 1.0), 4: (one, 0.99)}, timesteps, n_classes, blank)
    ocr_m.mask_amount_padding(late, ink, blank)
    assert [int(t) for t in np.flatnonzero(late.argmax(-1) != blank)] == [0, 2, 4]

    # unsure emissions inside the boundary are left alone
    inside = _ctc_rows({0: (one, 0.6), 2: (one, 0.6)}, timesteps, n_classes, blank)
    ocr_m.mask_amount_padding(inside, ink, blank)
    assert [int(t) for t in np.flatnonzero(inside.argmax(-1) != blank)] == [0, 2]


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

    # 1b. Empty slot over a darker, cooler blurred scene (smooth gradients only)
    yy, xx = np.mgrid[0:203, 0:164].astype(np.float32)
    dark = np.stack([110 + 0.3 * xx, 125 + 0.1 * yy, 120 + 0.2 * xx], axis=-1)
    dark[:, :12, :] += np.linspace(90, 0, 12)[None, :, None]  # soft bright glare on the left
    dark_img = Image.fromarray(np.clip(dark, 0, 255).astype(np.uint8))
    assert layout.find_card_label_strip(dark_img, 1.0) is None
    assert layout.is_empty_material_slot(dark_img, 1.0) is True

    # 1c. A card whose strip colour is shifted (colour filter): no strip found, but sharp digits
    tinted = np.full((203, 164, 3), (105, 95, 153), dtype=np.uint8)
    tinted[160:197, :, :] = (200, 215, 235)
    tinted[170:190, 76:88, :] = (60, 60, 70)
    tinted_img = Image.fromarray(tinted)
    assert layout.find_card_label_strip(tinted_img, 1.0) is None
    assert layout.is_empty_material_slot(tinted_img, 1.0) is False

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

