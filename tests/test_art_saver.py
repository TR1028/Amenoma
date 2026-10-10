import json

import pytest

pytest.importorskip('ZODB')
pytest.importorskip('persistent')
pytest.importorskip('transaction')

import ArtsInfo  # noqa: E402
import art_saver  # noqa: E402
import art_saver_EN  # noqa: E402
import postprocess  # noqa: E402


def corrected(info, lang='CHS'):
    info = dict(info)
    info.setdefault('equipped', '')
    info.setdefault('locked', False)
    info.setdefault('astral_mark', False)
    info.setdefault('unactivated', [])
    postprocess.correct_artifact(info, lang)
    return info


GOBLET = {
    "name": "沉波之盏", "type": "空之杯", "star": 5, "level": "+20",
    "main_attr_name": "冰元素伤害加成", "main_attr_value": "46.6%",
    "subattr_1": "元素充能效率+18.1%", "subattr_2": "暴击率+7.4%",
    "subattr_3": "防御力+63", "subattr_4": "暴击伤害+6.2%",
}
FLOWER = {"level": "+20", "main_attr_name": "生命值", "main_attr_value": "4,780", "name": "野花记忆的绿野",
          "subattr_1": "元素充能效率+4.5%", "subattr_2": "攻击力+15.7%", "subattr_3": "暴击伤害+14.0%",
          "subattr_4": "元素精通+42", "type": "生之花", "star": 5}
# 6.0: +0 sands with 3 substats, the 4th one is shown greyed out
SANDS_UNACTIVATED = {
    "name": "坚铜罗盘", "type": "时之沙", "star": 5, "level": "+0",
    "main_attr_name": "攻击力", "main_attr_value": "7.0%",
    "subattr_1": "暴击率+3.1%", "subattr_2": "暴击伤害+7.0%", "subattr_3": "元素精通+19",
    "subattr_4": "防御力+5.8%（待激活）",
    "equipped": "神里绫华已装备", "locked": True, "astral_mark": True,
}


def export_good(db, location=True):
    return db.exportGOODJSON({"path": "", "exportLocation": location})


def test_legacy_examples_are_valid():
    db = art_saver.ArtDatabase()
    assert db.add(corrected(GOBLET), None, raise_error=True)
    assert db.add(corrected(FLOWER), None, raise_error=True)
    good = export_good(db)
    assert [a['setKey'] for a in good['artifacts']] == ['HeartOfDepth', 'ViridescentVenerer']
    goblet = good['artifacts'][0]
    assert goblet['mainStatKey'] == 'cryo_dmg_' and goblet['slotKey'] == 'goblet'
    assert {s['key']: s['value'] for s in goblet['substats']} == {
        'enerRech_': 18.1, 'critRate_': 7.4, 'def': 63, 'critDMG_': 6.2}
    assert 'unactivatedSubstats' not in goblet and 'astralMark' not in goblet


def test_unactivated_substat_is_kept_apart():
    info = corrected(SANDS_UNACTIVATED)
    art = art_saver.Artifact(info, None)
    assert [str(s) for s in art.substats] == ['暴击率+3.1%', '暴击伤害+7.0%', '元素精通+19']
    assert [str(s) for s in art.unactivated_substats] == ['防御力+5.8%']
    assert art.stat.type == art_saver.ArtifactStatType.FIGHT_PROP_ATTACK_PERCENT
    assert art.astral_mark and art.locked

    db = art_saver.ArtDatabase()
    assert db.add(info, None, raise_error=True)
    good = export_good(db)['artifacts'][0]
    assert good['mainStatKey'] == 'atk_'
    assert good['location'] == 'KamisatoAyaka'
    assert good['lock'] is True and good['astralMark'] is True
    assert [s['key'] for s in good['substats']] == ['critRate_', 'critDMG_', 'eleMas']
    assert good['unactivatedSubstats'] == [{'key': 'def_', 'value': 5.8}]
    # the other formats have no unactivated substats: only the 3 rolled ones are exported
    mona = db.exportGenshinArtJSON({"path": "", "exportLocation": True})
    assert len(mona['sand'][0]['normalTags']) == 3


def test_unactivated_marked_by_pixels_without_postprocess():
    # Artifact also honours ocr.detect_info's 'unactivated' keys when postprocess did not run
    info = dict(SANDS_UNACTIVATED, subattr_4='防御力+5.8%', unactivated=['subattr_4'], setid=5,
                equipped='')
    art = art_saver.Artifact(info, None)
    assert len(art.substats) == 3 and len(art.unactivated_substats) == 1


def test_impossible_unactivated_value_is_dropped():
    info = corrected(dict(SANDS_UNACTIVATED, subattr_4='防御力+9.9%（待激活）'))
    art = art_saver.Artifact(info, None)
    assert len(art.unactivated_substats) == 0
    assert len(art.substats) == 3


def test_main_stat_percent_comes_from_the_slot():
    # v3.4 decided ATK vs ATK% from the python type of the OCR value: '46.6' (no '%') became flat ATK
    info = corrected(dict(GOBLET, name='坚铜罗盘', type='时之沙', main_attr_name='攻击力', main_attr_value='46.6'))
    art = art_saver.Artifact(info, None)
    assert art.stat.type == art_saver.ArtifactStatType.FIGHT_PROP_ATTACK_PERCENT
    assert art.stat.value == pytest.approx(0.466)
    # also without postprocess (main.py / old callers): the slot decides
    del info['main_stat_key']
    assert art_saver.Artifact(info, None).stat.type == art_saver.ArtifactStatType.FIGHT_PROP_ATTACK_PERCENT


def test_substat_without_plus_sign():
    stat = art_saver.Artifact.parseSubstat('暴击率3.9%')
    assert stat.type == art_saver.ArtifactStatType.FIGHT_PROP_CRITICAL and str(stat) == '暴击率+3.9%'


def test_good_export_skips_one_and_two_star():
    hp = ArtsInfo.MainAttrValue[2]['FIGHT_PROP_HP'][0]
    two_star = corrected({"name": "冒险家之花", "type": "生之花", "star": 2, "level": "+0",
                          "main_attr_name": "生命值", "main_attr_value": str(hp)})
    db = art_saver.ArtDatabase()
    assert db.add(two_star, None, raise_error=True)
    assert export_good(db)['artifacts'] == []  # GO would import rarity 0 as a 5 star artifact
    assert len(db.exportGenshinArtJSON({"path": "", "exportLocation": True})['flower']) == 1


def test_export_location_flag(tmp_path):
    db = art_saver.ArtDatabase()
    db.add(corrected(SANDS_UNACTIVATED), None, raise_error=True)
    assert 'location' not in export_good(db, location=False)['artifacts'][0]
    path = tmp_path / 'artifacts.GOOD.json'
    db.exportGOODJSON({"path": str(path), "exportLocation": True})
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data['format'] == 'GOOD' and data['version'] == 1 and len(data['artifacts']) == 1


def test_new_set_exports_with_its_good_key():
    setid = ArtsInfo.SetNamesGOOD.index('MarechausseeHunter')
    info = corrected(dict(GOBLET, name=ArtsInfo.ArtNames[setid][3]))
    db = art_saver.ArtDatabase()
    assert db.add(info, None, raise_error=True)
    assert export_good(db)['artifacts'][0]['setKey'] == 'MarechausseeHunter'
    mingyu = db.exportGenmoCalcJSON({"path": "", "exportLocation": True})
    assert mingyu[0]['asKey'] == ArtsInfo.SetNamesMingyuLab[setid]


def test_en_artifact():
    info = corrected({
        "name": "Copper Compass", "type": "Sands of Eon", "star": 5, "level": "+0",
        "main_attr_name": "ATK", "main_attr_value": "7.0%",
        "subattr_1": "CRIT Rate+3.1%", "subattr_2": "CRIT DMG+7.0%", "subattr_3": "Elemental Mastery+19",
        "subattr_4": "DEF+5.8% (unactivated)", "equipped": "Equipped: Furina",
    }, 'EN')
    db = art_saver_EN.ArtDatabase()
    assert db.add(info, None, raise_error=True)
    good = db.exportGOODJSON({"path": "", "exportLocation": True})['artifacts'][0]
    assert good['location'] == 'Furina' and good['mainStatKey'] == 'atk_'
    assert good['unactivatedSubstats'] == [{'key': 'def_', 'value': 5.8}]
