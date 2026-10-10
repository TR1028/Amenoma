# -*- coding: utf-8 -*-
"""Data integrity tests for 7.1 ArtsInfo, MaterialInfo, and RCC JSON files."""
import json
import os
import pytest

import ArtsInfo
import MaterialInfo
import postprocess

RCC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ArtScanner', 'rcc')


def test_arts_info_set_count():
    assert len(ArtsInfo.SetNames) == 63
    assert len(ArtsInfo.Setnames_EN) == 63
    assert len(ArtsInfo.SetNamesGOOD) == 63
    assert len(ArtsInfo.SetNamesGenshinArt) == 63
    assert len(ArtsInfo.SetNamesMingyuLab) == 63


def test_arts_pieces_count():
    # 63 sets pieces + 1 consumable list
    assert len(ArtsInfo.ArtNames) == 64
    assert len(ArtsInfo.ArtNames_EN) == 64
    # Total artifact pieces indexed (excluding consumables)
    assert len(ArtsInfo.PieceIndex) == 299
    assert len(ArtsInfo.PieceIndex_EN) == 299


def test_consumables_at_end():
    consumables_chs = ['祝圣精华', '祝圣油膏', '祝圣之霜']
    consumables_en = ['Sanctifying Unction', 'Sanctifying Essence', 'Sanctifying Elixir']
    assert ArtsInfo.ConsumableNames == consumables_chs
    assert ArtsInfo.ConsumableNames_EN == consumables_en
    assert ArtsInfo.ArtNames[-1] == consumables_chs
    assert ArtsInfo.ArtNames_EN[-1] == consumables_en


def test_good_mappings():
    # All 63 sets must have a valid GOOD set key and valid rarities
    assert len(ArtsInfo.SetNamesGOOD) == 63
    for good_key in ArtsInfo.SetNamesGOOD:
        assert good_key in ArtsInfo.SetRarities
        assert isinstance(ArtsInfo.SetRarities[good_key], list)
        assert len(ArtsInfo.SetRarities[good_key]) > 0

    # 5 slots
    assert len(postprocess.LEGAL_MAIN_STATS) == 5
    assert postprocess.LEGAL_MAIN_STATS[0] == ['FIGHT_PROP_HP']
    assert postprocess.LEGAL_MAIN_STATS[1] == ['FIGHT_PROP_ATTACK']


def test_material_info():
    assert len(MaterialInfo.MaterialsNameCHS) == 907
    assert len(MaterialInfo.MaterialsNameEN) == 907
    # Check no duplicate names
    assert len(set(MaterialInfo.MaterialsNameCHS)) == 907
    assert len(set(MaterialInfo.MaterialsNameEN)) == 907


def test_rcc_json_files():
    for fname in ('artifact_sets_v71.json', 'char_map_v71.json', 'material_names_v71.json'):
        path = os.path.join(RCC_DIR, fname)
        assert os.path.exists(path), f"Missing {fname}"
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert isinstance(data, (dict, list))
            assert len(data) > 0
