# -*- coding: utf-8 -*-
"""Unit tests for ArtScanner.charset_util: vocabulary loading, geometry, decoding, and compatibility."""
import json
import os
import tempfile
import numpy as np
import pytest

import charset_util
from charset_util import Charset, CharsetError, can_read_unactivated, resolve_charset

RCC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ArtScanner', 'rcc')


def test_charset_basic_properties():
    cs = Charset(['A', 'B', 'C'], width=240, height=16, max_length=15)
    assert len(cs) == 3
    assert cs.num_classes == 5  # mask(0) + 3 chars + blank(4)
    assert cs.mask_index == 0
    assert cs.blank_index == 4
    assert cs.indices_to_text([1, 2, 3]) == 'ABC'
    assert cs.indices_to_text([0, 1, 4, 2, 99]) == 'AB'


def test_charset_validation():
    with pytest.raises(CharsetError, match='empty charset'):
        Charset([], 240, 16, 15)

    with pytest.raises(CharsetError, match='duplicate'):
        Charset(['A', 'B', 'A'], 240, 16, 15)

    with pytest.raises(CharsetError, match='non-empty strings'):
        Charset(['A', ''], 240, 16, 15)


def test_greedy_decode():
    # Charset: 1='A', 2='B', blank=3
    cs = Charset(['A', 'B'], width=10, height=16, max_length=10)
    # 4 classes: 0=mask, 1=A, 2=B, 3=blank
    # Batch of 2, time=4
    logits = np.zeros((2, 4, 4), dtype=np.float32)
    # Sequence 1: A, A, blank, B -> "AB"
    logits[0, 0, 1] = 10.0  # A
    logits[0, 1, 1] = 10.0  # A
    logits[0, 2, 3] = 10.0  # blank
    logits[0, 3, 2] = 10.0  # B

    # Sequence 2: blank, B, B, blank -> "B"
    logits[1, 0, 3] = 10.0  # blank
    logits[1, 1, 2] = 10.0  # B
    logits[1, 2, 2] = 10.0  # B
    logits[1, 3, 3] = 10.0  # blank

    decoded = cs.greedy_decode(logits)
    assert decoded == ['AB', 'B']


def test_can_read_unactivated():
    assert not can_read_unactivated(['暴', '击', '率', '+', '3', '.', '9', '%'], 'CHS')
    assert can_read_unactivated(['暴', '击', '率', '待', '激', '活', '（', '）'], 'CHS')

    assert not can_read_unactivated(['C', 'R', 'I', 'T', ' '], 'EN')
    assert can_read_unactivated(list('(unactivated)CRIT Rate+3.9%'), 'EN')


def test_charset_serialization_roundtrip():
    cs = Charset(['x', 'y', 'z'], width=320, height=16, max_length=24, meta={'lang': 'chs'})
    d = cs.to_dict()
    assert d['num_classes'] == 5
    assert d['max_length'] == 24

    cs2 = Charset.from_dict(d)
    assert cs2.characters == ['x', 'y', 'z']
    assert cs2.width == 320
    assert cs2.meta.get('lang') == 'chs'


def test_shipped_charset_files_exist_and_match():
    # Verify CHS and EN artifact & material charset json files in rcc/
    chs_art_path = os.path.join(RCC_DIR, 'models_CHS', 'charset_chs_art.json')
    chs_mat_path = os.path.join(RCC_DIR, 'models_CHS', 'charset_chs_mat.json')
    en_art_path = os.path.join(RCC_DIR, 'models_EN', 'charset_en_art.json')
    en_mat_path = os.path.join(RCC_DIR, 'models_EN', 'charset_en_mat.json')

    for p in (chs_art_path, chs_mat_path, en_art_path, en_mat_path):
        assert os.path.isfile(p), f"Missing charset file: {p}"
        cs = Charset.from_json(p)
        assert len(cs) > 50
        assert cs.width in (240, 384, 512)
        assert cs.height == 16
