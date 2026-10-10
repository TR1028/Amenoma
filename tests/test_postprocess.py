import pytest

import ArtsInfo
import postprocess
from postprocess import RecognitionError


def chs_info(**overrides):
    info = {
        'name': '沉波之盏', 'type': '空之杯', 'star': 5, 'level': '+20',
        'main_attr_name': '冰元素伤害加成', 'main_attr_value': '46.6%',
        'subattr_1': '元素充能效率+18.1%', 'subattr_2': '暴击率+7.4%',
        'subattr_3': '防御力+63', 'subattr_4': '暴击伤害+6.2%',
        'equipped': '', 'locked': False, 'astral_mark': False, 'unactivated': [],
    }
    info.update(overrides)
    return info


def en_info(**overrides):
    info = {
        'name': 'Goblet of Thundering Deep', 'type': 'Goblet of Eonothem', 'star': 5, 'level': '+20',
        'main_attr_name': 'Cryo DMG Bonus', 'main_attr_value': '46.6%',
        'subattr_1': 'Energy Recharge+18.1%', 'subattr_2': 'CRIT Rate+7.4%',
        'subattr_3': 'DEF+63', 'subattr_4': 'CRIT DMG+6.2%',
        'equipped': '', 'locked': False, 'astral_mark': False, 'unactivated': [],
    }
    info.update(overrides)
    return info


# ----------------------------------------------------------------------------- fuzzy matching
def test_fuzzy_match_exact_and_close():
    names = ['夏祭之花', '夏祭终末', '无垢之花']
    assert postprocess.fuzzy_match('夏祭之花', names).value == '夏祭之花'
    m = postprocess.fuzzy_match('无垢之化', names)
    assert (m.value, m.distance) == ('无垢之花', 1)


def test_fuzzy_match_rejects_far_and_ambiguous():
    assert postprocess.fuzzy_match('完全不同的名字', ['夏祭之花', '无垢之花']).value is None
    # equally close to all four circlets: never guess
    tiaras = ['祭水礼冠', '祭火礼冠', '祭雷礼冠', '祭冰礼冠']
    m = postprocess.fuzzy_match('祭?礼冠', tiaras)
    assert m.value is None and m.distance == 1 and m.runner_up == 1


def test_match_limit():
    assert postprocess.match_limit('攻击力') == 1
    assert postprocess.match_limit('HP') == 1
    assert postprocess.match_limit('Viridescent Arrow Feather') == 8


# ------------------------------------------------------------------------------------ numbers
@pytest.mark.parametrize('text, percent, expected', [
    ('46.6%', None, 0.466),
    ('4,780', None, 4780),
    ('311', None, 311),
    ('3.9', True, 0.039),
    ('39%', True, 0.039),  # decimal point lost by OCR
    ('5%', True, 0.05),
    ('1,234', False, 1234),
    ('+20', None, 20),
    (144, None, 144),
])
def test_decode_value(text, percent, expected):
    assert postprocess.decode_value(text, percent) == pytest.approx(expected)


def test_decode_value_errors():
    with pytest.raises(ValueError):
        postprocess.decode_value('abc')
    assert postprocess.try_decode_value('abc', True) is None


@pytest.mark.parametrize('text, expected', [('+20', 20), ('+2O', 20), ('+0', 0), ('20', 20), ('', None),
                                            ('+120', None), (16, 16)])
def test_parse_level(text, expected):
    assert postprocess.parse_level(text) == expected


# ----------------------------------------------------------------------------------- substats
@pytest.mark.parametrize('text, key, value, normalized, unactivated', [
    ('暴击率+3.9%', 'FIGHT_PROP_CRITICAL', 0.039, '暴击率+3.9%', False),
    ('暴击率3.9%', 'FIGHT_PROP_CRITICAL', 0.039, '暴击率+3.9%', False),  # '+' lost
    ('暴击率+39', 'FIGHT_PROP_CRITICAL', 0.039, '暴击率+3.9%', False),  # '.' and '%' lost
    ('攻击力+5.8%', 'FIGHT_PROP_ATTACK_PERCENT', 0.058, '攻击力+5.8%', False),
    ('攻击力+5.8', 'FIGHT_PROP_ATTACK_PERCENT', 0.058, '攻击力+5.8%', False),  # '%' lost
    ('攻击力+19', 'FIGHT_PROP_ATTACK', 19, '攻击力+19', False),
    ('生命值+1,016', 'FIGHT_PROP_HP', 1016, '生命值+1,016', False),
    ('元素精通+23', 'FIGHT_PROP_ELEMENT_MASTERY', 23, '元素精通+23', False),
    ('暴击伤害#7.8%', 'FIGHT_PROP_CRITICAL_HURT', 0.078, '暴击伤害+7.8%', False),
    ('暴击率+3.9%（待激活）', 'FIGHT_PROP_CRITICAL', 0.039, '暴击率+3.9%', True),
    ('防御力+5.8%(待激活)', 'FIGHT_PROP_DEFENSE_PERCENT', 0.058, '防御力+5.8%', True),
    ('暴击伤害+7.8%（待激', 'FIGHT_PROP_CRITICAL_HURT', 0.078, '暴击伤害+7.8%', True),
])
def test_parse_substat_chs(text, key, value, normalized, unactivated):
    sub = postprocess.parse_substat(text, 'CHS')
    assert sub is not None
    assert sub.key == key
    assert sub.value == pytest.approx(value)
    assert sub.text == normalized
    assert sub.unactivated == unactivated


@pytest.mark.parametrize('text, key, value, unactivated', [
    ('CRIT Rate+3.9%', 'FIGHT_PROP_CRITICAL', 0.039, False),
    ('CRIT Rate+3.9% (unactivated)', 'FIGHT_PROP_CRITICAL', 0.039, True),
    ('Elemental Mastery+19 (Unactivated)', 'FIGHT_PROP_ELEMENT_MASTERY', 19, True),
    ('ATK+5.8%', 'FIGHT_PROP_ATTACK_PERCENT', 0.058, False),
    ('ATK+19', 'FIGHT_PROP_ATTACK', 19, False),
    ('HP 239', 'FIGHT_PROP_HP', 239, False),
    ('Energy Recharge+6.5%', 'FIGHT_PROP_CHARGE_EFFICIENCY', 0.065, False),
])
def test_parse_substat_en(text, key, value, unactivated):
    sub = postprocess.parse_substat(text, 'EN')
    assert (sub.key, sub.unactivated) == (key, unactivated)
    assert sub.value == pytest.approx(value)


@pytest.mark.parametrize('text', ['', '+3.9%', '暴击率', '某某属性+3.9%'])
def test_parse_substat_rejects(text):
    assert postprocess.parse_substat(text, 'CHS') is None


# --------------------------------------------------------------------------------- main stats
@pytest.mark.parametrize('text, slot, expected', [
    ('生命值', 0, 'FIGHT_PROP_HP'),
    ('乱码', 0, 'FIGHT_PROP_HP'),  # the flower only has flat HP
    ('攻击力', 1, 'FIGHT_PROP_ATTACK'),
    ('攻击力', 2, 'FIGHT_PROP_ATTACK_PERCENT'),  # sands: always ATK%
    ('生命值', 3, 'FIGHT_PROP_HP_PERCENT'),
    ('防御力', 4, 'FIGHT_PROP_DEFENSE_PERCENT'),
    ('元素充能效率', 2, 'FIGHT_PROP_CHARGE_EFFICIENCY'),
    ('冰元素伤害加成', 3, 'FIGHT_PROP_ICE_ADD_HURT'),
    ('治疗加成', 4, 'FIGHT_PROP_HEAL_ADD'),
    ('暴击率', 4, 'FIGHT_PROP_CRITICAL'),
    ('暴击率', 2, None),  # not possible on sands
    ('冰元素伤害加成', 4, None),  # not possible on a circlet
])
def test_resolve_main_stat(text, slot, expected):
    assert postprocess.resolve_main_stat(text, slot, 'CHS')[0] == expected


def test_legal_main_stats_have_values():
    for slot, keys in enumerate(postprocess.LEGAL_MAIN_STATS):
        for key in keys:
            assert key in ArtsInfo.MainAttrValue[5], key
            assert key in ArtsInfo.MainAttrNames and key in ArtsInfo.MainAttrNames_EN


# ----------------------------------------------------------------------------------- equipped
@pytest.mark.parametrize('text, expected', [
    ('', ''),
    ('神里绫华已装备', '神里绫华'),
    ('雷电将车已装备', '雷电将军'),  # 1 OCR error in the name
    ('茜特菈莉已装备', '茜特菈莉'),  # character added after 3.4
    ('某个玩家昵称已装备', 'Traveler'),  # the Traveler shows the player's nickname
    ('随便什么文字', ''),
])
def test_match_equipped_chs(text, expected):
    assert postprocess.match_equipped(text, 'CHS')[0] == expected


@pytest.mark.parametrize('text, expected', [
    ('', ''),
    ('Equipped: Kamisato Ayaka', 'Kamisato Ayaka'),
    ('Equipped: Raiden Shogum', 'Raiden Shogun'),
    ('Equipped: Furina', 'Furina'),
    ('Equipped: SomeNickname123', 'Traveler'),
    ('Equipped:Furina', 'Furina'),
])
def test_match_equipped_en(text, expected):
    assert postprocess.match_equipped(text, 'EN')[0] == expected


@pytest.mark.parametrize('text, lang', [('神里绫化已装备', 'CHS'), ('Equipped: Kamisato Ayako', 'EN')])
def test_match_equipped_tie_is_not_guessed(text, lang):
    # one edit away from both Ayaka and Ayato: v3.4 picked whichever came first
    location, match = postprocess.match_equipped(text, lang)
    assert match.value is None and match.distance == match.runner_up == 1
    assert location == 'Traveler'


def test_equipped_threshold_ignores_the_suffix():
    # v3.4 compared the distance with len('<name>已装备') // 3 = 2 for a 4 character name, so half of
    # the name could be wrong; now only the name part counts (limit 1 for 4 characters)
    location, match = postprocess.match_equipped('雷电XX已装备', 'CHS')
    assert match.candidate == '雷电将军' and match.distance == 2
    assert location == 'Traveler'


# ---------------------------------------------------------------------------- correct_artifact
def test_correct_artifact_chs():
    info = chs_info(name='沉波之盖', type='空之杯', equipped='神里绫华已装备')
    warnings = postprocess.correct_artifact(info, 'CHS')
    assert info['name'] == '沉波之盏'
    assert info['setid'] == ArtsInfo.SetNamesGOOD.index('HeartOfDepth')
    assert info['set_key'] == 'HeartOfDepth'
    assert info['slot'] == 3 and info['type'] == '空之杯'
    assert info['level'] == 20 and info['star'] == 5
    assert info['main_stat_key'] == 'FIGHT_PROP_ICE_ADD_HURT'
    assert info['equipped'] == '神里绫华'
    assert info['unactivated_substats'] == []
    assert info['consumable'] is False
    assert any('沉波之盖' in w for w in warnings)


def test_correct_artifact_new_set():
    # a set released after 3.4 (Marechaussee Hunter)
    piece = ArtsInfo.ArtNames[ArtsInfo.SetNamesGOOD.index('MarechausseeHunter')][2]
    info = chs_info(name=piece, type='时之沙', main_attr_name='攻击力', main_attr_value='46.6%',
                    subattr_1='暴击率+3.9%', subattr_2='暴击伤害+7.8%', subattr_3='元素精通+23',
                    subattr_4='防御力+23')
    postprocess.correct_artifact(info, 'CHS')
    assert info['set_key'] == 'MarechausseeHunter'
    assert info['main_stat_key'] == 'FIGHT_PROP_ATTACK_PERCENT'


def test_correct_artifact_unknown_name_fails():
    with pytest.raises(RecognitionError):
        postprocess.correct_artifact(chs_info(name='完全不认识的圣遗物'), 'CHS')


def test_correct_artifact_type_line_breaks_name_tie():
    # '夏祭之?' is 1 edit away from the flower / sands / circlet of the same set
    info = chs_info(name='夏祭之X', type='时之沙', main_attr_name='攻击力', main_attr_value='46.6%')
    postprocess.correct_artifact(info, 'CHS')
    assert info['name'] == '夏祭之刻' and info['slot'] == 2


def test_correct_artifact_type_mismatch_trusts_name():
    info = chs_info(type='生之花')
    warnings = postprocess.correct_artifact(info, 'CHS')
    assert info['type'] == '空之杯' and info['slot'] == 3
    assert any('trusting the name' in w for w in warnings)


@pytest.mark.parametrize('star', [0, 6, None, 3])
def test_correct_artifact_bad_rarity(star):
    # Heart of Depth only exists as 4 and 5 star
    with pytest.raises(RecognitionError):
        postprocess.correct_artifact(chs_info(star=star), 'CHS')


def test_correct_artifact_illegal_main_stat():
    with pytest.raises(RecognitionError):
        postprocess.correct_artifact(chs_info(main_attr_name='暴击率'), 'CHS')  # crit rate goblet


def test_correct_artifact_level_from_main_value():
    info = chs_info(level='+?')
    warnings = postprocess.correct_artifact(info, 'CHS')
    assert info['level'] == 20
    assert any('from the main stat value' in w for w in warnings)


def test_correct_artifact_duplicate_substat_fails():
    with pytest.raises(RecognitionError):
        postprocess.correct_artifact(chs_info(subattr_4='暴击率+3.9%'), 'CHS')


def test_correct_artifact_unactivated_by_text():
    info = chs_info(level='+0', main_attr_value='7.0%', subattr_1='元素充能效率+5.2%', subattr_2='暴击率+3.1%',
                    subattr_3='防御力+19', subattr_4='暴击伤害+7.8%（待激活）')
    info['main_attr_value'] = '7.0%'
    postprocess.correct_artifact(info, 'CHS')
    assert 'subattr_4' not in info
    assert info['unactivated_substats'] == ['暴击伤害+7.8%']
    assert info['subattr_1'] == '元素充能效率+5.2%'


def test_correct_artifact_unactivated_by_pixels():
    # old vocabulary: the marker itself is not readable, the grey line was flagged by panel_detect
    info = chs_info(level='+0', main_attr_value='7.0%', subattr_1='元素充能效率+5.2%', subattr_2='暴击率+3.1%',
                    subattr_3='防御力+19', subattr_4='暴击伤害+7.8%', unactivated=['subattr_4'])
    postprocess.correct_artifact(info, 'CHS')
    assert 'subattr_4' not in info
    assert info['unactivated_substats'] == ['暴击伤害+7.8%']


@pytest.mark.parametrize('name', ['祝圣精华', '祝圣油膏', '祝圣之霜'])
def test_correct_artifact_consumables(name):
    info = {'name': name, 'type': '', 'star': 4, 'level': '', 'main_attr_name': '', 'main_attr_value': ''}
    postprocess.correct_artifact(info, 'CHS')
    assert info['consumable'] is True


def test_correct_artifact_en():
    info = en_info(name='Goblet of Thundering Deap', equipped='Equipped: Furina',
                   subattr_4='CRIT DMG+6.2% (unactivated)', level='+0', main_attr_value='7.0%')
    postprocess.correct_artifact(info, 'EN')
    assert info['name'] == 'Goblet of Thundering Deep'
    assert info['set_key'] == 'HeartOfDepth'
    assert info['equipped'] == 'Furina'
    assert info['unactivated_substats'] == ['CRIT DMG+6.2%']
    assert info['consumable'] is False


def test_correct_artifact_en_elixir():
    info = {'name': 'Sanctifying Elixir', 'type': '', 'star': 5, 'level': '', 'main_attr_name': '',
            'main_attr_value': ''}
    postprocess.correct_artifact(info, 'EN')
    assert info['consumable'] is True


def test_every_piece_round_trips():
    for lang, table in (('CHS', ArtsInfo.PieceIndex), ('EN', ArtsInfo.PieceIndex_EN)):
        for piece, (setid, slot) in table.items():
            m = postprocess.fuzzy_match(piece, postprocess.tables(lang).pieces)
            assert m.value == piece
