"""Post-processing of the artifact OCR result (pure python: ArtsInfo + Levenshtein, no GUI / TF).

The v3.4 ``utils.*_auto_correct`` helpers always snapped an OCR string to the closest known name,
however far away it was.  With sets / characters missing from the data (or characters missing from
the OCR vocabulary) that silently produced wrong artifacts, or an IndexError.  Here:

* every fuzzy match has a distance limit *and* must be unambiguous (the runner-up must be farther
  away), otherwise the field is unknown and :func:`correct_artifact` raises
  :class:`RecognitionError`; the scanner then counts the artifact as failed and keeps its screenshot;
* the slot comes from the piece name (``ArtsInfo.PieceIndex``) and is cross-checked with the type
  line; the rarity must exist for the set (``ArtsInfo.SetRarities``);
* the main stat must be legal for the slot (``LEGAL_MAIN_STATS``): ATK / HP / DEF main stats are flat
  on the flower / plume and percentages on sands / goblet / circlet, whatever the value OCR says;
* substat lines are parsed with a regex that tolerates a missing '+' / '%' / decimal point and the
  6.0 unactivated suffix ("（待激活）" / "(unactivated)", also cut off); any other text after the value
  rejects the line (a misread).  Unactivated substats are returned separately so that they are
  exported as GOOD ``unactivatedSubstats`` instead of being counted as rolled; a bad unactivated line
  is only dropped (with a warning), it never fails the artifact.
"""
import re
from collections import namedtuple

import Levenshtein

try:  # imported as ArtScanner.postprocess ...
    from . import ArtsInfo
except ImportError:  # ... or as a top-level module (the app, PyInstaller builds, tests)
    import ArtsInfo

# ------------------------------------------------------------------------------------ game rules
_COMMON_MAIN = ['FIGHT_PROP_HP_PERCENT', 'FIGHT_PROP_ATTACK_PERCENT', 'FIGHT_PROP_DEFENSE_PERCENT',
                'FIGHT_PROP_ELEMENT_MASTERY']
_DMG_BONUS = ['FIGHT_PROP_PHYSICAL_ADD_HURT', 'FIGHT_PROP_ROCK_ADD_HURT', 'FIGHT_PROP_WIND_ADD_HURT',
              'FIGHT_PROP_ICE_ADD_HURT', 'FIGHT_PROP_WATER_ADD_HURT', 'FIGHT_PROP_FIRE_ADD_HURT',
              'FIGHT_PROP_ELEC_ADD_HURT', 'FIGHT_PROP_GRASS_ADD_HURT']
# main stats that can roll on each slot, indexed like ArtsInfo.TypeNames (flower .. circlet)
LEGAL_MAIN_STATS = [
    ['FIGHT_PROP_HP'],  # flower
    ['FIGHT_PROP_ATTACK'],  # plume
    _COMMON_MAIN + ['FIGHT_PROP_CHARGE_EFFICIENCY'],  # sands
    _COMMON_MAIN + _DMG_BONUS,  # goblet
    _COMMON_MAIN + ['FIGHT_PROP_CRITICAL', 'FIGHT_PROP_CRITICAL_HURT', 'FIGHT_PROP_HEAL_ADD'],  # circlet
]

# substats that are always / never percentages; ATK / HP / DEF exist as both
_ALWAYS_PERCENT = {'FIGHT_PROP_CRITICAL', 'FIGHT_PROP_CRITICAL_HURT', 'FIGHT_PROP_CHARGE_EFFICIENCY'}
_NEVER_PERCENT = {'FIGHT_PROP_ELEMENT_MASTERY'}

# 6.0+: the greyed out 4th substat of a +0..+3 artifact ends with "（待激活）" / " (unactivated)"
UNACTIVATED_MARKERS = ('待激活', 'unactivated')

EQUIPPED_SUFFIX_CHS = '已装备'
EQUIPPED_PREFIX_EN = 'Equipped: '

# the Traveler is shown with the player's nickname: an equipped line with an unknown name is the Traveler
TRAVELER = 'Traveler'


class RecognitionError(ValueError):
    """The OCR result does not map to a known artifact: it is counted as failed instead of guessed."""


def is_percent_key(key):
    """True for FIGHT_PROP_* stats displayed as a percentage (see ArtsInfo.Formats)."""
    return ArtsInfo.Formats[key].endswith('%}')


def format_stat(name, key, value):
    """'暴击率', 'FIGHT_PROP_CRITICAL', 0.039 -> '暴击率+3.9%' (same text as ArtifactStat.__str__)."""
    return name + '+' + ArtsInfo.Formats[key].format(value + 1e-5)


# ------------------------------------------------------------------------------- fuzzy matching
Match = namedtuple('Match', ['value', 'candidate', 'distance', 'runner_up'])
Match.__doc__ = """Result of fuzzy_match: ``value`` is the accepted candidate or None; ``candidate`` /
``distance`` describe the closest candidate and ``runner_up`` the distance of the second closest."""


def match_limit(candidate, ratio=1 / 3):
    """Largest accepted edit distance: ``len(candidate) // 3`` (at least 1).

    With the ambiguity check below, a 4-character CHS piece name accepts at most 1 edit (not 2,
    which could turn e.g. '无垢之花' into '夏祭之花'), while longer EN piece names still accept
    2-3 edits.
    """
    return max(1, int(len(candidate) * ratio))


def fuzzy_match(text, candidates, ratio=1 / 3, max_distance=None):
    """Closest candidate if it is within ``match_limit`` (or ``max_distance``) and unambiguous.

    Unlike the v3.4 auto-correct this never snaps far-off or tied strings: e.g. "祭?礼冠" is as close
    to 祭水礼冠 as to 祭火礼冠 / 祭雷礼冠 / 祭冰礼冠 and is rejected.
    """
    text = (text or '').strip()
    unique = list(dict.fromkeys(candidates))
    if text in unique:
        return Match(text, text, 0, None)
    best = best_d = second_d = None
    for candidate in unique:
        d = Levenshtein.distance(text, candidate)
        if best_d is None or d < best_d:
            best, best_d, second_d = candidate, d, best_d
        elif second_d is None or d < second_d:
            second_d = d
    if best is None:
        return Match(None, None, None, None)
    limit = match_limit(best, ratio) if max_distance is None else max_distance
    accepted = best_d <= limit and (second_d is None or second_d > best_d)
    return Match(best if accepted else None, best, best_d, second_d)


# --------------------------------------------------------------------------- language tables
class _Tables(object):
    def __init__(self, lang):
        chs = lang == 'CHS'
        self.lang = lang
        self.piece_index = ArtsInfo.PieceIndex if chs else ArtsInfo.PieceIndex_EN
        self.consumables = list(ArtsInfo.ConsumableNames if chs else ArtsInfo.ConsumableNames_EN)
        self.pieces = list(self.piece_index) + self.consumables
        self.set_names = ArtsInfo.SetNames if chs else ArtsInfo.Setnames_EN
        self.types = list(ArtsInfo.TypeNames if chs else ArtsInfo.TypeNames_EN)
        self.main_stats = ArtsInfo.MainAttrNames if chs else ArtsInfo.MainAttrNames_EN
        self.sub_stats = ArtsInfo.SubAttrNames if chs else ArtsInfo.SubAttrNames_EN
        # display name -> FIGHT_PROP key without the _PERCENT suffix
        self.sub_base_keys = {v: k for k, v in self.sub_stats.items() if not k.endswith('_PERCENT')}
        self.users = ArtsInfo.UsersCHS if chs else ArtsInfo.UsersEN


_TABLES = {}


def normalize_lang(lang):
    value = (lang or 'CHS').upper()
    if value in ('CHS', 'ZH', 'ZH_CN', 'CN'):
        return 'CHS'
    if value in ('EN', 'ENG'):
        return 'EN'
    raise ValueError('unsupported language %r (CHS / EN)' % (lang,))


def tables(lang='CHS'):
    lang = normalize_lang(lang)
    if lang not in _TABLES:
        _TABLES[lang] = _Tables(lang)
    return _TABLES[lang]


# ----------------------------------------------------------------------------------- numbers
_NUMBER_RE = re.compile(r'\d[\d.,]*')
_DECIMALS_RE = re.compile(r'^(.*?)[.,](\d+)$')


def _number_parts(number):
    """'1,234' -> ('1234', None); '3.9' -> ('3', '9'); '46.6' -> ('46', '6').

    Three digits after the last separator make it a thousands separator, otherwise decimals.
    """
    number = number.strip('.,')
    m = _DECIMALS_RE.match(number)
    if m and len(m.group(2)) != 3:
        return re.sub(r'\D', '', m.group(1)) or '0', m.group(2)
    return re.sub(r'\D', '', number), None


def decode_value(text, percent=None):
    """Displayed value -> number: '46.6%' -> 0.466, '4,780' -> 4780, ('3.9', percent=True) -> 0.039.

    ``percent=None`` decides from the text ('%' or a decimal point).  Game percentages always have one
    decimal, so a percentage read without its decimal point gets one back: ('39', True) -> 0.039.
    """
    if isinstance(text, bool):
        raise ValueError('not a value: %r' % (text,))
    if isinstance(text, (int, float)):
        return text
    s = str(text)
    m = _NUMBER_RE.search(s)
    if not m:
        raise ValueError('no number in %r' % (text,))
    integer, decimals = _number_parts(m.group(0))
    if percent is None:
        percent = '%' in s[m.end():] or decimals is not None
    if not percent:
        return int(integer)
    if decimals is None:
        integer, decimals = (integer[:-1] or '0', integer[-1]) if len(integer) >= 2 else (integer, '0')
    return round(float(integer + '.' + decimals) / 100, 6)


def try_decode_value(text, percent):
    try:
        return decode_value(text, percent)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------------- substats
SubStat = namedtuple('SubStat', ['key', 'name', 'value', 'text', 'unactivated'])
SubStat.__doc__ = """A parsed substat line: FIGHT_PROP ``key`` (with _PERCENT for ATK% / HP% / DEF%),
display ``name``, numeric ``value`` (0.039 for 3.9%), normalised ``text`` ('暴击率+3.9%') and whether
the line carried the 6.0 unactivated marker."""


def strip_unactivated(text):
    """-> (text without the 6.0 unactivated suffix, whether there was one)."""
    low = text.lower()
    cut = min([i for i in (low.find(m) for m in UNACTIVATED_MARKERS) if i >= 0] or [-1])
    if cut < 0:
        return text, False
    return text[:cut].rstrip(' \t(（'), True


def is_marker_fragment(text):
    """'(unactivat', '（待激', '(unactlvated)' ...: the start of a (cut off / slightly misread) marker."""
    core = text.strip().lstrip('(（').rstrip(')）').strip().lower()
    if len(core) < 2:
        return False
    for marker in UNACTIVATED_MARKERS:
        prefix = marker[:len(core)]
        if Levenshtein.distance(core, prefix) <= len(core) // 4:
            return True
    return False


# well formed substat values: percentages have one decimal (OCR may lose the point: '39' = 3.9%),
# flat values are integers with an optional thousands separator
_PERCENT_VALUE_RE = re.compile(r'^(\d{1,2}[.,]\d|\d{1,3})$')
_FLAT_VALUE_RE = re.compile(r'^(\d{1,3},\d{3}|\d{1,4})$')


def parse_substat(text, lang='CHS'):
    """Parse one substat line ('暴击率+3.9%', 'CRIT Rate+3.9% (unactivated)', 'ATK 19', ...).

    After the value only a '%' and a bracketed suffix (the 6.0 unactivated marker, possibly cut off
    by the crop) may follow; any other trailing text means the line was misread and it is rejected.

    :return: SubStat, or None if the line can not be parsed reliably
    """
    t = tables(lang)
    s = (text or '').replace('#', '+').replace('＋', '+').replace('％', '%')
    s, unactivated = strip_unactivated(s)
    s = re.sub(r'^\W+', '', s.strip())  # bullet / noise before the name
    m = _NUMBER_RE.search(s)
    if not m:
        return None
    name_text = s[:m.start()].strip().rstrip('+').strip()
    rest = s[m.end():].strip()
    has_percent = rest.startswith('%')
    if has_percent:
        rest = rest[1:].strip()
    if rest:
        if rest[0] not in '(（':
            return None
        if is_marker_fragment(rest):
            unactivated = True
    match = fuzzy_match(name_text, list(t.sub_base_keys))
    if match.value is None:
        return None
    base = t.sub_base_keys[match.value]
    number = m.group(0)
    integer, decimals = _number_parts(number)
    if base in _ALWAYS_PERCENT:
        percent = True
    elif base in _NEVER_PERCENT:
        percent = False
    else:  # ATK / HP / DEF: '%' or a decimal point make it ATK% / HP% / DEF%
        percent = has_percent or decimals is not None
    if not (_PERCENT_VALUE_RE if percent else _FLAT_VALUE_RE).match(number):
        return None
    key = base + '_PERCENT' if percent and base + '_PERCENT' in t.sub_stats else base
    value = decode_value(number, percent)
    return SubStat(key, match.value, value, format_stat(match.value, key, value), unactivated)


# ------------------------------------------------------------------------------- other fields
def resolve_main_stat(text, slot, lang='CHS'):
    """Main stat name -> (FIGHT_PROP key legal for ``slot``, Match).

    The key is None if the text is not (unambiguously) one of the main stats of the slot.  The flower
    and the plume have a single possible main stat, which is returned whatever the text says.
    """
    t = tables(lang)
    legal = LEGAL_MAIN_STATS[slot]
    by_name = {t.main_stats[k]: k for k in legal}
    m = fuzzy_match(text, list(by_name))
    if m.value is not None:
        return by_name[m.value], m
    if len(legal) == 1:
        return legal[0], m
    return None, m


def split_equipped(text, lang='CHS'):
    """Character name part of the "equipped" line, or None if the line does not say equipped.

    CHS: '<name>已装备'; EN: 'Equipped: <name>'.
    """
    text = (text or '').strip()
    if normalize_lang(lang) == 'CHS':
        if len(text) < 3 or Levenshtein.distance(text[-3:], EQUIPPED_SUFFIX_CHS) > 1:
            return None
        return text[:-3].strip()
    if len(text) < 10:
        return None
    colon = text.find(':')
    if 0 <= colon <= 12 and Levenshtein.distance(text[:colon + 1], EQUIPPED_PREFIX_EN.strip()) <= 3:
        return text[colon + 1:].strip()
    if Levenshtein.distance(text[:10], EQUIPPED_PREFIX_EN) > 4:
        return None
    return text[10:].strip()


def match_equipped(text, lang='CHS'):
    """-> (location, Match): '' if not equipped, the character's display name (key of
    ArtsInfo.UsersCHS / UsersEN), or 'Traveler' when the name is not a known character (the
    Traveler is shown with the player's nickname).  The distance limit only looks at the name, not at
    the '已装备' / 'Equipped: ' part (v3.4 counted that part as well, so short names snapped easily).
    """
    name = split_equipped(text, lang)
    if name is None:
        return '', None
    m = fuzzy_match(name, tables(lang).users)
    if m.value is not None:
        return m.value, m
    return TRAVELER, m


_LEVEL_TRANSLATION = {ord(k): v for k, v in {'O': '0', 'o': '0', 'D': '0', 'l': '1', 'I': '1', 'i': '1',
                                               '|': '1', 'S': '5', 's': '5', 'B': '8', 'Z': '2',
                                               'z': '2'}.items()}


def parse_level(text):
    """'+20' -> 20, '+2O' -> 20; None if there is no 1-2 digit number."""
    if isinstance(text, int) and not isinstance(text, bool):
        return text
    digits = re.sub(r'\D', '', str(text if text is not None else '').translate(_LEVEL_TRANSLATION))
    if not digits or len(digits) > 2:
        return None
    return int(digits)


def main_value_levels(key, rarity, value):
    """Levels at which the main stat ``key`` of a ``rarity`` star artifact shows ``value``."""
    table = ArtsInfo.MainAttrValue.get(rarity, {}).get(key)
    if not table or value is None:
        return []
    tol = 0.0006 if is_percent_key(key) else 0.6
    return [level for level, v in enumerate(table) if abs(v - value) <= tol]


# ------------------------------------------------------------------------------------ artifact
def correct_artifact(info, lang='CHS'):
    """Validate and normalise an OCR result in place (replaces UIMain.autoCorrect).

    Sets ``name`` (piece name) and ``consumable``; for artifacts also ``setid``, ``set_key`` (GOOD),
    ``slot`` (0-4), ``type``, ``star``, ``level`` (int), ``equipped`` ('' / character display name /
    'Traveler'), ``main_attr_name``, ``main_stat_key``, the normalised ``subattr_N`` of the active
    substats and ``unactivated_substats`` (normalised texts; their ``subattr_N`` keys are removed).

    :return: list of warnings about recoverable inconsistencies (already corrected)
    :raise RecognitionError: a field can not be recognised unambiguously
    """
    t = tables(lang)
    warnings = []

    raw_name = info.get('name', '')
    m = fuzzy_match(raw_name, t.pieces)
    if m.value is None:
        # ambiguous between pieces of different slots: a clearly read type line can settle it
        tm = fuzzy_match(info.get('type', ''), t.types)
        if tm.value is not None:
            slot_guess = t.types.index(tm.value)
            same_slot = fuzzy_match(raw_name, [p for p, (_, s) in t.piece_index.items() if s == slot_guess])
            if same_slot.value is not None:
                warnings.append('name %r matched among the %s pieces only' % (raw_name, tm.value))
                m = same_slot
    if m.value is None:
        raise RecognitionError('unknown artifact name %r (closest %r, distance %s)'
                               % (raw_name, m.candidate, m.distance))
    if m.distance:
        warnings.append('name %r -> %r (distance %d)' % (raw_name, m.value, m.distance))
    info['name'] = m.value
    info['consumable'] = m.value in t.consumables
    if info['consumable']:
        return warnings

    setid, slot = t.piece_index[m.value]
    info['setid'] = setid
    info['set_key'] = ArtsInfo.SetNamesGOOD[setid]

    raw_type = info.get('type', '')
    tm = fuzzy_match(raw_type, t.types)
    if tm.value is None:
        warnings.append('type %r not recognised, %r is a %s' % (raw_type, m.value, t.types[slot]))
    elif t.types.index(tm.value) != slot:
        warnings.append('type %r does not match %r (a %s), trusting the name' % (raw_type, m.value, t.types[slot]))
    info['slot'] = slot
    info['type'] = t.types[slot]

    try:
        star = int(info.get('star'))
    except (TypeError, ValueError):
        star = None
    allowed = ArtsInfo.SetRarities.get(info['set_key'])
    if star is None or not 1 <= star <= 5 or (allowed and star not in allowed):
        raise RecognitionError('%r star(s) is not possible for %s (%s)' % (info.get('star'), t.set_names[setid], allowed))
    info['star'] = star

    raw_main = info.get('main_attr_name', '')
    key, mm = resolve_main_stat(raw_main, slot, lang)
    if key is None:
        raise RecognitionError('main stat %r is not possible on a %s (closest %r)' % (raw_main, t.types[slot], mm.candidate))
    if mm.value is None or mm.distance:
        warnings.append('main stat %r -> %s' % (raw_main, t.main_stats[key]))
    info['main_stat_key'] = key
    info['main_attr_name'] = t.main_stats[key]

    raw_level = info.get('level')
    level = parse_level(raw_level)
    main_value = try_decode_value(info.get('main_attr_value'), is_percent_key(key))
    levels = main_value_levels(key, star, main_value)
    if level is None or level > ArtsInfo.RarityToMaxLvs[star - 1]:
        if len(levels) != 1:
            raise RecognitionError('level %r not recognised' % (raw_level,))
        warnings.append('level %r -> +%d (from the main stat value %r)'
                        % (raw_level, levels[0], info.get('main_attr_value')))
        level = levels[0]
    elif levels and level not in levels:
        warnings.append('main stat value %r does not match +%d (it does at %s)'
                        % (info.get('main_attr_value'), level, levels))
    info['level'] = level

    raw_equipped = info.get('equipped', '')
    equipped, em = match_equipped(raw_equipped, lang)
    if em is not None and em.value is None:
        warnings.append('equipped %r: unknown character, assuming the Traveler' % (raw_equipped,))
    info['equipped'] = equipped

    marked = set(info.get('unactivated') or [])
    unactivated, seen = [], set()
    for tag in sorted(k for k in list(info) if k.startswith('subattr_')):
        text = info[tag]
        sub = parse_substat(text, lang)
        if tag in marked or strip_unactivated(text or '')[1] or (sub is not None and sub.unactivated):
            # 6.0 grey line: not rolled yet, exported apart; a bad read only loses this line
            del info[tag]
            if sub is None:
                warnings.append('unactivated substat %r not recognised, dropped' % (text,))
            elif sub.key == key or sub.key in seen:
                warnings.append('unactivated substat %r repeats the main stat or another substat, dropped'
                                % (text,))
            else:
                seen.add(sub.key)
                unactivated.append(sub.text)
            continue
        if sub is None:
            raise RecognitionError('substat %r not recognised' % (text,))
        if sub.key == key or sub.key in seen:
            raise RecognitionError('substat %r repeats the main stat or another substat' % (text,))
        seen.add(sub.key)
        info[tag] = sub.text
    if unactivated and level >= 4:
        warnings.append('unactivated substat on a +%d artifact' % level)
    info['unactivated_substats'] = unactivated
    return warnings
