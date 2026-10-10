# -*- coding: utf-8 -*-
"""Shared, TensorFlow-free helpers for the Amenoma OCR model trainers.

This module is imported by the four ``train_model_*.py`` scripts, by
``trainer_tf.py`` and by the multiprocessing data workers.  It deliberately does
NOT import TensorFlow so that data-generation worker processes stay light.

What lives here
---------------
* paths of the shared assets (``genshin.ttf``, Reliquary*ExcelConfigData.json)
* charset helpers + the companion ``charset_<lang>_<kind>.json`` format shared
  with ``ArtScanner`` (see ``ArtScanner/rcc/models_*/charset_*.json``)::

      {"charset": [c1, ..., cN], "num_classes": N + 2, "mask_index": 0,
       "blank_index": N + 1, "width": W, "height": H, "max_length": L,
       "model": "<file>.h5", "lang": "CHS"|"EN", "kind": "artifact"|"material",
       "target_version": "x.y.z", "source": "..."}

  Index semantics (identical to the historical ``StringLookup(mask_token="",
  num_oov_indices=0)`` + ``keras.backend.ctc_*`` setup): class 0 is the
  mask/padding token, class ``i`` (1..N) is ``charset[i - 1]`` and class
  ``N + 1`` (the last one) is the CTC blank.
* a NumPy label codec / greedy CTC decoder (no ``StringLookup`` needed)
* glyph-coverage checks for ``genshin.ttf`` with fallback fonts
  (genshin.ttf's ``.notdef`` glyph is EMPTY, so a missing glyph would silently
  render as blank space while the label still contains the character)
* synthetic text-line rendering and the exact preprocessing used at inference
  time by ``ArtScanner/ocr*.py``
* artifact / material sample generators (incl. the 6.0 "(待激活)" /
  "(unactivated)" 4th substat) and a bounded multiprocessing batch stream.

Must stay compatible with the user's runtime (Python 3.8, NumPy 1.18,
Pillow 8.1) as well as current Python / NumPy 2 / Pillow 10+.
"""

import collections
import itertools
import json
import multiprocessing
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- paths
TRAINER_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TRAINER_DIR)
ARTSCANNER_DIR = os.path.join(REPO_ROOT, 'ArtScanner')
for _p in (REPO_ROOT, TRAINER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_FONT = os.path.join(ARTSCANNER_DIR, 'rcc', 'genshin.ttf')
RELIQUARY_LEVEL_JSON = os.path.join(ARTSCANNER_DIR, 'Tools', 'ReliquaryLevelExcelConfigData.json')
RELIQUARY_AFFIX_JSON = os.path.join(ARTSCANNER_DIR, 'Tools', 'ReliquaryAffixExcelConfigData.json')
SHIPPED_MODEL_DIRS = {
    'chs': os.path.join(ARTSCANNER_DIR, 'rcc', 'models_CHS'),
    'en': os.path.join(ARTSCANNER_DIR, 'rcc', 'models_EN'),
}

# Fonts tried (in order, after --fallback-font / $AMENOMA_FALLBACK_FONTS) for
# characters that genshin.ttf does not contain.  Only used for those characters.
SYSTEM_FALLBACK_FONTS = [
    r'C:\Windows\Fonts\msyhbd.ttc',
    r'C:\Windows\Fonts\msyh.ttc',
    r'C:\Windows\Fonts\simhei.ttf',
    '/System/Library/Fonts/PingFang.ttc',
    '/System/Library/Fonts/STHeiti Medium.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
]

# 6.0+: a +0 artifact with 3 initial substats shows its 4th substat greyed out,
# followed by this marker.  All variants are put in the vocab / training data;
# ArtScanner's parser strips them (case-insensitive) and exports the line as
# GOOD ``unactivatedSubstats``.
UNACTIVATED_MARKERS = {
    'chs': ['（待激活）', '(待激活)'],
    'en': [' (unactivated)', ' (Unactivated)'],
}

# How the marker shows up inside the substat crop box: 'full', 'clipped' (cut
# off at a random x by the right edge of the crop; a partly visible glyph stays
# in the image but not in the label) or 'none' (wrapped to the next line,
# outside the crop: grey substat text only).  In game, EN "CRIT DMG+7.8%
# (unactivated)" ends at x~571 and "Energy Recharge+6.5% (unactivated)" at
# x~685 (wider than the 656 px panel, so it wraps), while the crop ends at
# x=560; CHS "元素充能效率+6.5%（待激活）" ends at x~494 and fits.
UNACTIVATED_STYLES = {
    'chs': (('full', 0.9), ('clipped', 0.05), ('none', 0.05)),
    'en': (('full', 0.3), ('clipped', 0.35), ('none', 0.35)),
}

LANG_LABEL = {'chs': 'CHS', 'en': 'EN'}
KIND_LABEL = {'art': 'artifact', 'mat': 'material'}


def charset_filename(lang, kind):
    """``charset_chs_art.json`` etc. (names expected by ArtScanner's OCR loaders)."""
    return 'charset_{}_{}.json'.format(lang, kind)


# --------------------------------------------------------------------------- charset
def build_charset(strings):
    """Sorted set of all characters (same construction as the historical scripts)."""
    return sorted(set(''.join(strings)))


def charset_payload(charset, width, height, max_length, lang, kind, **meta):
    charset = list(charset)
    payload = collections.OrderedDict()
    payload['charset'] = charset
    payload['num_classes'] = len(charset) + 2
    payload['mask_index'] = 0
    payload['blank_index'] = len(charset) + 1
    payload['width'] = int(width)
    payload['height'] = int(height)
    payload['max_length'] = int(max_length)
    payload['lang'] = LANG_LABEL.get(lang, lang)
    payload['kind'] = KIND_LABEL.get(kind, kind)
    for key, value in meta.items():
        if value is not None:
            payload[key] = value
    return payload


def save_charset_json(path, charset, width, height, max_length, lang, kind, **meta):
    folder = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder):
        os.makedirs(folder)
    payload = charset_payload(charset, width, height, max_length, lang, kind, **meta)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
        f.write('\n')
    return path


def load_charset_json(path):
    """Load a charset JSON (dict format above; a bare list is also accepted)."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        data = {'charset': data}
    charset = None
    for key in ('charset', 'characters', 'vocab', 'vocabulary'):
        if key in data:
            charset = list(data[key])
            break
    if charset is None:
        raise ValueError('{}: no "charset" list found'.format(path))
    if charset and charset[0] == '':
        charset = charset[1:]  # tolerate a StringLookup vocabulary that includes the mask token
    if len(set(charset)) != len(charset):
        raise ValueError('{}: duplicated characters in charset'.format(path))
    data['charset'] = charset
    num_classes = data.get('num_classes')
    if num_classes is not None and int(num_classes) != len(charset) + 2:
        raise ValueError('{}: num_classes={} but len(charset)+2={}'.format(path, num_classes, len(charset) + 2))
    return data


class LabelCodec(object):
    """Text <-> class-index conversion with the historical index layout."""

    def __init__(self, charset, max_length):
        self.charset = list(charset)
        self.index = {c: i + 1 for i, c in enumerate(self.charset)}
        self.max_length = int(max_length)
        self.num_classes = len(self.charset) + 2
        self.blank_index = len(self.charset) + 1

    def unknown_chars(self, text):
        return sorted(set(c for c in text if c not in self.index))

    def encode(self, texts):
        out = np.zeros((len(texts), self.max_length), dtype=np.int64)
        for row, text in enumerate(texts):
            if len(text) > self.max_length:
                raise ValueError('label longer than max_length={}: {!r}'.format(self.max_length, text))
            try:
                out[row, :len(text)] = [self.index[c] for c in text]
            except KeyError as e:
                raise ValueError('character {!r} of {!r} is not in the charset'.format(e.args[0], text))
        return out

    def decode_indices(self, pred):
        """Greedy CTC decoding to class indices (lists, mask/out-of-range dropped).

        Same semantics as ``keras.backend.ctc_decode(greedy=True)[:, :max_length]``
        followed by the inverted ``StringLookup`` used historically (and as
        ``ArtScanner/charset_util.Charset.greedy_decode``): merge repeats, drop
        the blank (last class), truncate to ``max_length``, then drop index 0.
        """
        best = np.asarray(pred).argmax(axis=-1)
        n, blank = len(self.charset), self.blank_index
        out = []
        for row in best:
            labels = []
            prev = None
            for k in row:
                k = int(k)
                if k != prev and k != blank:
                    labels.append(k)
                prev = k
            out.append([k for k in labels[:self.max_length] if 1 <= k <= n])
        return out

    def decode(self, pred):
        """Greedy CTC decoding of softmax outputs ``[batch, time, num_classes]``."""
        return [''.join(self.charset[k - 1] for k in row) for row in self.decode_indices(pred)]

    def count_correct(self, y_true, y_pred):
        """Number of rows whose greedy decoding equals the (0-padded) label row."""
        hits = 0
        for truth, pred in zip(np.asarray(y_true), self.decode_indices(y_pred)):
            truth = [int(k) for k in truth if k > 0]
            hits += int(truth == pred)
        return hits


def ctc_min_time_steps(text):
    """Minimum CTC input length for a label (a blank is needed between repeats)."""
    return len(text) + sum(1 for a, b in zip(text, text[1:]) if a == b)


# --------------------------------------------------------------------------- fonts
class MissingGlyphError(ValueError):
    pass


def _font_codepoints(path):
    """Set of code points in the font's cmap (None if fontTools is unavailable)."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return None
    try:
        tt = TTFont(path, fontNumber=0, lazy=True)
        cps = set((tt.getBestCmap() or {}).keys())
        tt.close()
        return cps
    except Exception:  # pragma: no cover - exotic font files
        return None


def _text_length(font, text):
    if hasattr(font, 'getlength'):
        return font.getlength(text)
    return font.getsize(text)[0]  # Pillow < 8


def _text_bbox(font, text):
    if hasattr(font, 'getbbox'):
        return font.getbbox(text)
    w, h = font.getsize(text)  # Pillow < 8
    return 0, 0, w, h


class FontBank(object):
    """genshin.ttf plus fallback fonts for characters it does not cover."""

    def __init__(self, primary=None, fallbacks=None, use_system_fallbacks=True):
        primary = primary or DEFAULT_FONT
        if not os.path.exists(primary):
            raise IOError('font not found: {}'.format(primary))
        candidates = [primary] + list(fallbacks or [])
        env = os.environ.get('AMENOMA_FALLBACK_FONTS', '')
        candidates += [p for p in env.split(os.pathsep) if p]
        if use_system_fallbacks:
            candidates += SYSTEM_FALLBACK_FONTS
        self.paths = []
        for p in candidates:
            if p and os.path.exists(p) and p not in self.paths:
                self.paths.append(p)
        self._codepoints = {}
        self._fonts = {}
        self._choice = {}
        self._notdef = {}

    @property
    def primary(self):
        return self.paths[0]

    def font(self, size, idx=0):
        key = (idx, int(size))
        f = self._fonts.get(key)
        if f is None:
            f = ImageFont.truetype(self.paths[idx], int(size))
            self._fonts[key] = f
        return f

    def _render_probe(self, idx, ch):
        im = Image.new('L', (96, 96), 0)
        ImageDraw.Draw(im).text((16, 16), ch, 255, font=self.font(48, idx))
        return np.asarray(im)

    def has_glyph(self, idx, ch):
        if idx not in self._codepoints:
            self._codepoints[idx] = _font_codepoints(self.paths[idx])
        cps = self._codepoints[idx]
        if cps is not None:
            return ord(ch) in cps
        # Without fontTools: blank glyph or identical to .notdef -> missing.
        probe = self._render_probe(idx, ch)
        if probe.max() == 0:
            return False
        if idx not in self._notdef:
            self._notdef[idx] = self._render_probe(idx, u'\U0010FFFD')
        return not np.array_equal(probe, self._notdef[idx])

    def font_index_for(self, ch):
        if ch in self._choice:
            return self._choice[ch]
        if ch.isspace():
            idx = 0
        else:
            idx = None
            for i in range(len(self.paths)):
                if self.has_glyph(i, ch):
                    idx = i
                    break
        self._choice[ch] = idx
        return idx

    def coverage(self, chars):
        """Return ``(missing_in_primary, {char: fallback_path}, uncovered)``."""
        chars = sorted(set(c for c in chars if not c.isspace()))
        primary_missing = [c for c in chars if not self.has_glyph(0, c)]
        fallback, uncovered = {}, []
        for c in primary_missing:
            idx = self.font_index_for(c)
            if idx is None:
                uncovered.append(c)
            else:
                fallback[c] = self.paths[idx]
        return primary_missing, fallback, uncovered

    def runs(self, text):
        runs = []
        for ch in text:
            idx = self.font_index_for(ch)
            if idx is None:
                raise MissingGlyphError('no font has a glyph for {!r} (U+{:04X}) in {!r}'.format(ch, ord(ch), text))
            if runs and runs[-1][1] == idx:
                runs[-1][0] += ch
            else:
                runs.append([ch, idx])
        return [(s, i) for s, i in runs]

    def text_advance(self, text, size):
        """Horizontal ink + advance width of ``text`` when drawn at ``size``."""
        runs = self.runs(text)
        return sum(_text_length(self.font(size, i), s) for s, i in runs)

    def text_extent(self, text, size):
        """(right, bottom) of the ink box when drawn at the origin (top-left anchor)."""
        runs = self.runs(text)
        if len(runs) == 1 and runs[0][1] == 0:
            box = _text_bbox(self.font(size), text)
            return int(np.ceil(box[2])), int(np.ceil(box[3]))
        ascent, descent = self.font(size).getmetrics()
        width = sum(_text_length(self.font(size, i), s) for s, i in runs)
        return int(np.ceil(width)) + 2, int(ascent + descent) + 2

    def draw_text(self, draw, xy, text, size, fill):
        runs = self.runs(text)
        if len(runs) == 1 and runs[0][1] == 0:
            # Fast path, pixel-identical to the historical generators.
            draw.text(xy, text, fill, font=self.font(size))
            return
        ascent = self.font(size).getmetrics()[0]
        x, baseline = float(xy[0]), xy[1] + ascent
        for s, idx in runs:
            f = self.font(size, idx)
            draw.text((x, baseline), s, fill, font=f, anchor='ls')
            x += _text_length(f, s)


# --------------------------------------------------------------------------- rendering
def split_two_lines(text, rng, limit_range=(26, 35)):
    """Wrap a long name onto two lines at a word boundary (greedy, random width).

    The historical generator always moved only the last word to line 2; the game
    wraps greedily by width, so a 48-char name becomes e.g.
    "Moment That Ceased Upon Waking" / "From Grand Dreams".
    """
    words = text.split(' ')
    if len(words) < 2:
        return [text]
    limit = rng.randint(limit_range[0], limit_range[1] + 1)
    line1 = words[0]
    i = 1
    while i < len(words) - 1 and len(line1) + 1 + len(words[i]) <= limit:
        line1 += ' ' + words[i]
        i += 1
    return [line1 + ' ', ' '.join(words[i:])]


def two_line_step(fonts, size, rng=None):
    """Baseline-to-baseline distance for wrapped names.

    The historical ``int(1.2 * size)`` is tighter than HYWenHei's natural line
    height (ascent + descent ~ 1.25-1.3 * size) and leaves 0-2 px between the
    descenders of line 1 and the ascenders of line 2, so ``resplice`` cannot
    find the gap.  A UI text box uses at least the natural line height.
    """
    ascent, descent = fonts.font(size).getmetrics()
    return ascent + descent + (int(rng.randint(0, 4)) if rng is not None else 1)


def render_text_image(fonts, text, rng, size_range=(15, 40), min_canvas=(550, 55), two_lines=False,
                      fore_range=(0, 75), back_range=(150, 255)):
    """Draw ``text`` like the historical generators (random colours / offset / size).

    Differences: the canvas grows to fit the text (the old fixed 550x55 / 770x60
    canvases clipped long lines such as substats with the 6.0 marker at large
    font sizes, producing images that no longer matched their labels), and
    characters missing from genshin.ttf are drawn with a fallback font.
    """
    pos = (rng.randint(0, 10), rng.randint(0, 10))
    back = tuple(int(rng.randint(back_range[0], back_range[1])) for _ in range(3))
    fore = tuple(int(rng.randint(fore_range[0], fore_range[1])) for _ in range(3))
    if two_lines:
        lines = split_two_lines(text, rng)
        size = int(rng.randint(30, 50) / 2)  # historical two-line size range
    else:
        lines = [text]
        size = int(rng.randint(size_range[0], size_range[1]))
    line_step = two_line_step(fonts, size, rng)
    extents = [fonts.text_extent(line, size) for line in lines]
    width = pos[0] + max(e[0] for e in extents) + 12
    height = pos[1] + line_step * (len(lines) - 1) + extents[-1][1] + 12
    img = Image.new('RGB', (max(min_canvas[0], width), max(min_canvas[1], height)), back)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        fonts.draw_text(draw, (pos[0], pos[1] + i * line_step), line, size, fore)
    return img


def render_clipped_text(fonts, text, clip_from, rng, size_range=(15, 40), min_canvas=(550, 55),
                        fore_range=(0, 75), back_range=(150, 255)):
    """One line of ``text`` whose tail (from character ``clip_from`` on) is cut
    off at a random x, like a substat whose 6.0 marker runs past the right edge
    of the crop box.  Returns ``(img, label)``: the label keeps the characters
    whose advance ends before the cut (trailing spaces removed); a partly
    visible glyph stays in the image but not in the label."""
    pos = (rng.randint(0, 10), rng.randint(0, 10))
    back = tuple(int(rng.randint(back_range[0], back_range[1])) for _ in range(3))
    fore = tuple(int(rng.randint(fore_range[0], fore_range[1])) for _ in range(3))
    size = int(rng.randint(size_range[0], size_range[1]))
    edges = [pos[0] + fonts.text_advance(text[:i], size) for i in range(len(text) + 1)]
    lo, hi = edges[clip_from], edges[-1]
    cut = lo + rng.rand() * max(hi - lo, 1.0)
    keep = max([i for i in range(clip_from, len(text) + 1) if edges[i] <= cut] or [clip_from])
    right, bottom = fonts.text_extent(text, size)
    img = Image.new('RGB', (max(min_canvas[0], pos[0] + right + 12), max(min_canvas[1], pos[1] + bottom + 12)), back)
    fonts.draw_text(ImageDraw.Draw(img), pos, text, size, fore)
    return img.crop((0, 0, max(int(np.ceil(cut)), 1), img.size[1])), text[:keep].rstrip()


# --------------------------------------------------------------------------- preprocessing
# Exact copies of the inference-side preprocessing in ArtScanner/ocr*.py
# (ocr.py / ocr_m.py = "chs", ocr_EN.py / ocr_m_EN.py = "en").
_BILINEAR = getattr(Image, 'BILINEAR', None) or Image.Resampling.BILINEAR


def to_gray(text_img):
    text_img = np.array(text_img)
    if len(text_img.shape) > 2:
        text_img = (text_img[..., :3] @ [[[0.299], [0.587], [0.114]]])[:, :, 0]
    return np.array(text_img, np.float32)


def normalize(img, auto_inverse=True):
    img -= img.min()
    img /= img.max()
    if auto_inverse and img[-1, -1] > 0.5:
        img = 1 - img
    return img


def binarization(img, thresh=0.5):
    return np.where((img < thresh), 0, img)


def crop(img, tol=0.7):
    # img is 2D image data
    # tol  is tolerance
    mask = img > tol
    m, n = img.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def resize_to_height(img, height):
    return (
            np.array(
                Image.fromarray(np.uint8(img * 255)).resize(
                    (int(img.shape[1] * height / img.shape[0]), height),
                    _BILINEAR, )
            ) / 255)


def pad_to_width(img, width):
    if img.shape[1] >= width:
        return img[:, :width]
    return np.pad(
        img, [[0, 0], [0, width - img.shape[1]]], mode="constant", constant_values=0
    )


def zoom(img, height_=60):
    return (
            np.array(
                Image.fromarray(np.uint8(img * 255)).resize(
                    (int(img.shape[1] * height_ / img.shape[0]), height_),
                    _BILINEAR, )
            ) / 255)


def resplice(img):
    """EN only: put a two-line (wrapped) name side by side on one line.

    The historical version only tested row 30 of the 60-px zoomed crop, which
    misses wraps whose first line has descenders (e.g. "...Upon Waking" /
    "From Grand Dreams": the gap sits at rows 32-35) - both lines were then
    squashed into a single unreadable 16-px line.  The first branch is
    unchanged (so 3.4 models see identical inputs); the second one splits at
    the empty row closest to the middle within rows 20..40.  ArtScanner's
    ocr_EN.py / ocr_m_EN.py must use the same function.
    """
    line_height = 27
    if (img[30] == 0.).all():
        res1 = img[:line_height, :]
        res2 = img[-line_height:, :]
        space = np.zeros((res1.shape[0], 10))
        res = np.concatenate((res1, space, res2), axis=1)
        return crop(res, tol=0)
    empty = np.flatnonzero((img[20:41] == 0.).all(axis=1))
    if len(empty) == 0:
        return img
    gap = 20 + int(empty[np.argmin(np.abs(empty - 10))])
    top = crop(img[:gap, :], tol=0)
    bottom = crop(img[gap:, :], tol=0)
    h = max(top.shape[0], bottom.shape[0])
    top = np.pad(top, [[0, h - top.shape[0]], [0, 0]], mode='constant')
    bottom = np.pad(bottom, [[0, h - bottom.shape[0]], [0, 0]], mode='constant')
    return crop(np.concatenate((top, np.zeros((h, 10)), bottom), axis=1), tol=0)


def preprocess_unpadded(text_img, style, height=16):
    """Preprocessing up to (excluding) ``pad_to_width``; ``style`` is "chs" or "en"."""
    result = to_gray(text_img)
    result = normalize(result, True)
    result = binarization(result)
    if style == 'en':
        result = crop(result, tol=0)
        result = zoom(result)
        result = resplice(result)
    else:
        result = crop(result)
        result = normalize(result, False)
    return resize_to_height(result, height)


def preprocess(text_img, style, width, height=16):
    return pad_to_width(preprocess_unpadded(text_img, style, height), width)


# --------------------------------------------------------------------------- data tables
def _load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_artifact_tables(lang):
    """Strings and stat-value tables used by the artifact sample generator."""
    import ArtScanner.ArtsInfo as Info

    if lang == 'chs':
        names = sum(Info.ArtNames, [])
        types = list(Info.TypeNames)
        main_names = dict(Info.MainAttrNames)
        sub_names = dict(Info.SubAttrNames)
        users = list(Info.UsersCHS.keys())
    else:
        names = sum(Info.ArtNames_EN, [])
        types = list(Info.TypeNames_EN)
        main_names = dict(Info.MainAttrNames_EN)
        sub_names = dict(Info.SubAttrNames_EN)
        users = list(Info.UsersEN.keys())

    main_db = _load_json(RELIQUARY_LEVEL_JSON)
    sub_db = _load_json(RELIQUARY_AFFIX_JSON)
    # Pre-computed once (the historical scripts re-scanned the databases per sample).
    main_values = {}
    for prop in main_names:
        main_values[prop] = [j['Value'] for i in main_db for j in i['AddProps'] if j['PropType'] == prop]
    sub_values = {}
    for prop in sub_names:
        sub_values[prop] = [[i['PropValue'] for i in sub_db if i['DepotId'] == depot and i['PropType'] == prop]
                            for depot in [101, 201, 301, 401, 501]]
    return {
        'names': [str(n) for n in names],
        'types': types,
        'main_names': main_names,
        'sub_names': sub_names,
        'users': [str(u) for u in users],
        'formats': dict(Info.Formats),
        'main_values': main_values,
        'sub_values': sub_values,
    }


def load_material_names(lang):
    from ArtScanner.MaterialInfo import MaterialsNameCHS, MaterialsNameEN
    return [str(n) for n in (MaterialsNameCHS if lang == 'chs' else MaterialsNameEN).keys()]


def artifact_charset_strings(lang, tables=None):
    tables = tables or load_artifact_tables(lang)
    strings = (tables['names'] + tables['types'] + list(tables['main_names'].values())
               + list(tables['sub_names'].values()) + list(".,+%0123456789") + tables['users'])
    strings += [u'已装备'] if lang == 'chs' else ['Equipped: ']
    strings += UNACTIVATED_MARKERS[lang]
    return strings


def material_charset_strings(lang, names=None):
    return list(names or load_material_names(lang)) + list("0123456789")


# --------------------------------------------------------------------------- samplers
class ArtifactSampler(object):
    """One sample = one artifact detail panel (6 fixed fields + 1..4 substats).

    Mirrors the historical generator in train_model_artifact(_EN).py; new:
    with probability ``p_unactivated`` the last substat line carries the 6.0
    unactivated marker and is drawn in grey.
    """

    def __init__(self, lang, seed=None, font=None, fallback_fonts=None, p_unactivated=0.35):
        self.lang = lang
        self.style = lang
        self.rng = np.random.RandomState(seed)
        self.tables = load_artifact_tables(lang)
        self.fonts = FontBank(font, fallback_fonts)
        self.p_unactivated = float(p_unactivated)
        self.min_canvas = (550, 55) if lang == 'chs' else (770, 60)

    def _choice(self, seq):
        return seq[self.rng.randint(len(seq))]

    def gen_main_attr_value(self):
        t = self.tables
        prop = self._choice(list(t['main_names'].keys()))
        return t['formats'][prop].format(self._choice(t['main_values'][prop]))

    def gen_single_sub_attr(self):
        t = self.tables
        prop = self._choice(list(t['sub_names'].keys()))
        rare = self.rng.choice(5, p=[0.0625, 0.0625, 0.125, 0.25, 0.5])
        n_upgrades = self.rng.randint(1, rare + 3)
        value = self.rng.choice(t['sub_values'][prop][rare], size=n_upgrades).sum()
        return t['sub_names'][prop] + '+' + t['formats'][prop].format(value)

    def gen_equipped(self):
        user = self._choice(self.tables['users'])
        return user + u'已装备' if self.lang == 'chs' else 'Equipped: ' + user

    def _unactivated_style(self):
        styles, probs = zip(*UNACTIVATED_STYLES[self.lang])
        return self.rng.choice(styles, p=probs)

    def texts(self):
        t = self.tables
        fields = collections.OrderedDict()
        fields['name'] = self._choice(t['names'])
        fields['type'] = self._choice(t['types'])
        fields['main_attr_name'] = self._choice(list(t['main_names'].values()))
        fields['main_attr_value'] = self.gen_main_attr_value()
        fields['level'] = '+' + str(self.rng.randint(0, 21))
        fields['equipped'] = self.gen_equipped()
        n_sub = self.rng.randint(1, 5)
        for i in range(n_sub):
            fields['subattr_{}'.format(i + 1)] = self.gen_single_sub_attr()
        unactivated = None
        if self.rng.rand() < self.p_unactivated:
            key = 'subattr_{}'.format(n_sub)
            style = self._unactivated_style()
            marker = self._choice(UNACTIVATED_MARKERS[self.lang])
            unactivated = (key, style, marker)
        return fields, unactivated

    def sample(self):
        """``(images, labels, keys)`` for one panel; ``keys`` are the field names
        (``'subattr_N+unactivated'`` for the grey 6.0 line)."""
        fields, unactivated = self.texts()
        images, labels, keys = [], [], []
        unact_key = unactivated[0] if unactivated else None
        for key in sorted(fields.keys()):
            text = fields[key]
            if key == unact_key:
                _, style, marker = unactivated
                if style == 'full':
                    full_text = text + marker
                    img = render_text_image(self.fonts, full_text, self.rng, min_canvas=self.min_canvas,
                                            fore_range=(90, 141), back_range=(190, 256))
                    lbl = full_text
                elif style == 'clipped':
                    full_text = text + marker
                    img, lbl = render_clipped_text(self.fonts, full_text, len(text), self.rng,
                                                  min_canvas=self.min_canvas,
                                                  fore_range=(90, 141), back_range=(190, 256))
                else:  # 'none'
                    img = render_text_image(self.fonts, text, self.rng, min_canvas=self.min_canvas,
                                            fore_range=(90, 141), back_range=(190, 256))
                    lbl = text
                images.append(img)
                labels.append(lbl)
                keys.append(key + '+unactivated')
            else:
                two_lines = self.lang == 'en' and key == 'name' and (
                        len(text) >= 35 or (len(text) >= 30 and self.rng.rand() < 0.3))
                images.append(render_text_image(self.fonts, text, self.rng, min_canvas=self.min_canvas,
                                                two_lines=two_lines))
                labels.append(text)
                keys.append(key)
        return images, labels, keys


class MaterialSampler(object):
    """One sample = 8 (name, amount) pairs, as in train_model_materials(_EN).py."""

    def __init__(self, lang, seed=None, font=None, fallback_fonts=None, pairs=8):
        self.lang = lang
        self.style = lang
        self.rng = np.random.RandomState(seed)
        self.names = load_material_names(lang)
        self.fonts = FontBank(font, fallback_fonts)
        self.pairs = int(pairs)

    def texts(self):
        out = []
        for _ in range(self.pairs):
            out.append(self.names[self.rng.randint(len(self.names))])
            out.append(str(self.rng.randint(0, 9999)))
        return out

    def sample(self):
        labels = self.texts()
        images = [render_text_image(self.fonts, text, self.rng, min_canvas=(770, 60),
                                    two_lines=len(text) >= 35)
                  for text in labels]
        keys = ['name' if i % 2 == 0 else 'amount' for i in range(len(labels))]
        return images, labels, keys


SAMPLERS = {'art': ArtifactSampler, 'mat': MaterialSampler}


def make_sampler(cfg, seed):
    cls = SAMPLERS[cfg['kind']]
    kwargs = dict(seed=seed, font=cfg.get('font'), fallback_fonts=cfg.get('fallback_fonts'))
    if cfg['kind'] == 'art':
        kwargs['p_unactivated'] = cfg.get('p_unactivated', 0.35)
    return cls(cfg['lang'], **kwargs)


def make_batch(sampler, codec, width, height, samples_per_step=1, skip_unknown=False):
    """Render ``samples_per_step`` panels into one batch.

    Returns ``(x, y, labels, keys)`` with ``x`` float32 ``[B, width, height, 1]``
    and ``y`` int64 ``[B, max_length]``.  ``skip_unknown`` drops lines that the
    codec cannot encode (used to evaluate older models on data generated from
    newer tables) instead of raising.
    """
    xs, labels, keys = [], [], []
    while not labels:
        for _ in range(max(1, int(samples_per_step))):
            images, texts, fields = sampler.sample()
            for img, text, key in zip(images, texts, fields):
                if skip_unknown and (codec.unknown_chars(text) or len(text) > codec.max_length):
                    continue
                arr = preprocess(img, sampler.style, width, height)
                if not np.isfinite(arr).all() or arr.max() <= 0:
                    continue  # degenerate render; never expected with genshin.ttf
                xs.append(arr.T[None, :, :, None])
                labels.append(text)
                keys.append(key)
    x = np.concatenate(xs, axis=0).astype(np.float32)
    return x, codec.encode(labels), labels, keys


# --------------------------------------------------------------------------- batch stream
_WORKER = {}


def _worker_init(cfg, base_seed):
    ident = multiprocessing.current_process()._identity
    wid = ident[-1] if ident else 0
    _WORKER['cfg'] = cfg
    _WORKER['sampler'] = make_sampler(cfg, (int(base_seed) * 7919 + wid * 104729) % (2 ** 31))
    _WORKER['codec'] = LabelCodec(cfg['charset'], cfg['max_length'])


def _worker_batch(_):
    cfg = _WORKER['cfg']
    x, y, _labels, _keys = make_batch(_WORKER['sampler'], _WORKER['codec'], cfg['width'], cfg['height'],
                                      cfg.get('samples_per_step', 1))
    return x, y


class BatchStream(object):
    """Infinite iterator of ``(x, y)`` training batches for ``model.fit``.

    ``workers > 0`` renders batches in that many ``spawn`` worker processes with
    bounded prefetching (the historical scripts rendered every sample in the
    training thread).  ``cfg`` is a plain dict: lang, kind, charset, max_length,
    width, height, font, fallback_fonts, samples_per_step, p_unactivated.
    Errors raised in a worker (e.g. a label longer than ``max_length``) are
    re-raised here.  Call :meth:`close` (or use ``with``) when done.
    """

    def __init__(self, cfg, workers=0, seed=0, prefetch=None):
        self.cfg = cfg
        self.workers = int(workers)
        self._closed = False
        self._pool = None
        if self.workers <= 0:
            self._sampler = make_sampler(cfg, seed)
            self._codec = LabelCodec(cfg['charset'], cfg['max_length'])
        else:
            ctx = multiprocessing.get_context('spawn')
            self._pool = ctx.Pool(self.workers, initializer=_worker_init, initargs=(cfg, seed))
            self._pending = collections.deque()
            for _ in range(int(prefetch or self.workers * 4)):
                self._pending.append(self._pool.apply_async(_worker_batch, (0,)))

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        if self.workers <= 0:
            x, y, _labels, _keys = make_batch(self._sampler, self._codec, self.cfg['width'], self.cfg['height'],
                                              self.cfg.get('samples_per_step', 1))
            return x, y
        while True:
            try:
                result = self._pending[0].get(timeout=1.0)
                break
            except multiprocessing.TimeoutError:
                if self._closed:
                    raise StopIteration
        try:
            self._pending.popleft()
            self._pending.append(self._pool.apply_async(_worker_batch, (0,)))
        except (AttributeError, ValueError, AssertionError):  # pool closed meanwhile
            raise StopIteration
        return result

    next = __next__

    def close(self):
        self._closed = True
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.terminate()
            pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def validation_set(cfg, n_lines, seed=12345, skip_unknown=False):
    """Fixed synthetic validation lines: ``(x, labels, keys)``."""
    sampler = make_sampler(cfg, seed)
    codec = LabelCodec(cfg['charset'], cfg['max_length'])
    xs, labels, keys = [], [], []
    while len(labels) < n_lines:
        x, _y, texts, fields = make_batch(sampler, codec, cfg['width'], cfg['height'], skip_unknown=skip_unknown)
        xs.append(x)
        labels.extend(texts)
        keys.extend(fields)
    return np.concatenate(xs, axis=0)[:n_lines], labels[:n_lines], keys[:n_lines]


# --------------------------------------------------------------------------- geometry check
def artifact_extreme_labels(lang, tables=None):
    """Every fixed artifact label plus the longest generated value lines.

    Substat values are the 5-star maximum (6 rolls of the highest tier), with
    and without the 6.0 unactivated marker.
    """
    t = tables or load_artifact_tables(lang)
    out = list(t['names']) + list(t['types']) + list(t['main_names'].values()) + ['+0', '+20']
    for prop, values in t['main_values'].items():
        if values:
            out += [t['formats'][prop].format(v) for v in (min(values), max(values))]
    out += [(u + u'已装备') if lang == 'chs' else ('Equipped: ' + u) for u in t['users']]
    for prop, name in t['sub_names'].items():
        tiers = [v for v in t['sub_values'][prop] if v]
        if not tiers:
            continue
        line = name + '+' + t['formats'][prop].format(max(tiers[-1]) * 6)
        out.append(line)
        out += [line + m for m in UNACTIVATED_MARKERS[lang]]
    return out


def material_extreme_labels(lang, names=None):
    return list(names or load_material_names(lang)) + ['0', '9998']


def geometry_report(strings, style, width, height, max_length, fonts, sizes=(18, 30), rng_seed=0,
                    two_line_threshold=None):
    """Check that every label fits ``max_length``, the CTC time axis and ``width``.

    Each distinct string is rendered (dark on light, no random offsets) at the
    given font sizes and run through the real preprocessing; the widest
    un-padded result is compared with ``width`` (anything wider is truncated by
    ``pad_to_width`` at inference time).  Strings at least
    ``two_line_threshold`` long are wrapped like the samplers do (EN names).
    """
    rng = np.random.RandomState(rng_seed)
    time_steps = width // 8
    report = {'time_steps': time_steps, 'too_long': [], 'ctc_overflow': [], 'too_wide': [],
              'max_len': 0, 'max_ctc_steps': 0, 'max_px': 0, 'widest': None, 'longest': None}
    for text in sorted(set(strings)):
        if len(text) > report['max_len']:
            report['max_len'], report['longest'] = len(text), text
        need = ctc_min_time_steps(text)
        report['max_ctc_steps'] = max(report['max_ctc_steps'], need)
        if len(text) > max_length:
            report['too_long'].append(text)
        if need > time_steps:
            report['ctc_overflow'].append(text)
        two = two_line_threshold is not None and len(text) >= two_line_threshold
        for size in sizes:
            if two:
                lines = split_two_lines(text, rng)
                size = max(15, min(24, size))  # two-line font range of the samplers
            else:
                lines = [text]
            step = two_line_step(fonts, size)
            ext = [fonts.text_extent(l, size) for l in lines]
            img = Image.new('RGB', (max(e[0] for e in ext) + 20, step * (len(lines) - 1) + ext[-1][1] + 20),
                            (236, 229, 216))
            draw = ImageDraw.Draw(img)
            for i, line in enumerate(lines):
                fonts.draw_text(draw, (6, 6 + i * step), line, size, (73, 83, 102))
            px = preprocess_unpadded(img, style, height).shape[1]
            if px > report['max_px']:
                report['max_px'], report['widest'] = px, text
            if px > width and text not in report['too_wide']:
                report['too_wide'].append(text)
    return report


def format_geometry_report(report, width, max_length):
    lines = ['  longest label : {} chars (max_length {}) {!r}'.format(report['max_len'], max_length,
                                                                       report['longest']),
             '  CTC time steps: need {} / have {} (width {} / 8)'.format(report['max_ctc_steps'],
                                                                       report['time_steps'], width),
             '  widest render : {} px of {} px {!r}'.format(report['max_px'], width, report['widest'])]
    for key, what in (('too_long', 'longer than max_length'), ('ctc_overflow', 'more CTC steps than time steps'),
                      ('too_wide', 'wider than the input (truncated)')):
        if report[key]:
            lines.append('  !! {} label(s) {}: {}'.format(len(report[key]), what, report[key][:5]))
    return '\n'.join(lines)
