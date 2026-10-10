"""OCR charset (vocabulary) handling, decoupled from the game-data tables.

Background
----------
Up to v3.4 every OCR class rebuilt its vocabulary at start-up from the *current* data tables
(ArtsInfo / char_map / material_names).  A CRNN model however only understands the exact
vocabulary it was trained with: output class ``i`` (1 <= i <= N) means ``charset[i-1]``, class 0 is
the StringLookup mask token and class ``N+1`` (the last one) is the CTC blank.  Adding new artifact
sets / characters / materials to the tables therefore changed N or shifted the index -> character
mapping and silently broke the shipped models.

Every model now has a companion JSON file next to its ``.h5`` file::

    {"charset": ["%", "+", ...], "num_classes": N + 2, "mask_index": 0, "blank_index": N + 1,
     "width": 240, "height": 16, "max_length": 15, ...}

Unknown keys (``model``, ``lang``, ``kind``, ``target_version``, ``font``, ...) are ignored, except
that a per-language file whose ``model`` key names a *different* model file is only used as a last
resort.  Lookup order for ``<dir>/<stem>.h5``:

1. ``<dir>/<stem>.charset.json``             model specific (preferred for newly trained models)
2. ``<dir>/charset_<lang>_<art|mat>.json``   per language / kind (the pinned v3.4.0 vocabularies)
3. the legacy vocabulary rebuilt from the current data tables, accepted only if its size matches the
   model output (a warning is logged).

The candidate's ``num_classes`` must equal the model's output dimension.  The model's input shape,
when known, wins over the ``width`` / ``height`` stored in the JSON.

Pure python + numpy (no TensorFlow import) so that it can be unit tested without TF.  Must stay
compatible with the Python 3.8 / numpy 1.18 runtime.
"""
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

KIND_ALIASES = {"artifact": "art", "art": "art", "material": "mat", "mat": "mat"}

# Geometry of the v3.4.0 models (used when an older JSON file lacks these keys and for the legacy
# fallback).
LEGACY_GEOMETRY = {
    ("CHS", "art"): {"width": 240, "height": 16, "max_length": 15},
    ("CHS", "mat"): {"width": 240, "height": 16, "max_length": 15},
    ("EN", "art"): {"width": 384, "height": 16, "max_length": 40},
    ("EN", "mat"): {"width": 384, "height": 16, "max_length": 40},
}


class CharsetError(RuntimeError):
    """No vocabulary matching the model could be found."""


def _norm(lang, kind):
    lang = str(lang).upper()
    if lang not in ("CHS", "EN"):
        raise ValueError("unknown OCR language %r (expected 'CHS' or 'EN')" % (lang,))
    try:
        kind = KIND_ALIASES[str(kind).lower()]
    except KeyError:
        raise ValueError("unknown OCR model kind %r (expected 'artifact' or 'material')" % (kind,))
    return lang, kind


class Charset(object):
    """An OCR vocabulary plus the input geometry of the model it belongs to."""

    def __init__(self, characters, width, height, max_length, source="", meta=None):
        characters = list(characters)
        if not characters:
            raise CharsetError("empty charset")
        bad = [c for c in characters if not isinstance(c, str) or c == ""]
        if bad:
            raise CharsetError("charset entries must be non-empty strings, got %r" % (bad[:5],))
        if len(set(characters)) != len(characters):
            seen, dup = set(), []
            for c in characters:
                if c in seen:
                    dup.append(c)
                seen.add(c)
            raise CharsetError("duplicate charset entries: %r" % (dup[:10],))
        self.characters = characters
        self.width = int(width)
        self.height = int(height)
        self.max_length = int(max_length)
        self.source = source
        self.meta = dict(meta or {})

    def __len__(self):
        return len(self.characters)

    def __repr__(self):
        return "Charset(%d chars, %dx%d, max_length=%d, source=%r)" % (
            len(self.characters), self.width, self.height, self.max_length, self.source)

    @property
    def num_classes(self):
        """Model output dimension: mask (0) + characters (1..N) + CTC blank (N+1)."""
        return len(self.characters) + 2

    @property
    def mask_index(self):
        return 0

    @property
    def blank_index(self):
        return len(self.characters) + 1

    # ------------------------------------------------------------------ decoding
    def indices_to_text(self, indices):
        """Map class indices to text.  The mask (0), the CTC blank and the -1 padding produced by
        ``ctc_decode`` (and anything else out of range) map to the empty string, exactly like the
        inverted ``StringLookup(oov_token="", mask_token="")`` that was used before."""
        n = len(self.characters)
        chars = self.characters
        out = []
        for value in np.asarray(indices).ravel():
            i = int(value)
            if 1 <= i <= n:
                out.append(chars[i - 1])
        return "".join(out)

    def decode_dense(self, dense):
        """Decode a dense (batch, length) index matrix, e.g. ``ctc_decode(...)[0][0]``."""
        return [self.indices_to_text(row) for row in np.asarray(dense)]

    def greedy_decode(self, pred, max_length=None):
        """Greedy CTC decoding of softmax output ``pred`` with shape (batch, time, num_classes).

        Equivalent to ``tf.keras.backend.ctc_decode(pred, input_length=[time]*batch, greedy=True)``
        (merge repeated labels, then drop blanks; the blank is the last class) followed by
        ``[:, :max_length]`` and the index -> character lookup.
        """
        pred = np.asarray(pred)
        if pred.ndim != 3:
            raise ValueError("expected a (batch, time, classes) array, got shape %r" % (pred.shape,))
        if pred.shape[-1] != self.num_classes:
            raise CharsetError("prediction has %d classes but the charset expects %d"
                               % (pred.shape[-1], self.num_classes))
        limit = self.max_length if max_length is None else int(max_length)
        blank = self.blank_index
        texts = []
        for row in np.argmax(pred, axis=-1):
            labels = []
            prev = None
            for idx in row:
                idx = int(idx)
                if idx != prev and idx != blank:
                    labels.append(idx)
                prev = idx
            texts.append(self.indices_to_text(labels[:limit]))
        return texts

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self, **extra):
        data = {
            "charset": list(self.characters),
            "num_classes": self.num_classes,
            "mask_index": self.mask_index,
            "blank_index": self.blank_index,
            "width": self.width,
            "height": self.height,
            "max_length": self.max_length,
        }
        for key, value in self.meta.items():
            data.setdefault(key, value)
        data.update(extra)
        return data

    def save(self, path, **extra):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(**extra), f, ensure_ascii=False, indent=1)
            f.write("\n")

    @classmethod
    def from_dict(cls, data, defaults=None, source=""):
        if "charset" not in data:
            raise CharsetError("%s: missing 'charset' key" % (source or "charset json",))
        defaults = defaults or {}
        geometry = {}
        for key in ("width", "height", "max_length"):
            value = data.get(key, defaults.get(key))
            if value is None:
                raise CharsetError("%s: missing %r" % (source or "charset json", key))
            geometry[key] = value
        known = {"charset", "width", "height", "max_length"}
        meta = {k: v for k, v in data.items() if k not in known}
        charset = cls(data["charset"], source=source, meta=meta, **geometry)
        # Consistency checks for the optional derived keys.
        for key, expected in (("num_classes", charset.num_classes),
                              ("blank_index", charset.blank_index),
                              ("mask_index", 0)):
            if key in data and int(data[key]) != expected:
                raise CharsetError("%s: %s=%r but the charset implies %d"
                                   % (source or "charset json", key, data[key], expected))
        return charset

    @classmethod
    def from_json(cls, path, defaults=None):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, defaults=defaults, source=path)


# ---------------------------------------------------------------------------------------------
# Legacy (v3.4) runtime vocabulary generation.  Kept for models that ship without a JSON file.
# ---------------------------------------------------------------------------------------------
def build_dynamic_charset(kind, lang):
    """Rebuild the vocabulary from the *current* data tables, the way the OCR classes did up to
    v3.4.  Only valid for a model trained on exactly these tables."""
    lang, kind = _norm(lang, kind)
    if kind == "art":
        import ArtsInfo
        if lang == "CHS":
            parts = (sum(ArtsInfo.ArtNames, [])
                     + ArtsInfo.TypeNames
                     + list(ArtsInfo.MainAttrNames.values())
                     + list(ArtsInfo.SubAttrNames.values())
                     + list(".,+%0123456789")
                     + list(ArtsInfo.UsersCHS)
                     + list("已装备"))
        else:
            parts = (sum(ArtsInfo.ArtNames_EN, [])
                     + ArtsInfo.TypeNames_EN
                     + list(ArtsInfo.MainAttrNames_EN.values())
                     + list(ArtsInfo.SubAttrNames_EN.values())
                     + list(".,+%0123456789")
                     + list(ArtsInfo.UsersEN)
                     + list("Equipped: "))
    else:
        import MaterialInfo
        names = MaterialInfo.MaterialsNameCHS if lang == "CHS" else MaterialInfo.MaterialsNameEN
        parts = list(names) + list("0123456789")
    return sorted(set("".join(parts)))


def candidate_paths(model_path, lang, kind):
    lang, kind = _norm(lang, kind)
    model_dir, model_file = os.path.split(os.path.abspath(model_path))
    stem = os.path.splitext(model_file)[0]
    return [
        os.path.join(model_dir, stem + ".charset.json"),
        os.path.join(model_dir, "charset_%s_%s.json" % (lang.lower(), kind)),
    ]


def resolve_charset(model_path, lang, kind, num_classes=None, input_shape=None, allow_legacy=True):
    """Find the vocabulary for ``model_path`` (see the module docstring for the lookup order).

    :param num_classes: output dimension of the loaded model (``model.output_shape[-1]``); when
        given, candidates with a different size are rejected.
    :param input_shape: ``model.input_shape`` = (batch, width, height, 1); when given it overrides
        the width / height stored in the JSON.
    """
    lang, kind = _norm(lang, kind)
    defaults = LEGACY_GEOMETRY[(lang, kind)]
    model_file = os.path.basename(model_path)
    rejected = []
    deferred = []
    chosen = None
    paths = candidate_paths(model_path, lang, kind)
    for rank, path in enumerate(paths):
        if not os.path.isfile(path):
            continue
        try:
            cs = Charset.from_json(path, defaults=defaults)
        except (CharsetError, ValueError, OSError) as e:
            rejected.append("%s: %s" % (path, e))
            continue
        if num_classes is not None and cs.num_classes != int(num_classes):
            rejected.append("%s: %d classes, model has %s" % (path, cs.num_classes, num_classes))
            continue
        declared = cs.meta.get("model")
        if rank > 0 and declared and declared != model_file:
            deferred.append(cs)
            continue
        chosen = cs
        break
    if chosen is None and deferred:
        chosen = deferred[0]
        logger.warning("charset %s declares model %r, using it for %s anyway (sizes match)",
                       chosen.source, chosen.meta.get("model"), model_file)
    if chosen is None and allow_legacy:
        cs = Charset(build_dynamic_charset(kind, lang), source="legacy:data-tables", **defaults)
        if num_classes is None or cs.num_classes == int(num_classes):
            logger.warning("no charset json for %s, using the vocabulary rebuilt from the current "
                           "data tables (only correct if the model was trained on them)", model_file)
            chosen = cs
        else:
            rejected.append("legacy vocabulary from the data tables: %d classes, model has %s"
                            % (cs.num_classes, num_classes))
    if chosen is None:
        raise CharsetError(
            "No OCR charset matches model %s (lang=%s, kind=%s). Put a '<model>.charset.json' or "
            "'charset_%s_%s.json' next to the model. Tried: %s"
            % (model_path, lang, kind, lang.lower(), kind, "; ".join(rejected) or "nothing found"))
    if input_shape is not None and len(input_shape) >= 3:
        width, height = input_shape[1], input_shape[2]
        if width is not None and height is not None and (int(width), int(height)) != (chosen.width, chosen.height):
            logger.warning("%s: charset geometry %dx%d differs from the model input %sx%s; using the model's",
                           model_file, chosen.width, chosen.height, width, height)
            chosen.width, chosen.height = int(width), int(height)
    return chosen


def load_charset_for_model(model, model_path, lang, kind):
    """Convenience wrapper for a loaded Keras model."""
    num_classes = None
    input_shape = None
    try:
        num_classes = int(model.output_shape[-1])
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        input_shape = tuple(model.input_shape)
    except (AttributeError, TypeError):
        pass
    return resolve_charset(model_path, lang, kind, num_classes=num_classes, input_shape=input_shape)


# 6.0+: characters a model needs to read the "（待激活）" / "(unactivated)" suffix of the greyed out
# substat.  Without them (e.g. the shipped v3.4.0 models) the CRNN misreads the whole line, sometimes
# into a plausible wrong value, so such lines must not be OCR'd at all.
UNACTIVATED_MARKER_CHARS = {"CHS": "待激活", "EN": "(unactivated)"}


def can_read_unactivated(characters, lang):
    """True if the vocabulary ``characters`` contains the 6.0 unactivated marker of ``lang``."""
    lang = str(lang).upper()
    chars = set(characters)
    return all(c in chars for c in UNACTIVATED_MARKER_CHARS[lang])
