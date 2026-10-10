import os

# Game version the data tables (artifact sets, characters, materials) are kept up to date with.
TARGET_VERSION = "7.1.0"

# Suffix of the generated data files in rcc/ (artifact_sets_/char_map_/material_names_<DATA_VERSION>.json).
# Regenerate them with:  python ArtScanner/Tools/update_game_data.py --version 7.1
DATA_VERSION = "v71"

# Version of the shipped OCR weights (rcc/models_*/*_<MODEL_VERSION>.h5).  Models are versioned
# independently of the data tables: the vocabulary a model was trained with is pinned by its charset
# JSON (rcc/models_*/charset_*.json or <model>.charset.json, see charset_util.py), so updating the
# data tables can never silently change how an existing model is decoded.
MODEL_VERSION = "7.1.0"

# Older shipped weights tried, in order, when the model file for MODEL_VERSION is absent.  The CHS
# models were retrained for 7.1 (model_trainer/, warm-started from 3.4.0); the EN models were not
# retrained yet, so the EN UI keeps loading models_EN/*_EN_3.4.0.h5 until *_EN_7.1.0.h5 is added.
FALLBACK_MODEL_VERSIONS = ("3.4.0",)


def model_path(bundle_dir, lang, kind):
    """OCR weights to load for ``lang`` ('CHS' / 'EN') and ``kind`` ('artifact' / 'material').

    Returns rcc/models_<lang>/<kind>_model_[EN_]<version>.h5 for the first version of
    (MODEL_VERSION,) + FALLBACK_MODEL_VERSIONS whose file exists, else the MODEL_VERSION path
    (so the loader's error names the expected file).
    """
    lang = lang.upper()
    subdir = 'models_CHS' if lang == 'CHS' else 'models_EN'
    stem = kind + ('_model_' if lang == 'CHS' else '_model_EN_')
    paths = [os.path.join(bundle_dir, 'rcc', subdir, stem + version + '.h5')
             for version in (MODEL_VERSION,) + tuple(FALLBACK_MODEL_VERSIONS)]
    for path in paths:
        if os.path.isfile(path):
            return path
    return paths[0]
