# Game version the data tables (artifact sets, characters, materials) are kept up to date with.
TARGET_VERSION = "7.1.0"

# Suffix of the generated data files in rcc/ (artifact_sets_/char_map_/material_names_<DATA_VERSION>.json).
# Regenerate them with:  python ArtScanner/Tools/update_game_data.py --version 7.1
DATA_VERSION = "v71"

# Version of the shipped OCR weights (rcc/models_*/*_<MODEL_VERSION>.h5).  Models are versioned
# independently of the data tables: the vocabulary a model was trained with is pinned by its charset
# JSON (rcc/models_*/charset_*.json or <model>.charset.json, see charset_util.py), so updating the
# data tables can never silently change how an existing model is decoded.
MODEL_VERSION = "3.4.0"
