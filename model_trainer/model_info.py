MODEL_TARGET_VERSION = '3.4.0'

MODEL_DIR = 'train/'

ARTIFACTS_MODEL_NAME = 'artifact-weights-improvement-63-1.000.hdf5'
ARTIFACTS_MODEL_NAME_EN = 'artifact-weights-improvement-EN-92-1.000.hdf5'

MATERIALS_MODEL_NAME = 'materials-weights-improvement-59-1.000.hdf5'
MATERIALS_MODEL_NAME_EN = 'materials-weights-improvement-EN-85-1.000.hdf5'

OUTPUT_DIR = './output/'


# ---------------------------------------------------------------------------
# 7.x update: ArtScanner loads rcc/models_*/<model>_<config.MODEL_VERSION>.h5
# and builds its data tables for config.TARGET_VERSION.  Newly trained models
# are named after the data version they were trained on (TARGET_VERSION, read
# from ArtScanner/config.py; the constant above is only the fallback): set
# config.MODEL_VERSION to the same value to use them.  MODEL_DIR / OUTPUT_DIR
# are resolved relative to this directory (not the CWD) by trainer_tf.py and
# converters/model_converter.py; when one of the checkpoint names above does
# not exist there, the best matching checkpoint is used.
def _artscanner_target_version(default=MODEL_TARGET_VERSION):
    import os
    import re
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ArtScanner', 'config.py')
    try:
        with open(path, encoding='utf-8') as f:
            match = re.search(r'''^TARGET_VERSION\s*=\s*['"]([^'"]+)['"]''', f.read(), re.M)
    except (IOError, OSError):
        return default
    return match.group(1) if match else default


MODEL_TARGET_VERSION = _artscanner_target_version()
