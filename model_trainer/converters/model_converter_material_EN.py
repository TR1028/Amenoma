"""Export the EN material weights as rcc/models_EN/material_model_EN_<version>.h5 (+ charset JSON).

7.x update: the model architecture, vocabulary (from the charset JSON written
next to the checkpoints) and geometry come from train_model_materials_EN.SPEC /
trainer_tf instead of being duplicated here; see model_converter.py.
"""
import os
import sys

_TRAINER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TRAINER_DIR not in sys.path:
    sys.path.insert(0, _TRAINER_DIR)

import model_info  # noqa: E402
import trainer_tf  # noqa: E402
from train_model_materials_EN import SPEC  # noqa: E402


def convert(weights=None, out_dir=None, version=None, charset=None, weights_dir=None):
    return trainer_tf.convert_checkpoint(SPEC, model_info.MATERIALS_MODEL_NAME_EN, weights=weights, out_dir=out_dir,
                                         version=version, charset=charset, weights_dir=weights_dir)
