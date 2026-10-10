# -*- coding: utf-8 -*-
"""Train the EN artifact OCR model -> rcc/models_EN/artifact_model_EN_<version>.h5

7.x update (see README.md):
* shared data generation / preprocessing / CTC code (trainer_common.py,
  trainer_tf.py): multiprocess data workers, graph-mode training, no
  deprecated NumPy aliases, TF 2.3 and current TF (tf_keras) supported
* the training vocabulary is written to charset_en_art.json next to the
  checkpoints and next to the exported .h5 (+ <model>.charset.json)
* the vocabulary and the synthetic panels include the 6.0 unactivated 4th
  substat ("CRIT Rate+3.9% (unactivated)", drawn in grey)
* input width 384 -> 512 and max_length 40 -> 56: piece names now reach 48
  characters / 428 px ("Moment That Ceased Upon Waking From Grand Dreams");
  512 px hold about 56 characters, the CTC output has 64 time steps
* long names wrap greedily onto two lines like in game, with the natural line
  height, and resplice() also finds the gap when line 1 has descenders
  (ArtScanner's ocr_EN.py / ocr_m_EN.py use the same resplice)

Run from model_trainer/ (or anywhere):
    python train_model_artifact_EN.py --warm-start        # fine-tune the shipped model (recommended)
    python train_model_artifact_EN.py                     # historical full schedule from scratch
    python train_model_artifact_EN.py --schedule smoke    # 2 training steps + export (pipeline check)
    python train_model_artifact_EN.py --charset-only      # vocabulary JSON + geometry report, no TF
    python train_model_artifact_EN.py --help
"""
import numpy as np

import trainer_tf

SPEC = dict(
    lang='en', kind='art',
    width=512, height=16, max_length=56,
    model_name='ocr_model_en',
    checkpoint='artifact-weights-improvement-EN-{epoch:02d}-{ctc_accu:.3f}.hdf5',
    min_lr=1e-7,
    warmup=((8192, 1), (4096, 2), (2048, 4), (1024, 8)),
    main=(512, 512),
    scale_ratio=0.625,
    script='train_model_artifact_EN.py',
)


# -- test functions (--predict-image) --
class Config_EN:
    name_coords = [33, 4, 619, 72]
    type_coords = [32, 89, 350, 134]
    main_attr_name_coords = [35, 200, 350, 240]
    main_attr_value_coords = [35, 240, 350, 300]
    star_coords = [30, 310, 350, 360]
    level_coords = [43, 414, 112, 444]
    subattr_1_coords = [67, 480, 560, 520]
    subattr_2_coords = [67, 532, 560, 572]
    subattr_3_coords = [67, 584, 560, 624]
    subattr_4_coords = [67, 636, 560, 676]
    equipped_coords = [105, 1060, 500, 1100]
    lock_coords = [570, 405, 620, 455]


def extract_art_info_EN(art_img, scale_ratio=0.625):
    """Crop the text fields of an artifact detail panel (historical test helper,
    ``np.float`` -> ``float``).  Grey 6.0 unactivated substat lines are kept (the
    model reads them with their "(unactivated)" marker) when ArtScanner's
    panel_detect is available."""
    info = {}
    for key in ('name', 'type', 'main_attr_name', 'main_attr_value', 'level', 'equipped'):
        info[key] = art_img.crop([i * scale_ratio for i in getattr(Config_EN, key + '_coords')])
    subs = [art_img.crop([i * scale_ratio for i in getattr(Config_EN, 'subattr_{}_coords'.format(n))])  # [73, 83, 102]
            for n in range(1, 5)]
    try:
        from ArtScanner import panel_detect
        present, _unactivated = panel_detect.split_substat_lines(subs, scale_ratio)
    except ImportError:
        present = []
        for n, sub in enumerate(subs, 1):
            if np.all(np.abs(np.array(sub, float) - [[[73, 83, 102]]]).max(axis=-1) > 25):
                break
            present.append(n)
    for n in present:
        info['subattr_{}'.format(n)] = subs[n - 1]
    return info


if __name__ == '__main__':
    raise SystemExit(trainer_tf.main(SPEC, extract_fields=extract_art_info_EN))
