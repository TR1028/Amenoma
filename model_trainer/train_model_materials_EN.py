# -*- coding: utf-8 -*-
"""Train the EN material OCR model -> rcc/models_EN/material_model_EN_<version>.h5

7.x update (see README.md):
* shared data generation / preprocessing / CTC code (trainer_common.py,
  trainer_tf.py): multiprocess data workers, graph-mode training, no
  deprecated NumPy aliases, TF 2.3 and current TF (tf_keras) supported
* the training vocabulary is written to charset_en_mat.json next to the
  checkpoints and next to the exported .h5 (+ <model>.charset.json)
* input width 384 -> 512 and max_length 40 -> 56: "Scattered Piece of
  Decarabian's Dream" already rendered 374-387 px wide (truncated at 384), and
  the longest 7.1 name has 52 characters / 465 px ("Super-Duper Invincible
  Shining Sparkly Magic Crystal")
* long names wrap greedily onto two lines with the natural line height, and
  resplice() also finds the gap when line 1 has descenders (ArtScanner's
  ocr_m_EN.py uses the same resplice)

Run from model_trainer/ (or anywhere):
    python train_model_materials_EN.py --warm-start        # fine-tune the shipped model (recommended)
    python train_model_materials_EN.py                     # historical full schedule from scratch
    python train_model_materials_EN.py --schedule smoke    # 2 training steps + export (pipeline check)
    python train_model_materials_EN.py --charset-only      # vocabulary JSON + geometry report, no TF
    python train_model_materials_EN.py --help
"""
import trainer_tf

SPEC = dict(
    lang='en', kind='mat',
    width=512, height=16, max_length=56,
    model_name='ocr_model_v1',
    checkpoint='materials-weights-improvement-EN-{epoch:02d}-{ctc_accu:.3f}.hdf5',
    min_lr=1e-7,
    warmup=((8192, 1), (4096, 2), (2048, 4), (1024, 8)),
    main=(512, 512),
    script='train_model_materials_EN.py',
)

if __name__ == '__main__':
    raise SystemExit(trainer_tf.main(SPEC))
