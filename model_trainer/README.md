# model_trainer: OCR model training (7.x update)

These scripts train the four CRNN models that ArtScanner uses to read
artifact and material text:

| script | runtime model (`ArtScanner/rcc/...`) | input W×H | max_length | charset 7.1 | v3.4 model |
|---|---|---|---|---|---|
| `train_model_artifact.py` | `models_CHS/artifact_model_<ver>.h5` | 320×16 | 24 | 741 | 240×16, 15, 504 |
| `train_model_artifact_EN.py` | `models_EN/artifact_model_EN_<ver>.h5` | 512×16 | 56 | 72 | 384×16, 40, 70 |
| `train_model_materials.py` | `models_CHS/material_model_<ver>.h5` | 320×16 | 20 | 1123 | 240×16, 15, 705 |
| `train_model_materials_EN.py` | `models_EN/material_model_EN_<ver>.h5` | 512×16 | 56 | 70 | 384×16, 40, 66 |

The architecture has not changed: MobileNetV3-Small, then Dense(96), then two
BiLSTMs, then a softmax over charset + 2 classes, trained with CTC. No weight
depends on the input width. This means a v3.4 model can be fine-tuned at the
new widths (the "warm start" described below).

## What changed for 7.x

- **Shared code.** The four scripts now hold only a `SPEC` dict, plus the
  historical screenshot test helpers for artifacts. The rest lives in two
  modules:
  - `trainer_common.py` (no TensorFlow): charset JSON, label codec and greedy
    CTC decoder, fonts, synthetic rendering, the exact ArtScanner
    preprocessing, samplers, the multiprocess batch stream and the geometry
    check.
  - `trainer_tf.py`: the model, CTC loss and accuracy, schedules, warm start,
    evaluation and export.

  `converters/` reuses the same code. It no longer keeps its own copy of the
  architecture and vocabulary.
- **Compatibility.**
  - Deprecated NumPy aliases are gone (`np.float` is now `float`).
  - The code runs on the ArtScanner runtime (Python 3.8, TF 2.3, NumPy 1.18,
    Pillow 8.1) and on current TF with `tf_keras`.
  - Training runs in graph mode. The old NumPy metric forced
    `run_eagerly=True`; it now runs in `tf.py_function`.
  - Samples are rendered in `--workers` processes.
- **Vocabulary.**
  - Every run writes `charset_<lang>_<art|mat>.json` in the format of
    `ArtScanner/charset_util.py`: `charset`, `num_classes` = N+2,
    `mask_index` 0, `blank_index` N+1, `width`, `height`, `max_length`, plus
    metadata. It goes next to the checkpoints in `train/` and next to every
    exported `.h5`, which also gets its own `<model>.charset.json`.
  - ArtScanner no longer rebuilds the vocabulary from the data tables. As a
    result, adding sets, characters or materials can no longer silently break
    a shipped model.
  - The texts come from ArtScanner's tables (`ArtsInfo`, `MaterialInfo`; for
    7.1 these are `rcc/*_v71.json`). Stat values come from
    `ArtScanner/Tools/Reliquary*ExcelConfigData.json`.
- **6.0 unactivated substats.**
  - The artifact vocabularies contain the markers `（待激活）`, `(待激活)`,
    ` (unactivated)` and ` (Unactivated)`.
  - With probability `--p-unactivated` (default 0.35), the last substat of a
    synthetic panel gets one of these markers and is drawn grey, like in game.
    The model reads the whole line, for example `暴击率+3.9%（待激活）`.
  - ArtScanner strips the marker when it parses the line.
- **Geometry.**
  - Input widths and `max_length` were raised (see the table above) so that the
    longest 7.1 labels fit. The binding cases are:
    - `元素充能效率+38.9%（待激活）`: 17 chars, 240 px.
    - `Moment That Ceased Upon Waking From Grand Dreams`: 48 chars, 428–453 px.
    - `Super-Duper Invincible Shining Sparkly Magic Crystal`: 52 chars,
      465–488 px. The pixel range depends on the Pillow version.
    - `切片辖域管控模块·零质点`: 12 chars, 214 px.
  - Every run checks this first and aborts if a label is too long or too wide
    (`--force` overrides). `--charset-only` prints the report without
    training.
- **Fonts.**
  - `ArtScanner/rcc/genshin.ttf` (HYWenHei 85W) has a glyph for every
    character of the 7.1 vocabularies and data dumps: 0 are missing.
  - Its `.notdef` glyph is empty, so a missing character would render as blank
    space while the label still contains it. The trainers check coverage via
    the font's cmap.
  - Characters genshin.ttf lacks are drawn with a fallback font. The trainer
    tries, in order, `--fallback-font`, `$AMENOMA_FALLBACK_FONTS` (separated by
    `os.pathsep`), then common system CJK fonts (Microsoft YaHei, SimHei,
    PingFang, Noto CJK, WenQuanYi, Droid). Training stops if no font has the
    glyph.
  - `--check-font [JSON...]` reports coverage, including all strings of extra
    JSON files.
- **EN two-line names.**
  - Long synthetic names wrap like in game, with the natural line height.
  - `resplice()` now also finds the line gap when line 1 has descenders. It is
    the same function as in ArtScanner's `ocr_EN.py` / `ocr_m_EN.py`.

## Environments

- **Python 3.8 + TF 2.3** (`requirements.txt`, the ArtScanner `.venv38`).
  Recommended, because `.h5` files written there load directly in ArtScanner.
  The original models were trained in this setup on a GPU (`dev_env.yml`:
  CUDA 10.1, cuDNN 7.6); on CPU, install `tensorflow-cpu==2.3.0`.
- **TF 2.4–2.15**, or **TF ≥ 2.16 with the matching `tf_keras`** package. The
  scripts set `TF_USE_LEGACY_KERAS=1` themselves. Keras 3 is not supported.
  - A full `.h5` saved by TF ≥ 2.4 contains layers that TF 2.3 cannot read.
    Re-export it under TF 2.3 from the portable weights file that every export
    also writes:
    ```
    python converters/model_converter.py --only art --weights output/weights/artifact_model_7.1.0.weights.npz
    ```
    Use `art`, `art_en`, `mat` or `mat_en`. Weights trained with TF 2.21 and
    re-exported under TF 2.3 give identical predictions (max diff 5e-9).

Run the scripts from any directory. Paths are resolved relative to
`model_trainer/`.

## Usage

```
cd model_trainer
python train_model_artifact.py --charset-only     # vocabulary + geometry report (no TF)
python train_model_artifact.py --check-font       # glyph coverage of genshin.ttf
python train_model_artifact.py --warm-start       # fine-tune the shipped v3.4 model (recommended)
python train_model_artifact.py                    # historical full schedule from scratch
python train_model_artifact.py --schedule smoke   # 2 steps + export: pipeline check
python train_model_artifact.py --evaluate output/models_CHS/artifact_model_7.1.0.h5
python train_model_artifact.py --convert train/artifact-weights-improvement-12-0.998.hdf5
python train_model_artifact.py --predict-image panel.png --model output/models_CHS/artifact_model_7.1.0.h5
```

The same options apply to the other three scripts (`--help`).

- **`--warm-start`** starts from the newest `ArtScanner/rcc/models_*/<model>_<ver>.h5`
  and its pinned charset JSON.
  - The output layer is remapped per character. Known characters keep their
    trained column. New characters start with a random column and a low bias.
  - Use `--init-from FILE` (with `--init-charset JSON` if needed) to start from
    another model, checkpoint or `.npz`.
  - The default schedule is then `finetune`: the long phase only, 200 epochs ×
    512 steps, learning rate 5e-4.
- **Default `full` schedule** (from scratch): the historical warm-up phases
  (8192×1, 4096×2, 2048×4 and 1024×8 steps; 1024×4 for CHS materials), then up
  to 512 epochs × 512 steps.
  - Like before: ModelCheckpoint
    `train/<kind>-weights-improvement[-EN]-<epoch>-<acc>.hdf5` (weights only,
    best only) and ReduceLROnPlateau.
  - New: the fixed synthetic validation set (`--val-lines`, default 1024) is
    monitored instead of the training batch accuracy.
  - New: training stops once validation accuracy is ≥ `--stop-at` (0.999) for
    `--stop-patience` (5) epochs; optional `--patience` early stopping;
    TerminateOnNaN.
  - Ctrl+C stops training and still exports the best weights so far.
- **Outputs.** At the end, the best weights are exported to `--out-dir`
  (default `output/`):
  ```
  output/models_CHS/artifact_model_<ver>.h5             # runtime model (include_optimizer=False)
  output/models_CHS/artifact_model_<ver>.charset.json   # its vocabulary (model specific)
  output/models_CHS/charset_chs_art.json                # same content, per-language name
  output/weights/artifact_model_<ver>.weights.npz       # portable weights (+ .charset.json)
  ```
  - `<ver>` defaults to `TARGET_VERSION` from `ArtScanner/config.py` (7.1.0);
    change it with `--version`.
  - The export reloads the `.h5` and checks that its predictions match.
- **`--evaluate`** reports exact-match accuracy on synthetic lines rendered
  from the current tables, broken down per field, with sample errors.
  - Lines with characters outside the model's vocabulary are skipped.
  - For the v3.4 models on 7.1 data, all errors are new 7.x names: name and
    equipped fields score 0.53–0.97, every other field 1.000. Overall:
    CHS artifact 0.991, EN artifact 0.907, CHS material 0.947, EN material
    0.840. They need to be retrained.

## Installing new models in ArtScanner

1. Copy `output/models_<LANG>/<model>_<ver>.h5` **and**
   `<model>_<ver>.charset.json` into `ArtScanner/rcc/models_<LANG>/`.
   ArtScanner finds `<model>.charset.json` first, and it takes the input width
   from the model.
2. Set `MODEL_VERSION = "<ver>"` in `ArtScanner/config.py`. This loads
   `rcc/models_*/<model>_<MODEL_VERSION>.h5` for **all four** models.
   - If only some models were retrained, re-export the others under the new
     version name with their old vocabulary. For example:
     ```
     python train_model_materials_EN.py --convert ../ArtScanner/rcc/models_EN/material_model_EN_3.4.0.h5 --charset ../ArtScanner/rcc/models_EN/charset_en_mat.json
     ```
3. Leave the pinned `rcc/models_*/charset_<lang>_<kind>.json` files alone while
   the v3.4 models are still in `rcc/`, because they belong to them.
   `output/.../charset_<lang>_<kind>.json` can replace them once the old models
   are removed.

## Feasibility and timings

Measured on this machine: 24 vCPU Xeon (AVX2), no GPU, `--workers 4`, steady-state training throughput:

| environment | CHS artifact | EN artifact | CHS material | EN material |
|---|---|---|---|---|
| **Py3.8 / TF 2.3 (CPU)** | 1.9 steps/s (17 lines/s) | 1.3 steps/s (11 lines/s) | 2.1 steps/s (34 lines/s) | 1.6 steps/s (26 lines/s) |
| **Py3.13 / TF 2.21 (CPU)** | 2.3 steps/s (20 lines/s) | 1.4 steps/s (12 lines/s) | 2.4 steps/s (38 lines/s) | 1.8 steps/s (29 lines/s) |
| **TF 2.3 with `--samples-per-step 4`** | 1.3 steps/s (45 lines/s) | 0.9 steps/s (32 lines/s) | 1.5 steps/s (96 lines/s) | 1.2 steps/s (76 lines/s) |

### Training Time Projections (CPU)

- **Warm-start fine-tuning (`--warm-start`, recommended)**:
  Starting from the shipped 3.4 models with remapped output layers gives **76%–87% initial accuracy** from epoch 1.
  - 1 epoch (512 steps): ~4.5–6.5 minutes per model.
  - 20 epochs fine-tuning: **~1.5–2.2 hours per model**.
  - 60 epochs fine-tuning: **~4.5–6.5 hours per model**.
  - Feasibility on CPU: **Fully feasible overnight or in background**.

- **Full training from scratch**:
  Warm-up phases (~32,768 steps) + 60 main epochs (30,720 steps) ≈ 63,500 steps:
  - **~9–13 hours per model on CPU** (~40–50 hours total for all 4 models).
  - With a GPU (as in the original v3.4 setup via `dev_env.yml`), full training takes ~30–60 minutes per model.

## Notes

- `train/` holds checkpoints and the vocabulary of the current run.
  - If a new run changes the vocabulary or geometry, the old
    `charset_*.json` is renamed and the run warns you. Old checkpoints then
    need `--charset`.
  - Prefer a fresh `--train-dir` per run.
- `converters/model_converter.py [--only art,art_en,mat,mat_en] [--weights F]
  [--weights-dir D] [--out-dir D] [--version V]` exports checkpoints as runtime
  models.
  - Weights are chosen in this order: `--weights`, then `model_info.*_MODEL_NAME`
    in `--weights-dir` (default `train/`), then the best checkpoint there.
  - The vocabulary comes from the charset JSON next to the weights.
- `model_info.py`: the historical checkpoint names, and `MODEL_TARGET_VERSION`
  read from `ArtScanner/config.py`.
- Legacy files, kept unchanged:
  - `ocr_model.py`: an early prototype with a different architecture and
    undefined globals. It is not used.
  - `model_converter.ipynb`: the old notebook converter, with a stale
    architecture. Use `converters/model_converter.py` instead.
  - `mobilenetv3.py`: still used, as is.
  - `dev_env.yml`: the original Windows GPU conda environment.
