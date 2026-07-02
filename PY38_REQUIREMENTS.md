# Python 3.8 Runtime Requirements

Date: 2026-07-01
Target venv: `.venv38`
Verified interpreter: Python 3.8.10

This document records the minimal pip setup for running Amenoma's Windows UI and material scan/OCR workflow from source. It is derived from `ArtScanner/build_env_ui.yml`, the UI import graph, and the material/OCR investigation notes.

## Why Python 3.8

The historical project environments pin Python 3.8.8 and TensorFlow 2.3.0. The local `.venv38` uses Python 3.8.10, which is close enough for the Python 3.8 runtime line and is safer than the previous Python 3.10 venv for TensorFlow 2.3.0.

## Install Strategy

Install TensorFlow first and let pip resolve its compatible transitive packages, especially `numpy` and `h5py`. The conda env used `numpy=1.19.5`, but pip TensorFlow 2.3.0 may require an older compatible numpy range, so forcing the conda numpy pin in a pip venv can create resolver conflicts.

## Minimum Source UI Dependencies

These are needed for the current source UI import layout:

```text
PyQt5==5.15.9
tensorflow==2.3.0
protobuf==3.20.3
Pillow==8.1.2
mss==6.1.0
pywin32==300
mouse==0.7.1
keyboard==0.13.5
Levenshtein==0.13.0
ZODB==5.5.1
persistent==4.6.4
transaction==2.4.0
```

TensorFlow will install additional transitive dependencies, including compatible `numpy` and `h5py`.

## Full Material Scan/OCR Runtime

The same dependency set is sufficient for the current material scan/OCR path because TensorFlow supplies the model runtime dependencies and the app-level packages cover capture, input control, name correction, and ZODB storage.

## Commands

```powershell
cd D:\Amenoma
.\.venv38\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv38\Scripts\python.exe -m pip install -r requirements.txt
```

## Run UI

```powershell
cd D:\Amenoma\ArtScanner
..\.venv38\Scripts\python.exe UIMain.py
```

English UI:

```powershell
cd D:\Amenoma\ArtScanner
..\.venv38\Scripts\python.exe UIMain_EN.py
```

## Offline Crop Helper

Item mode only needs Pillow:

```powershell
.\.venv38\Scripts\python.exe tests\debug_crop_from_saved_image.py materials\debug_material_scan\example\000001_item.png out\amount.png
```

Screenshot mode additionally imports the scanner coordinate helper path and therefore needs `numpy`, `mouse`, `keyboard`, `pywin32`, `mss`, and `Levenshtein`.


### Install Results

Actual install result on 2026-07-01:

- Upgraded tooling to `pip 25.0.1`, `setuptools 75.3.4`, `wheel 0.45.1`.
- Installed `tensorflow==2.3.0` successfully.
- TensorFlow initially pulled `protobuf 5.29.6`, which caused TensorFlow import failure with `Descriptors cannot be created directly`.
- Fixed TensorFlow import by installing `protobuf==3.20.3`.
- Installed UI/scanner packages successfully.
- `pip check` passed after the protobuf pin.
- `import tensorflow as tf; print(tf.__version__)` printed `2.3.0`.
- Runtime imports passed for `PyQt5`, `PIL`, `mss`, `mouse`, `keyboard`, `Levenshtein`, `ZODB`, `persistent`, `transaction`, and `win32api`.
- `tests/debug_crop_from_saved_image.py --help` passed.
- `import UIMain` from `ArtScanner` passed.
- Loading `rcc/models_CHS/material_model_3.4.0.h5` through `ocr_m.OCR(...)` passed.

TensorFlow printed a CUDA warning about missing `cudart64_101.dll`; this is expected on a CPU-only environment and did not block import or model loading.

