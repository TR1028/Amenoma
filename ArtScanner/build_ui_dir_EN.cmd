rem DATA_VERSION must match config.py.  The EN models are not retrained for 7.1 yet: bundle the
rem 3.4.0 weights (config.FALLBACK_MODEL_VERSIONS, picked by config.model_path).
set MODEL_VERSION=3.4.0
set DATA_VERSION=v71

pyinstaller -w -D ^
--add-data "rcc/models_EN/artifact_model_EN_%MODEL_VERSION%.h5;./rcc/models_EN" ^
--add-data "rcc/models_EN/material_model_EN_%MODEL_VERSION%.h5;./rcc/models_EN" ^
--add-data "rcc/models_EN/*.json;./rcc/models_EN" ^
--add-data "Tools/ReliquaryLevelExcelConfigData.json;./Tools" ^
--add-data "Tools/ReliquaryAffixExcelConfigData.json;./Tools" ^
--add-data "rcc/genshin.ttf;./rcc" ^
--add-data "rcc/material_names_%DATA_VERSION%.json;./rcc" ^
--add-data "rcc/char_map_%DATA_VERSION%.json;./rcc" ^
--add-data "rcc/artifact_sets_%DATA_VERSION%.json;./rcc" ^
--hidden-import=h5py ^
--hidden-import=h5py.defs ^
--hidden-import=h5py.utils ^
--hidden-import=h5py.h5ac ^
--hidden-import=h5py._proxy ^
--uac-admin ^
-n Amenoma_EN ^
UImain_EN.py