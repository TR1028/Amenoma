rem MODEL_VERSION / DATA_VERSION must match config.py
set MODEL_VERSION=3.4.0
set DATA_VERSION=v71

pyinstaller -w -F ^
--add-data "rcc/models_CHS/artifact_model_%MODEL_VERSION%.h5;./rcc/models_CHS" ^
--add-data "rcc/models_CHS/material_model_%MODEL_VERSION%.h5;./rcc/models_CHS" ^
--add-data "rcc/models_CHS/*.json;./rcc/models_CHS" ^
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
-n Amenoma ^
UImain.py