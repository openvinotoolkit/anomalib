# Changelog

## v.0.3.0
### What's Changed
* 🛠 ⚠️ Fix configs to properly use pytorch-lightning==1.6 with GPU by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/234
* 🛠 Fix `get_version` in `setup.py` to avoid hard-coding version. by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/229
* 🐞 Fix image loggers by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/233
* Configurable metrics by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/230
* Make OpenVINO throughput optional in benchmarking by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/239
* 🔨 Minor fix: Ensure docs build runs only on isea-server by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/245
* 🏷  Rename `--model_config_path` to `config` by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/246
* Revert "🏷  Rename `--model_config_path` to `config`" by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/247
* ➕ Add `--model_config_path` deprecation warning to `inference.py` by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/248
* Add console logger by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/241
* Add segmentation mask to inference output by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/242
* 🛠 Fix broken mvtec link, and split url to fit to 120 by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/264
* 🛠  Fix mask filenames in folder dataset by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/249


**Full Changelog**: https://github.com/openvinotoolkit/anomalib/compare/v0.2.6...v0.3.0
## v.0.2.6
### What's Changed
* ✏️ Add `torchtext==0.9.1` to support Kaggle environments. by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/165
* 🛠 Fix `KeyError:'label'` in classification folder dataset by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/175
* 📝 Added MVTec license to the repo by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/177
* load best model from checkpoint by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/195
* Replace `SaveToCSVCallback` with PL `CSVLogger` by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/198
* WIP Refactor test by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/197
* 🔧 Dockerfile enhancements by @LukasBommes in https://github.com/openvinotoolkit/anomalib/pull/172
* 🛠 Fix visualization issue for fully defected images by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/194
* ✨ Add hpo search using `wandb` by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/82
* Separate train and validation transformations by @alexriedel1 in https://github.com/openvinotoolkit/anomalib/pull/168
* 🛠 Fix docs workflow by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/200
* 🔄 CFlow: Switch soft permutation to false by default to speed up training. by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/201
* Return only `image`, `path` and `label` for classification tasks in `Mvtec` and `Btech` datasets. by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/196
* 🗑 Remove `freia` as dependency and include it in `anomalib/models/components` by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/174
* Visualizer show classification and segmentation by @alexriedel1 in https://github.com/openvinotoolkit/anomalib/pull/178
* ↗️ Bump up `pytorch-lightning` version to `1.6.0` or higher by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/193
* 🛠 Refactor DFKDE model by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/207
* 🛠 Minor fixes: Update callbacks to AnomalyModule by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/208
* 🛠 Minor update: Update pre-commit docs by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/206
* ✨ Directory streaming by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/210
* ✏️ Updated documentation for development on Docker by @LukasBommes in https://github.com/openvinotoolkit/anomalib/pull/217
* 🏷 Fix Mac M1 dependency conflicts by @dreaquil in https://github.com/openvinotoolkit/anomalib/pull/158
* 🐞 Set tiling off in pathcore to correctly reproduce the stats. by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/222
* 🐞fix support for non-square images by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/204
* Allow specifying feature layer and pool factor in DFM by @nahuja-intel in https://github.com/openvinotoolkit/anomalib/pull/215
* 📝 Add GANomaly metrics to readme by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/224
* ↗️ Bump the version to 0.2.6 by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/223
* 📝 🛠 Fix inconsistent benchmarking throughput/time by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/221
* assign test split for folder dataset by @alexriedel1 in https://github.com/openvinotoolkit/anomalib/pull/220
* 🛠 Refactor model implementations by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/225

## New Contributors
* @LukasBommes made their first contribution in https://github.com/openvinotoolkit/anomalib/pull/172
* @dreaquil made their first contribution in https://github.com/openvinotoolkit/anomalib/pull/158
* @nahuja-intel made their first contribution in https://github.com/openvinotoolkit/anomalib/pull/215

**Full Changelog**: https://github.com/openvinotoolkit/anomalib/compare/v.0.2.5...v0.2.6
## v.0.2.5
### What's Changed
* Bugfix: fix random val/test split issue by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/48
* Fix Readmes by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/46
* Updated changelog by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/49
* add distinction between image and pixel threshold in postprocessor by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/50
* Fix docstrings by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/22
* Fix networkx requirement by @LeonidBeynenson in https://github.com/openvinotoolkit/anomalib/pull/52
* Add min-max normalization by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/53
* Change hardcoded dataset path to environ variable by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/51
* Added cflow algorithm by @blakshma in https://github.com/openvinotoolkit/anomalib/pull/47
* perform metric computation on cpu by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/64
* Fix Inferencer by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/60
* Updated readme for cflow and change default config to reflect results by @blakshma in https://github.com/openvinotoolkit/anomalib/pull/68
* Fixed issue with model loading by @blakshma in https://github.com/openvinotoolkit/anomalib/pull/69
* Docs/sa/fix readme by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/71
* Updated coreset subsampling method to improve accuracy by @blakshma in https://github.com/openvinotoolkit/anomalib/pull/73
* Revert "Updated coreset subsampling method to improve accuracy" by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/79
* Replace `SupportIndex` with `int` by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/76
* Added reference to official CFLOW repo by @blakshma in https://github.com/openvinotoolkit/anomalib/pull/81
* Fixed issue with k_greedy method by @blakshma in https://github.com/openvinotoolkit/anomalib/pull/80
* Fix Mix Data type issue on inferencer by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/77
* Create CODE_OF_CONDUCT.md by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/86
* ✨ Add GANomaly by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/70
* Reorder auc only when needed by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/91
* Bump up the pytorch lightning to master branch due to vulnurability issues by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/55
* 🚀 CI: Nightly Build by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/66
* Refactor by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/87
* Benchmarking Script by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/17
* 🐞 Fix tensor detach and gpu count issues in benchmarking script by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/100
* Return predicted masks in predict step by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/103
* Add Citation to the Readme by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/106
* Nightly build by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/104
* c_idx cast to LongTensor in random sparse projection by @alexriedel1 in https://github.com/openvinotoolkit/anomalib/pull/113
* Update Nightly by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/126
* Updated logos by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/131
* Add third-party-programs.txt file and update license by @LeonidBeynenson in https://github.com/openvinotoolkit/anomalib/pull/132
* 🔨 Increase inference + openvino support by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/122
* Fix/da/image size bug by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/135
* Fix/da/image size bug by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/140
* optimize compute_anomaly_score by using torch native funcrtions by @alexriedel1 in https://github.com/openvinotoolkit/anomalib/pull/141
* Fix IndexError in adaptive threshold computation by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/146
* Feature/data/btad by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/120
* Update for nncf_task by @AlexanderDokuchaev in https://github.com/openvinotoolkit/anomalib/pull/145
* fix non-adaptive thresholding bug by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/152
* Calculate feature map shape patchcore by @alexriedel1 in https://github.com/openvinotoolkit/anomalib/pull/148
* Add `transform_config` to the main `config.yaml` file. by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/156
* Add Custom Dataset Training Support by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/154
* Added extension as an option when saving the result images. by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/162
* Update `anomalib` version and requirements by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/163

## New Contributors
* @LeonidBeynenson made their first contribution in https://github.com/openvinotoolkit/anomalib/pull/52
* @blakshma made their first contribution in https://github.com/openvinotoolkit/anomalib/pull/47
* @alexriedel1 made their first contribution in https://github.com/openvinotoolkit/anomalib/pull/113
* @AlexanderDokuchaev made their first contribution in https://github.com/openvinotoolkit/anomalib/pull/145

**Full Changelog**: https://github.com/openvinotoolkit/anomalib/compare/v.0.2.4...v.0.2.5
## v.0.2.4
### What's Changed
* Bump up the version to 0.2.4 by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/45
* fix heatmap color scheme by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/44

**Full Changelog**: https://github.com/openvinotoolkit/anomalib/compare/v.0.2.3...v.0.2.4

## v.0.2.3
### What's Changed
* Address docs build failing issue by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/39
* Fix docs pipeline 📄 by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/41
* Feature/dick/anomaly score normalization by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/35
* Shuffle train dataloader by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/42


**Full Changelog**: https://github.com/openvinotoolkit/anomalib/compare/v0.2.2...v.0.2.3

## v0.2.0 Pre-release (2021-12-15)
### What's Changed
* Address compatibility issues with OTE, that are caused by the legacy code. by @samet-akcay in [#24](https://github.com/openvinotoolkit/anomalib/pull/24)
* Initial docs string by @ashwinvaidya17 in [#9](https://github.com/openvinotoolkit/anomalib/pull/9)
* Load model did not work correctly as DFMModel did not inherit by @ashwinvaidya17 in [#5](https://github.com/openvinotoolkit/anomalib/pull/5)
* Refactor/samet/data by @samet-akcay in [#8](https://github.com/openvinotoolkit/anomalib/pull/8)
* Delete make.bat by @samet-akcay in [#11](https://github.com/openvinotoolkit/anomalib/pull/11)
* TorchMetrics by @djdameln in [#7](https://github.com/openvinotoolkit/anomalib/pull/7)
* ONNX node naming by @djdameln in [#13](https://github.com/openvinotoolkit/anomalib/pull/13)
* Add FPS counter to `TimerCallback` by @ashwinvaidya17 in [#12](https://github.com/openvinotoolkit/anomalib/pull/12)


## Contributors
* @ashwinvaidya17
* @djdameln
* @samet-akcay

**Full Changelog**: https://github.com/openvinotoolkit/anomalib/commits/v0.2.0
