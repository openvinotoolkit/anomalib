# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

### Changed

### Deprecated

### Fixed

## [v0.7.0] - 2023-08-28

### Added

- AUPRO binning capability by @yann-cv in https://github.com/openvinotoolkit/anomalib/pull/1145
- Add support for receiving dataset paths as a list by @harimkang in https://github.com/openvinotoolkit/anomalib/pull/1265
- Add modelAPI compatible OpenVINO export by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/1219

### Changed

- Enable training with only normal images for MVTec by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/1241
- Improve default settings of EfficientAD in https://github.com/openvinotoolkit/anomalib/pull/1143
- Added the tracer_kwargs to the TorchFXFeatureExtractor class by @JoaoGuibs in https://github.com/openvinotoolkit/anomalib/pull/1214
- Replace cdist in Patchcore by @blaz-r in https://github.com/openvinotoolkit/anomalib/pull/1267
- Ignore hidden directories when creating Folder dataset by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/1268

### Fixed

- Fix typechecking error for toch.onnx.export by @ORippler in https://github.com/openvinotoolkit/anomalib/pull/1159
- Fix benchmarking type error by @blaz-r in https://github.com/openvinotoolkit/anomalib/pull/1155
- Fix incorrect shape mismatch between anomaly map and ground truth mask by @alexriedel1 in https://github.com/openvinotoolkit/anomalib/pull/1182
- Fix dataset keys in benchmarking notebook by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/1242
- Remove config from argparse in OpenVINO inference script by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/1257
- Fix EfficientAD number of steps for optimizer lr change by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/1266
- Fix unable to read the mas image by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/1277

## [v0.6.0] - 2023-06-15

### Added

- EfficientAD by @alexriedel1 in https://github.com/openvinotoolkit/anomalib/pull/1073
- AI-VAD bbox post-processing by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/1103
- Add dataset categories to data modules by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/1105
- Pass codedov token from environment by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/1102

### Changed

- OV API2.0 by @paularamo in https://github.com/openvinotoolkit/anomalib/pull/1098

### Deprecated

- OV API1.0 by @paularamo in https://github.com/openvinotoolkit/anomalib/pull/1098

###  Fixed

- Fix Fastflow ONNX export. by @jasonvanzelm in https://github.com/openvinotoolkit/anomalib/pull/1108
- Fix tile import typo by @xbkaishui in https://github.com/openvinotoolkit/anomalib/pull/1106
- Fix `pre-commit` issues caused by the EfficientAD PR by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/1114
- Bump requests from 2.26.0 to 2.31.0 in /requirements by @dependabot in https://github.com/openvinotoolkit/anomalib/pull/1100

### New Contributors

- @xbkaishui made their first contribution in https://github.com/openvinotoolkit/anomalib/pull/1106
- @jasonvanzelm made their first contribution in https://github.com/openvinotoolkit/anomalib/pull/1108

**Full Changelog**: https://github.com/openvinotoolkit/anomalib/compare/v0.5.1...v0.6.0

## [v0.5.1] - 2023-05-24

### Added

- 🧪 Add tests for tools by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/1069
- Add kolektor dataset by @Ravindu987 in https://github.com/openvinotoolkit/anomalib/pull/983

### Changed

- Rename `metadata_path` to `metadata` in `OpenvinoInferencer` in https://github.com/openvinotoolkit/anomalib/pull/1101
- 📝 [Notebooks] - Simplify the dobot notebooks. by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/1084
- Upgrade python to 3.10 by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/1035
- 📝 [Notebooks] - Install anomalib via pip in the Jupyter Notebooks by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/1091
- Update code-scan workflow to use Trivy by @yunchu in https://github.com/openvinotoolkit/anomalib/pull/1097

### Fixed

- Fix `init_state_dict` bug in `wrap_nncf_model` in https://github.com/openvinotoolkit/anomalib/pull/1101
- Fix mypy pep561 by @WenjingKangIntel in https://github.com/openvinotoolkit/anomalib/pull/1088
- 📝 [Notebooks] Fix the broken formatting by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/1090

## [v0.5.0] - 2023-05-09

### Added

- 📚 Add OpenVINO Inference to getting started notebook. by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/897
- Mvtec 3d by @alexriedel1 in https://github.com/openvinotoolkit/anomalib/pull/907
- MVTec 3D and Folder3D by @alexriedel1 in https://github.com/openvinotoolkit/anomalib/pull/942
- add reorder=True argument in aupro by @triet1102 in https://github.com/openvinotoolkit/anomalib/pull/944
- always reorder inputs when computing AUROC by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/945
- always reorder for aupr metric by @triet1102 in https://github.com/openvinotoolkit/anomalib/pull/975
- Add `README.md` files to `notebooks` directories and its subdirectories (<https://github.com/openvinotoolkit/anomalib/issues/993>)
- Set transformations from the config file by @alexriedel1 in https://github.com/openvinotoolkit/anomalib/pull/990
- Add contributors to `README.md` by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/995
- Add codeowners file by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/1019
- Configure reference frame for multi-frame video clips by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/1023
- [Algo] Attribute-based Representations for Accurate and Interpretable Video Anomaly Detection by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/1040

### Changed

- Switch to src layout by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/921
- Remove `config` flag from `OpenVINOInferencer` (<https://github.com/openvinotoolkit/anomalib/pull/939>)
- Add ruff as the main linter by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/936
- Add a new workflow for code scanning by @yunchu in https://github.com/openvinotoolkit/anomalib/pull/940
- Enable bandit scanning by @yunchu in https://github.com/openvinotoolkit/anomalib/pull/954
- 🐳 Update Containers and Readme by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/952
- Refactor AUPRO metric by @triet1102 in https://github.com/openvinotoolkit/anomalib/pull/991
- enable auto-fixing for ruff in pre-commit by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/1004
- Refactor strings and ints into enum.Enum by @WenjingKangIntel in https://github.com/openvinotoolkit/anomalib/pull/1044
- Modify codecov upload by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/1080

### Deprecated

- Remove torchvision and torchtext by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/903
- Remove codacy from ci docs by @ashwinvaidya17 in https://github.com/openvinotoolkit/anomalib/pull/924
- Remove config dependency from `OpenVINOInferencer` by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/939
- Remove config from torch inferencer by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/1001

###  Fixed

- Bugfix code logic to allow for passing of `nn.Module` to `TorchFXFeatureExtractor` by @ORippler in https://github.com/openvinotoolkit/anomalib/pull/935
- fix broken links to tutorials (ex guides) by @sergiev in https://github.com/openvinotoolkit/anomalib/pull/957
- Fixed outdated info in readme by @blaz-r in https://github.com/openvinotoolkit/anomalib/pull/969
- Fix ruff isort integration by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/976
- Fix/samples dataframe annotation by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/981
- Fixed openvino_inferencer in gradio_inference by @blaz-r in https://github.com/openvinotoolkit/anomalib/pull/972
- Fix issue in tutorial by @Ravindu987 in https://github.com/openvinotoolkit/anomalib/pull/997
- Fix tarfile vulnerability by @djdameln in https://github.com/openvinotoolkit/anomalib/pull/1003
- Cuda 11.4 dockerfile fix by @phcarval in https://github.com/openvinotoolkit/anomalib/pull/1021
- Make anomalib PEP 561 compliant for mypy by @WenjingKangIntel in https://github.com/openvinotoolkit/anomalib/pull/1038
- [Bug: 839] Crop in SSPCAB implementation by @isaacncz in https://github.com/openvinotoolkit/anomalib/pull/1057
- [Bug: 865] datamodule.setup() assertion failed by @isaacncz in https://github.com/openvinotoolkit/anomalib/pull/1058
- Fix logger message for test_split_ratio by @ugotsoul in https://github.com/openvinotoolkit/anomalib/pull/1071
- Fix notebook readme formatting by @samet-akcay in https://github.com/openvinotoolkit/anomalib/pull/1075

### New Contributors

- @triet1102 made their first contribution in https://github.com/openvinotoolkit/anomalib/pull/944
- @sergiev made their first contribution in https://github.com/openvinotoolkit/anomalib/pull/957
- @blaz-r made their first contribution in https://github.com/openvinotoolkit/anomalib/pull/969
- @ineiti made their first contribution in https://github.com/openvinotoolkit/anomalib/pull/987
- @Ravindu987 made their first contribution in https://github.com/openvinotoolkit/anomalib/pull/997
- @phcarval made their first contribution in https://github.com/openvinotoolkit/anomalib/pull/1021
- @WenjingKangIntel made their first contribution in https://github.com/openvinotoolkit/anomalib/pull/1038
- @isaacncz made their first contribution in https://github.com/openvinotoolkit/anomalib/pull/1057
- @ugotsoul made their first contribution in https://github.com/openvinotoolkit/anomalib/pull/1071

**Full Changelog**: https://github.com/openvinotoolkit/anomalib/compare/v0.4.0...v0.5.0

## [v0.4.0] - 2023-03-01

### Added

- Add Dobot notebook (<https://github.com/openvinotoolkit/anomalib/pull/928>)
- Add ShanghaiTech Campus video anomaly detection dataset (<https://github.com/openvinotoolkit/anomalib/pull/869>)
- Add `pyupgrade` to `pre-commit` configs, and refactor based on `pyupgrade` and `refurb` (<https://github.com/openvinotoolkit/anomalib/pull/845>)
- Add [CFA](https://arxiv.org/abs/2206.04325) model implementation (<https://github.com/openvinotoolkit/anomalib/pull/783>)
- Add RKDE model implementation (<https://github.com/openvinotoolkit/anomalib/pull/821>)
- Add Visual Anomaly (VisA) dataset adapter (<https://github.com/openvinotoolkit/anomalib/pull/824>)
- Add Synthetic anomalous dataset for validation and testing (https://github.com/openvinotoolkit/anomalib/pull/822)
- Add Detection task type support (https://github.com/openvinotoolkit/anomalib/pull/822)
- Add UCSDped and Avenue dataset implementation (https://github.com/openvinotoolkit/anomalib/pull/822)
- Add base classes for video dataset and video datamodule (https://github.com/openvinotoolkit/anomalib/pull/822)
- Add base classes for image dataset and image dataModule (https://github.com/openvinotoolkit/anomalib/pull/822)
- ✨ Add CSFlow model (<https://github.com/openvinotoolkit/anomalib/pull/657>)
- Log loss for existing trainable models (<https://github.com/openvinotoolkit/anomalib/pull/804>)
- Add section for community project (<https://github.com/openvinotoolkit/anomalib/pull/768>)
- ✨ Add torchfx feature extractor (<https://github.com/openvinotoolkit/anomalib/pull/675>)
- Add tiling notebook (<https://github.com/openvinotoolkit/anomalib/pull/712>)
- Add posargs to tox to enable testing a single file (https://github.com/openvinotoolkit/anomalib/pull/695)
- Add option to load metrics with kwargs (https://github.com/openvinotoolkit/anomalib/pull/688)
- 🐞 Add device flag to TorchInferencer (<https://github.com/openvinotoolkit/anomalib/pull/601>)

### Changed

- Configure reference frame for multi-frame video clips (<https://github.com/openvinotoolkit/anomalib/pull/1023>)
- Bump OpenVINO version to `2022.3.0` (<https://github.com/openvinotoolkit/anomalib/pull/932>)
- Remove the dependecy on a specific `torchvision` and `torchmetrics` packages.
- Bump PyTorch Lightning version to v.1.9.\* (<https://github.com/openvinotoolkit/anomalib/pull/870>)
- Make input image normalization and center cropping configurable from config (https://github.com/openvinotoolkit/anomalib/pull/822)
- Improve flexibility and configurability of subset splitting (https://github.com/openvinotoolkit/anomalib/pull/822)
- Switch to new datamodules design (https://github.com/openvinotoolkit/anomalib/pull/822)
- Make normalization and center cropping configurable through config (<https://github.com/openvinotoolkit/anomalib/pull/795>)
- Switch to new [changelog format](https://keepachangelog.com/en/1.0.0/). (<https://github.com/openvinotoolkit/anomalib/pull/777>)
- Rename feature to task (<https://github.com/openvinotoolkit/anomalib/pull/769>)
- make device configurable in OpenVINO inference (<https://github.com/openvinotoolkit/anomalib/pull/755>)
- 🚨 Fix torchmetrics version (<https://github.com/openvinotoolkit/anomalib/pull/754>)
- Improve NNCF initilization (<https://github.com/openvinotoolkit/anomalib/pull/740>)
- Migrate markdownlint + issue templates (<https://github.com/openvinotoolkit/anomalib/pull/738>)
- 🐞 Patch Timm Feature Extractor (<https://github.com/openvinotoolkit/anomalib/pull/714>)
- Padim arguments improvements (<https://github.com/openvinotoolkit/anomalib/pull/664>)
- 📊 Update DFM results (<https://github.com/openvinotoolkit/anomalib/pull/674>)
- Optimize anomaly score calculation for PatchCore (<https://github.com/openvinotoolkit/anomalib/pull/633>)

### Deprecated

- Deprecated PreProcessor class (<https://github.com/openvinotoolkit/anomalib/pull/795>)
- Deprecate OptimalF1 metric in favor of AnomalyScoreThreshold and F1Score (<https://github.com/openvinotoolkit/anomalib/pull/796>)

### Fixed

- Fix bug in `anomalib/data/utils/image.py` to check if the path is directory (<https://github.com/openvinotoolkit/anomalib/pull/919>)
- Fix bug in MVTec dataset download (<https://github.com/openvinotoolkit/anomalib/pull/842>)
- Add early stopping to CS-Flow model (<https://github.com/openvinotoolkit/anomalib/pull/817>)
- Fix remote container by removing version pinning in Docker files (<https://github.com/openvinotoolkit/anomalib/pull/797>)
- Fix PatchCore performance deterioration by reverting changes to Average Pooling layer (<https://github.com/openvinotoolkit/anomalib/pull/791>)
- Fix zero seed (<https://github.com/openvinotoolkit/anomalib/pull/766>)
- Fix #699 (<https://github.com/openvinotoolkit/anomalib/pull/700>)
- 🐞 Fix folder dataset for classification tasks (<https://github.com/openvinotoolkit/anomalib/pull/708>)
- Update torchmetrics to fix compute_on_cpu issue (<https://github.com/openvinotoolkit/anomalib/pull/711>)
- Correct folder mask path (<https://github.com/openvinotoolkit/anomalib/pull/660>)
- Fix >100% confidence issue for OpenVINO inference (<https://github.com/openvinotoolkit/anomalib/pull/667>)
- Update pre-commit links and some other minor fixes (<https://github.com/openvinotoolkit/anomalib/pull/672>)
- Fix black formatting issues. (<https://github.com/openvinotoolkit/anomalib/pull/674>)

## [v0.3.7] - 2022-10-28

### What's Changed

- Feature/comet logging by @sherpan in <https://github.com/openvinotoolkit/anomalib/pull/517>
- 🐞 Fix linting issues by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/535>
- 🐞 Bug Fix: Solve NaN values of anomaly scores for PatchCore model by @bsl546 in <https://github.com/openvinotoolkit/anomalib/pull/549>
- 🐞 Bug Fix: Help description for argument task by @youngquan in <https://github.com/openvinotoolkit/anomalib/pull/547>
- reutrn results of load_state_dict func by @zywvvd in <https://github.com/openvinotoolkit/anomalib/pull/546>
- 🔨 Pass `pre-trained` from config to `ModelLightning` by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/529>
- Benchmarking tool with Comet by @sherpan in <https://github.com/openvinotoolkit/anomalib/pull/545>
- Add map_location when loading the weights by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/562>
- Add patchcore to openvino export test + upgrade lightning by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/565>
- 🐞 Fix category check for folder dataset in anomalib CLI by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/567>
- Refactor `PreProcessor` and fix `Visualizer` denormalization issue. by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/570>
- 🔨 Check for successful openvino conversion by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/571>
- Comet HPO by @sherpan in <https://github.com/openvinotoolkit/anomalib/pull/563>
- Fix patchcore image-level score computation by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/580>
- Fix anomaly map computation in CFlow when batch size is 1. by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/589>
- Documentation refactor by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/576>
- ✨ Add notebook for hpo by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/592>
- 🐞 Fix comet HPO by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/597>
- ✨ Replace keys from benchmarking script by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/595>
- Update README.md by @Owaiskhan9654 in <https://github.com/openvinotoolkit/anomalib/pull/623>
- 🐳 Containerize CI by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/616>
- add deprecation warning to denormalize class by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/629>
- Anomalib CLI Improvements - Update metrics and create post_processing section in the config file by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/607>
- Convert adaptive_threshold to Enum in configs by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/637>
- Create meta_data.json with ONNX export as well as OpenVINO export by @calebmm in <https://github.com/openvinotoolkit/anomalib/pull/636>
- 🖌 refactor export callback by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/640>
- 🐞 Address docs build by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/639>
- Optimized inference with onnx for patchcore. by @acai66 in <https://github.com/openvinotoolkit/anomalib/pull/652>

New Contributors

- @sherpan made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/517>
- @bsl546 made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/549>
- @youngquan made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/547>
- @zywvvd made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/546>
- @Owaiskhan9654 made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/623>
- @calebmm made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/636>
- @acai66 made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/652>

**Full Changelog**: <https://github.com/openvinotoolkit/anomalib/compare/v0.3.6...v0.3.7>

## [v.0.3.6] - 2022-09-02

### What's Changed

- Add publish workflow + update references to main by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/480>
- Fix Dockerfile by @ORippler in <https://github.com/openvinotoolkit/anomalib/pull/478>
- Fix onnx export by rewriting GaussianBlur by @ORippler in <https://github.com/openvinotoolkit/anomalib/pull/476>
- DFKDE refactor to accept any layer name like other models by @ashishbdatta in <https://github.com/openvinotoolkit/anomalib/pull/482>
- 🐞 Log benchmarking results in sub folder by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/483>
- 🐞 Fix Visualization keys in new CLI by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/487>
- fix Perlin augmenter for non divisible image sizes by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/490>
- 📝 Update the license headers by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/491>
- change default parameter values for DRAEM by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/495>
- Add reset methods to metrics by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/488>
- Feature Extractor Refactor by @ashishbdatta in <https://github.com/openvinotoolkit/anomalib/pull/451>
- Convert `AnomalyMapGenerator` to `nn.Module` by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/497>
- Add github pr labeler to automatically label PRs by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/498>
- Add coverage by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/499>
- 🐞 Change if check by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/501>
- SSPCAB implementation by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/500>
- 🛠 Refactor Normalization by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/496>
- Enable generic exporting of a trained model to ONNX or OpenVINO IR by @ashishbdatta in <https://github.com/openvinotoolkit/anomalib/pull/509>
- Updated documentation to add examples for exporting model by @ashishbdatta in <https://github.com/openvinotoolkit/anomalib/pull/515>
- Ignore pixel metrics in classification task by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/516>
- Update export documentation by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/521>
- FIX: PaDiM didn't use config.model.pre_trained. by @jingt2ch in <https://github.com/openvinotoolkit/anomalib/pull/514>
- Reset adaptive threshold between epochs by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/527>
- Add PRO metric by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/508>
- Set full_state_update attribute in custom metrics by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/531>
- 🐞 Set normalization method from anomaly module by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/530>

New Contributors

- @ashishbdatta made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/482>
- @jingt2ch made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/514>

**Full Changelog**: <https://github.com/openvinotoolkit/anomalib/compare/v0.3.5...v0.3.6>

## [v.0.3.5] - 2022-08-02

### What's Changed

- 🐞 Fix inference for draem by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/470>
- 🐞 🛠 Bug fix in the inferencer by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/475>

**Full Changelog**: <https://github.com/openvinotoolkit/anomalib/compare/v0.3.4...v0.3.5>

## [v.0.3.4] - 2022-08-01

### What's Changed

- Add encoding to LONG_DESCRIPTION in setup.py by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/419>
- Fix visualization by @ORippler in <https://github.com/openvinotoolkit/anomalib/pull/417>
- Fix openvino circular import issue by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/416>
- Fix inferener arg names and weight path issue. by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/422>
- Remove the redundant `loss_val` by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/425>
- 📃 Add documentation for gradio inference by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/427>
- Add `pre_train` as a configurable parameter by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/431>
- 🛠 Fix config files and refactor dfkde by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/435>
- Add metric visualizations by @ORippler in <https://github.com/openvinotoolkit/anomalib/pull/429>
- Fix: data split issue by @jeongHwarr in <https://github.com/openvinotoolkit/anomalib/pull/404>
- 🚚 Move perlin noise to common folder by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/424>
- Support null seed by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/437>
- 🐞 Change if statement by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/439>
- Fix visualizer for `classification`, `mode=simple` by @ORippler in <https://github.com/openvinotoolkit/anomalib/pull/442>
- Feature/aupro test by @ORippler in <https://github.com/openvinotoolkit/anomalib/pull/444>
- Replace PyTorchLightning extras dependency by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/455>
- 🛠 Fix `tox` configuration by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/446>
- Ignore ipynb files to detect the repo language by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/456>
- Move configuration from tox to pyproject by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/458>
- Add Torch Inferencer and Update Openvino and Gradio Inferencers. by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/453>
- Address markdownlint issues by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/460>
- 🐞 Fix HPO by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/462>
- Remove docs requirements by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/467>
- Add codacy badge to readme by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/468>

New Contributors

- @ORippler made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/417>
- @jeongHwarr made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/404>

**Full Changelog**: <https://github.com/openvinotoolkit/anomalib/compare/0.3.3...0.3.4>

## [v.0.3.3] - 2022-07-05

### What's Changed

- 🚚 Move initialization log message to base class by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/363>
- 🚚 Move logging from train.py to the getter functions by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/365>
- 🚜 Refactor loss computation by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/364>
- 📝 Add a technical blog post to explain how to run anomalib. by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/359>
- 📚 Add datamodule jupyter notebooks. by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/357>
- 📝 Add benchmarking notebook by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/353>
- ➕ Add PyPI downloads badge to the readme. by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/370>
- 📃 Update README.md by @innat in <https://github.com/openvinotoolkit/anomalib/pull/382>
- 💻 Create Anomalib CLI by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/378>
- 🐞 Fix configs to remove logging heatmaps from classification models. by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/387>
- ✨ Add FastFlow model training testing inference via Anomalib API by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/386>
- 🐞 PaDim occasionally NaNs in anomaly map by @VdLMV in <https://github.com/openvinotoolkit/anomalib/pull/392>
- 🖼 Inference + Visualization by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/390>

New Contributors

- @innat made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/382>
- @VdLMV made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/392>

**Full Changelog**: <https://github.com/openvinotoolkit/anomalib/compare/v.0.3.2...v.0.3.3>

## [v.0.3.2] - 2022-06-09

### What's Changed

- Refactor `AnomalyModule` and `LightningModules` to explicitly define class arguments. by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/315>
- 🐞 Fix inferencer in Gradio by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/332>
- fix too many open images warning by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/334>
- Upgrade wandb version by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/340>
- Minor fix: Update folder dataset + notebooks link by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/338>
- Upgrade TorchMetrics version by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/342>
- 🚀 Set pylint version in tox.ini by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/345>
- Add metrics configuration callback to benchmarking by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/346>
- ➕ Add FastFlow Model by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/336>
- ✨ Add toy dataset to the repository by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/350>
- Add DRAEM Model by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/344>
- 📃Update documentation by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/280>
- 🏷️ Refactor Datamodule names by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/354>
- ✨ Add Reverse Distillation by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/343>

**Full Changelog**: <https://github.com/openvinotoolkit/anomalib/compare/v.0.3.1...v.0.3.2>

## [v.0.3.1] - 2022-05-17

### What's Changed

- 🔧 Properly assign values to dataframe in folder dataset. by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/272>
- ➕ Add warnings ⚠️ for inproper task setting in config files. by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/274>
- Updated CHANGELOG.md by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/276>
- ➕ Add long description to `setup.py` to make `README.md` PyPI friendly. by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/279>
- ✨ Add hash check to data download by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/284>
- ➕ Add Gradio by @julien-blanchon in <https://github.com/openvinotoolkit/anomalib/pull/283>
- 🔨 Fix nncf key issue in nightly job by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/238>
- Visualizer improvements pt1 by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/293>
- 🧪 Fix nightly by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/299>
- 🧪 Add tests for benchmarking script by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/297>
- ➕ add input_info to nncf config when not defined by user by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/307>
- 🐞 Increase tolerance + nightly path fix by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/318>
- ➕ Add jupyter notebooks directory and first tutorial for `getting-started` by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/292>

New Contributors

- @julien-blanchon made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/283>

**Full Changelog**: <https://github.com/openvinotoolkit/anomalib/compare/v0.3.0...v.0.3.1>

## [v.0.3.0] - 2022-04-25

### What's Changed

- 🛠 ⚠️ Fix configs to properly use pytorch-lightning==1.6 with GPU by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/234>
- 🛠 Fix `get_version` in `setup.py` to avoid hard-coding version. by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/229>
- 🐞 Fix image loggers by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/233>
- Configurable metrics by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/230>
- Make OpenVINO throughput optional in benchmarking by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/239>
- 🔨 Minor fix: Ensure docs build runs only on isea-server by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/245>
- 🏷 Rename `--model_config_path` to `config` by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/246>
- Revert "🏷 Rename `--model_config_path` to `config`" by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/247>
- ➕ Add `--model_config_path` deprecation warning to `inference.py` by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/248>
- Add console logger by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/241>
- Add segmentation mask to inference output by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/242>
- 🛠 Fix broken mvtec link, and split url to fit to 120 by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/264>
- 🛠 Fix mask filenames in folder dataset by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/249>

**Full Changelog**: <https://github.com/openvinotoolkit/anomalib/compare/v0.2.6...v0.3.0>

## [v.0.2.6] - 2022-04-12

### What's Changed

- ✏️ Add `torchtext==0.9.1` to support Kaggle environments. by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/165>
- 🛠 Fix `KeyError:'label'` in classification folder dataset by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/175>
- 📝 Added MVTec license to the repo by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/177>
- load best model from checkpoint by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/195>
- Replace `SaveToCSVCallback` with PL `CSVLogger` by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/198>
- WIP Refactor test by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/197>
- 🔧 Dockerfile enhancements by @LukasBommes in <https://github.com/openvinotoolkit/anomalib/pull/172>
- 🛠 Fix visualization issue for fully defected images by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/194>
- ✨ Add hpo search using `wandb` by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/82>
- Separate train and validation transformations by @alexriedel1 in <https://github.com/openvinotoolkit/anomalib/pull/168>
- 🛠 Fix docs workflow by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/200>
- 🔄 CFlow: Switch soft permutation to false by default to speed up training. by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/201>
- Return only `image`, `path` and `label` for classification tasks in `Mvtec` and `Btech` datasets. by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/196>
- 🗑 Remove `freia` as dependency and include it in `anomalib/models/components` by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/174>
- Visualizer show classification and segmentation by @alexriedel1 in <https://github.com/openvinotoolkit/anomalib/pull/178>
- ↗️ Bump up `pytorch-lightning` version to `1.6.0` or higher by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/193>
- 🛠 Refactor DFKDE model by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/207>
- 🛠 Minor fixes: Update callbacks to AnomalyModule by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/208>
- 🛠 Minor update: Update pre-commit docs by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/206>
- ✨ Directory streaming by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/210>
- ✏️ Updated documentation for development on Docker by @LukasBommes in <https://github.com/openvinotoolkit/anomalib/pull/217>
- 🏷 Fix Mac M1 dependency conflicts by @dreaquil in <https://github.com/openvinotoolkit/anomalib/pull/158>
- 🐞 Set tiling off in pathcore to correctly reproduce the stats. by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/222>
- 🐞fix support for non-square images by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/204>
- Allow specifying feature layer and pool factor in DFM by @nahuja-intel in <https://github.com/openvinotoolkit/anomalib/pull/215>
- 📝 Add GANomaly metrics to readme by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/224>
- ↗️ Bump the version to 0.2.6 by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/223>
- 📝 🛠 Fix inconsistent benchmarking throughput/time by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/221>
- assign test split for folder dataset by @alexriedel1 in <https://github.com/openvinotoolkit/anomalib/pull/220>
- 🛠 Refactor model implementations by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/225>

New Contributors

- @LukasBommes made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/172>
- @dreaquil made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/158>
- @nahuja-intel made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/215>

**Full Changelog**: <https://github.com/openvinotoolkit/anomalib/compare/v.0.2.5...v0.2.6>

## [v.0.2.5] - 2022-03-25

### What's Changed

- Bugfix: fix random val/test split issue by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/48>
- Fix Readmes by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/46>
- Updated changelog by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/49>
- add distinction between image and pixel threshold in postprocessor by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/50>
- Fix docstrings by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/22>
- Fix networkx requirement by @LeonidBeynenson in <https://github.com/openvinotoolkit/anomalib/pull/52>
- Add min-max normalization by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/53>
- Change hardcoded dataset path to environ variable by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/51>
- Added cflow algorithm by @blakshma in <https://github.com/openvinotoolkit/anomalib/pull/47>
- perform metric computation on cpu by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/64>
- Fix Inferencer by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/60>
- Updated readme for cflow and change default config to reflect results by @blakshma in <https://github.com/openvinotoolkit/anomalib/pull/68>
- Fixed issue with model loading by @blakshma in <https://github.com/openvinotoolkit/anomalib/pull/69>
- Docs/sa/fix readme by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/71>
- Updated coreset subsampling method to improve accuracy by @blakshma in <https://github.com/openvinotoolkit/anomalib/pull/73>
- Revert "Updated coreset subsampling method to improve accuracy" by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/79>
- Replace `SupportIndex` with `int` by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/76>
- Added reference to official CFLOW repo by @blakshma in <https://github.com/openvinotoolkit/anomalib/pull/81>
- Fixed issue with k_greedy method by @blakshma in <https://github.com/openvinotoolkit/anomalib/pull/80>
- Fix Mix Data type issue on inferencer by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/77>
- Create CODE_OF_CONDUCT.md by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/86>
- ✨ Add GANomaly by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/70>
- Reorder auc only when needed by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/91>
- Bump up the pytorch lightning to master branch due to vulnurability issues by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/55>
- 🚀 CI: Nightly Build by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/66>
- Refactor by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/87>
- Benchmarking Script by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/17>
- 🐞 Fix tensor detach and gpu count issues in benchmarking script by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/100>
- Return predicted masks in predict step by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/103>
- Add Citation to the Readme by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/106>
- Nightly build by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/104>
- c_idx cast to LongTensor in random sparse projection by @alexriedel1 in <https://github.com/openvinotoolkit/anomalib/pull/113>
- Update Nightly by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/126>
- Updated logos by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/131>
- Add third-party-programs.txt file and update license by @LeonidBeynenson in <https://github.com/openvinotoolkit/anomalib/pull/132>
- 🔨 Increase inference + openvino support by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/122>
- Fix/da/image size bug by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/135>
- Fix/da/image size bug by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/140>
- optimize compute_anomaly_score by using torch native funcrtions by @alexriedel1 in <https://github.com/openvinotoolkit/anomalib/pull/141>
- Fix IndexError in adaptive threshold computation by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/146>
- Feature/data/btad by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/120>
- Update for nncf_task by @AlexanderDokuchaev in <https://github.com/openvinotoolkit/anomalib/pull/145>
- fix non-adaptive thresholding bug by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/152>
- Calculate feature map shape patchcore by @alexriedel1 in <https://github.com/openvinotoolkit/anomalib/pull/148>
- Add `transform_config` to the main `config.yaml` file. by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/156>
- Add Custom Dataset Training Support by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/154>
- Added extension as an option when saving the result images. by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/162>
- Update `anomalib` version and requirements by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/163>

New Contributors

- @LeonidBeynenson made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/52>
- @blakshma made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/47>
- @alexriedel1 made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/113>
- @AlexanderDokuchaev made their first contribution in <https://github.com/openvinotoolkit/anomalib/pull/145>

**Full Changelog**: <https://github.com/openvinotoolkit/anomalib/compare/v.0.2.4...v.0.2.5>

## [v.0.2.4] - 2021-12-22

### What's Changed

- Bump up the version to 0.2.4 by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/45>
- fix heatmap color scheme by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/44>

**Full Changelog**: <https://github.com/openvinotoolkit/anomalib/compare/v.0.2.3...v.0.2.4>

## [v.0.2.3] - 2021-12-23

### What's Changed

- Address docs build failing issue by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/39>
- Fix docs pipeline 📄 by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/41>
- Feature/dick/anomaly score normalization by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/35>
- Shuffle train dataloader by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/42>

**Full Changelog**: <https://github.com/openvinotoolkit/anomalib/compare/v0.2.2...v.0.2.3>

## [v0.2.0 Pre-release] - 2021-12-14

### What's Changed

- Address compatibility issues with OTE, that are caused by the legacy code. by @samet-akcay in [#24](https://github.com/openvinotoolkit/anomalib/pull/24)
- Initial docs string by @ashwinvaidya17 in [#9](https://github.com/openvinotoolkit/anomalib/pull/9)
- Load model did not work correctly as DFMModel did not inherit by @ashwinvaidya17 in [#5](https://github.com/openvinotoolkit/anomalib/pull/5)
- Refactor/samet/data by @samet-akcay in [#8](https://github.com/openvinotoolkit/anomalib/pull/8)
- Delete make.bat by @samet-akcay in [#11](https://github.com/openvinotoolkit/anomalib/pull/11)
- TorchMetrics by @djdameln in [#7](https://github.com/openvinotoolkit/anomalib/pull/7)
- ONNX node naming by @djdameln in [#13](https://github.com/openvinotoolkit/anomalib/pull/13)
- Add FPS counter to `TimerCallback` by @ashwinvaidya17 in [#12](https://github.com/openvinotoolkit/anomalib/pull/12)

Contributors

- @ashwinvaidya17
- @djdameln
- @samet-akcay

**Full Changelog**: <https://github.com/openvinotoolkit/anomalib/commits/v0.2.0>
