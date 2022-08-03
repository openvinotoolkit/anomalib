# Changelog

## v.0.3.5

## What's Changed

- 🐞 Fix inference for draem by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/470>
- 🐞 🛠 Bug fix in the inferencer by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/475>

**Full Changelog**: <https://github.com/openvinotoolkit/anomalib/compare/v0.3.4...v0.3.5>

## v.0.3.4

## What's Changed

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

## v.0.3.3

## What's Changed

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

## v.0.3.2

## What's Changed

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

## v.0.3.1

## What's Changed

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

## v.0.3.0

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

## v.0.2.6

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

## v.0.2.5

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

## v.0.2.4

### What's Changed

- Bump up the version to 0.2.4 by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/45>
- fix heatmap color scheme by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/44>

**Full Changelog**: <https://github.com/openvinotoolkit/anomalib/compare/v.0.2.3...v.0.2.4>

## v.0.2.3

### What's Changed

- Address docs build failing issue by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/39>
- Fix docs pipeline 📄 by @ashwinvaidya17 in <https://github.com/openvinotoolkit/anomalib/pull/41>
- Feature/dick/anomaly score normalization by @djdameln in <https://github.com/openvinotoolkit/anomalib/pull/35>
- Shuffle train dataloader by @samet-akcay in <https://github.com/openvinotoolkit/anomalib/pull/42>

**Full Changelog**: <https://github.com/openvinotoolkit/anomalib/compare/v0.2.2...v.0.2.3>

## v0.2.0 Pre-release (2021-12-15)

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
