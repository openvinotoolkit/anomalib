# Example Configurations

With the Anomalib CLI, you can combine model and data configurations for `train`, `fit`, `validate`, `test`, and `predict`.

The configurations in this directory provide baseline YAML templates:

```text
configs/
├── data/
│   ├── adam_3d.yaml
│   ├── autovi.yaml
│   ├── avenue.yaml
│   ├── bmad.yaml
│   ├── btech.yaml
│   ├── datumaro.yaml
│   ├── folder.yaml
│   ├── kaputt.yaml
│   ├── kolektor.yaml
│   ├── mpdd.yaml
│   ├── mvtec.yaml
│   ├── mvtec_3d.yaml
│   ├── mvtec_loco.yaml
│   ├── mvtecad2.yaml
│   ├── realiad.yaml
│   ├── shanghaitech.yaml
│   ├── tabular.yaml
│   ├── ucsd_ped.yaml
│   ├── vad.yaml
│   └── visa.yaml
└── model/
    ├── ai_vad.yaml
    ├── anomaly_dino.yaml
    ├── anomalyvfm.yaml
    ├── cfa.yaml
    ├── cflow.yaml
    ├── cfm.yaml
    ├── csflow.yaml
    ├── dfkde.yaml
    ├── dfm.yaml
    ├── dinomaly.yaml
    ├── draem.yaml
    ├── dsr.yaml
    ├── efficient_ad.yaml
    ├── fastflow.yaml
    ├── fre.yaml
    ├── ganomaly.yaml
    ├── glass.yaml
    ├── inp_former.yaml
    ├── l2bt.yaml
    ├── padim.yaml
    ├── patchcore.yaml
    ├── patchflow/
    ├── reverse_distillation.yaml
    ├── stfpm.yaml
    ├── uflow.yaml
    └── uninet.yaml
```

## Examples

Train a model with combined config files:

```bash
anomalib train -c examples/configs/model/padim.yaml --data examples/configs/data/mvtec.yaml
```

```bash
anomalib train -c examples/configs/model/stfpm.yaml --data examples/configs/data/visa.yaml
```
