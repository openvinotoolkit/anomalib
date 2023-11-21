# Configurations

With the new CLI, you can use multiple configurations for calling `fit`, `validate`, `test`, and `predict`.

The configurations in this folder are organized as follows.

```bash
configs/
├── data
│   ├── avenue.yaml
│   ├── btech.yaml
│   ├── folder_3d.yaml
│   ├── folder.yaml
│   ├── inference.yaml
│   ├── kolektor.yaml
│   ├── mvtec_3d.yaml
│   ├── mvtec.yaml
│   ├── shanghaitec.yaml
│   ├── ucsd_ped.yaml
│   └── visa.yaml
└── model
    ├── ai_vad.yaml
    ├── cfa.yaml
    ├── cflow.yaml
    ├── csflow.yaml
    ├── dfkde.yaml
    ├── dfm.yaml
    ├── draem.yaml
    ├── efficient_ad.yaml
    ├── fastflow.yaml
    ├── ganomaly.yaml
    ├── padim.yaml
    ├── patchcore.yaml
    ├── reverse_distillation.yaml
    ├── rkde.yaml
    └── stfpm.yaml

```

## Examples

```bash
anomalib fit -c configs/model/padim.yaml --data configs/data/mvtec.yaml
```

```bash
anomalib fit -c configs/model/stfpm.yaml --data configs/data/visa.yaml
```
