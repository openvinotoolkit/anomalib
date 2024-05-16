anomalib fit --model Patchcore --data mvtec.yaml --default_root_dir export_path
anomalib export --model Patchcore --export_type TORCH --ckpt_path export_path/Patchcore/MVTec/bottle/latest/weights/lightning/model.ckpt
