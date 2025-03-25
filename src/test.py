from anomalib.data import UCSDped, ShanghaiTech
from anomalib.models import fuvas,AiVad
from anomalib.engine import Engine
from torchvision.transforms.v2 import Resize, Compose,Normalize
from anomalib.metrics import AUROC, F1Score, Evaluator
from anomalib.pre_processing import PreProcessor


train_set = UCSDped(root='/data4/video_AD/UCSD/UCSD_Anomaly_Dataset.v1p2',category='UCSDped2',
                    clip_length_in_frames=13,
                    frames_between_clips=10,
                    train_batch_size = 2,
                    eval_batch_size = 1,
                    num_workers=1)
# train_set = ShanghaiTech(root='/data4/video_AD/shanghaitech/shanghaitech',scene=1,
#                     clip_length_in_frames=13,
#                     frames_between_clips=1,
#                     train_batch_size = 2,
#                     eval_batch_size = 1,
#                     num_workers=1)

metrics = [
    AUROC(fields=["pred_score", "gt_label"],prefix="frame_"),
    AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
    F1Score(fields=["pred_label", "gt_label"])
]
transform = Compose([Resize((448, 512)), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
pre_processor = PreProcessor(transform=transform)
# Create evaluator with metrics
evaluator = Evaluator(test_metrics=metrics)
model = fuvas(backbone='x3d_s',layer='blocks.4',spatial_pool=True, pooling_kernel_size=1, do_seg=True,pre_processor=pre_processor,visualizer=False,evaluator=evaluator)
# model2 = AiVad(pre_processor=pre_processor,visualizer=False,evaluator=evaluator)
engine = Engine()

engine.train(datamodule=train_set, model=model)