import clip
import torch
from torch import nn, Tensor
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.ops import roi_align


class AiVadModel(nn.Module):
    def __init__(self):
        super().__init__()
        # initialize flow extractor
        self.flow_extractor = raft_small(weights=Raft_Small_Weights)
        # initialize region extractor
        self.region_extractor = keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights, box_score_thresh=0.8
        )
        # initialize feature extractor
        self.encoder, _ = clip.load("ViT-B/16")

    def forward(self, batch):
        self.flow_extractor.eval()
        self.region_extractor.eval()
        self.encoder.eval()

        batch_size = batch.shape[0]

        # 1. compute optical flow
        first_frame = batch[:, 0, ...]
        last_frame = batch[:, -1, ...]
        flow = self.flow_extractor(first_frame, last_frame)[
            -1
        ]  # only interested in output of last iteration of flow estimation

        # 2. extract regions
        regions = self.region_extractor(last_frame)
        # convert from list of [N, 4] tensors to single [N, 5] tensor where each row is [index-in-batch, x1, y1, x2, y2]
        boxes_list = [batch_item["boxes"] for batch_item in regions]
        indices = torch.repeat_interleave(
            torch.arange(len(regions)), Tensor([boxes.shape[0] for boxes in boxes_list]).int()
        )
        boxes = torch.cat([indices.unsqueeze(1).to(batch.device), torch.cat(boxes_list)], dim=1)

        flow_regions = roi_align(flow, boxes, output_size=[224, 224])
        rgb_regions = roi_align(last_frame, boxes, output_size=[224, 224])

        # 3. extract velocity
        velocity = extract_velocity(flow_regions)
        velocity = [velocity[indices == i] for i in range(batch_size)]  # convert back to list

        # 4. extract pose
        poses = extract_poses(regions)

        # 5. CLIP
        rgb_regions = [rgb_regions[indices == i] for i in range(batch_size)]  # convert back to list
        features = [self.encoder.encode_image(regions) for regions in rgb_regions]

        if self.training:
            # return features
            return velocity, poses, features


def extract_velocity(flows):
    # cartesian to polar
    # mag = torch.linalg.norm(flow, axis=0, ord=2)
    mag_batch = torch.linalg.norm(flows, axis=1, ord=2)
    # theta = torch.atan2(flow[0], flow[1])
    theta_batch = torch.atan2(flows[:, 0, ...], flows[:, 1, ...])

    # compute velocity histogram
    n_bins = 8
    velocity_hictograms = []
    for mag, theta in zip(mag_batch, theta_batch):
        histogram = torch.histogram(
            input=theta.cpu(), bins=n_bins, range=(-torch.pi, torch.pi), weight=mag.cpu() / mag.sum().cpu()
        )
        velocity_hictograms.append(histogram.hist)
    return torch.stack(velocity_hictograms)


def extract_poses(regions):
    poses = []
    for region in regions:
        boxes = region["boxes"].unsqueeze(1)
        keypoints = region["keypoints"]
        normalized_keypoints = (keypoints[..., :2] - boxes[..., :2]) / (boxes[..., 2:] - boxes[..., :2])
        poses.append(normalized_keypoints)
    return poses
