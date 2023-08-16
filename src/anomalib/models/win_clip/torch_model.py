# Original Code
# https://github.com/caoyunkang/WinClip.
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from .third_party import CLIPAD
from torch.nn import functional as F
from .third_party.ad_prompts import *
from PIL import Image

valid_backbones = ['ViT-B-16-plus-240']
valid_pretrained_datasets = ['laion400m_e32']

from torchvision import transforms

mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]

def _convert_to_rgb(image):
    return image.convert('RGB')

class WinClipAD(torch.nn.Module):
    def __init__(
            self,
            device,
            # output_size=(256, 256),
            backbone="ViT-B-16-plus-240",
            pretrained_dataset="laion400m_e32",
            scales=(2, 3),
            precision='fp32',
            # **kwargs
        ):
        '''

        :param out_size_h:
        :param out_size_w:
        :param device:
        :param backbone:
        :param pretrained_dataset:
        '''
        super(WinClipAD, self).__init__()

        # self.out_size_h, self.out_size_w = output_size
        self.precision = precision  # -40% GPU memory (2.8G->1.6G) with slight performance drop 

        self.device = device
        self.get_model(backbone, pretrained_dataset, scales)
        self.phrase_form = '{}'

        # version v1: no norm for each of linguistic embedding
        # version v1:    norm for each of linguistic embedding
        self.version = 'V1' # V1:
        # visual textual, textual_visual
        self.fusion_version = 'textual_visual'

        # self.transform = transforms.Compose([
        #     transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.BICUBIC),
        #     transforms.CenterCrop(kwargs['img_cropsize']),
        #     _convert_to_rgb,
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean_train, std=std_train)])

        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.NEAREST),
        #     transforms.CenterCrop(kwargs['img_cropsize']),
        #     transforms.ToTensor()])

        print(f'fusion version: {self.fusion_version}')

    def get_model(self, backbone, pretrained_dataset, scales):

        assert backbone in valid_backbones
        assert pretrained_dataset in valid_pretrained_datasets

        model, _, _ = CLIPAD.create_model_and_transforms(model_name=backbone, pretrained=pretrained_dataset, scales=scales, precision = self.precision)
        tokenizer = CLIPAD.get_tokenizer(backbone)
        model.eval().to(self.device)

        self.masks = model.visual.masks
        self.scale_begin_indx = model.visual.scale_begin_indx
        self.model = model
        self.tokenizer = tokenizer
        self.normal_text_features = None
        self.abnormal_text_features = None
        self.grid_size = model.visual.grid_size
        self.visual_gallery = None
        print("self.grid_size", self.grid_size)

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):

        if self.precision == "fp16":
            image = image.half()
        image_features = self.model.encode_image(image)
        return [f / f.norm(dim=-1, keepdim=True) for f in image_features]

    @torch.no_grad()
    def encode_text(self, text: torch.Tensor):
        text_features = self.model.encode_text(text)
        return text_features
    
    def build_text_feature_gallery(self, category: str):
        normal_phrases = []
        abnormal_phrases = []


        # some categories can be renamed to generate better embedding
        #if category == 'grid':
        #    category  = 'chain-link fence'
        #if category == 'toothbrush':
        #    category = 'brush' #'brush' #
        for template_prompt in template_level_prompts:
            # normal prompts
            for normal_prompt in state_level_normal_prompts:
                phrase = template_prompt.format(normal_prompt.format(category))
                normal_phrases += [phrase]

            # abnormal prompts
            for abnormal_prompt in state_level_abnormal_prompts:
                phrase = template_prompt.format(abnormal_prompt.format(category))
                abnormal_phrases += [phrase]

        normal_phrases = self.tokenizer(normal_phrases).to(self.device)
        abnormal_phrases = self.tokenizer(abnormal_phrases).to(self.device)

        
        if self.version == "V1":
            normal_text_features = self.encode_text(normal_phrases)
            abnormal_text_features = self.encode_text(abnormal_phrases)
        elif self.version == "V2":
            normal_text_features = []
            for phrase_id in range(normal_phrases.size()[0]):
                normal_text_feature = self.encode_text(normal_phrases[phrase_id].unsqueeze(0))
                normal_text_feature = normal_text_feature/normal_text_feature.norm(dim=-1, keepdim=True)
                normal_text_features.append(normal_text_feature)
            normal_text_features = torch.cat(normal_text_features, 0).half()
            abnormal_text_features = []
            for phrase_id in range(abnormal_phrases.size()[0]):
                abnormal_text_feature = self.encode_text(abnormal_phrases[phrase_id].unsqueeze(0))
                abnormal_text_feature = abnormal_text_feature/abnormal_text_feature.norm(dim=-1, keepdim=True)
                abnormal_text_features.append(abnormal_text_feature)
            abnormal_text_features = torch.cat(abnormal_text_features, 0).half()
        else:
            raise NotImplementedError

        avr_normal_text_features = torch.mean(normal_text_features, dim=0, keepdim=True)
        avr_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)

        self.avr_normal_text_features = avr_normal_text_features
        self.avr_abnormal_text_features = avr_abnormal_text_features
        self.text_features = torch.cat([self.avr_normal_text_features,
                                        self.avr_abnormal_text_features], dim=0)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def build_image_feature_gallery(self, normal_images):

        self.visual_gallery = []
        visual_features = self.encode_image(normal_images)

        for scale_index in range(len(self.scale_begin_indx)):
            if scale_index == len(self.scale_begin_indx) - 1:
                scale_features = visual_features[self.scale_begin_indx[scale_index]:]
            else:
                scale_features = visual_features[self.scale_begin_indx[scale_index]:self.scale_begin_indx[scale_index+1]]

            self.visual_gallery += [torch.cat(scale_features, dim=0)]


    def calculate_textual_anomaly_score(self, visual_features):
        N = visual_features[0].shape[0]
        scale_anomaly_scores = []
        token_anomaly_scores = torch.zeros((N,self.grid_size[0] * self.grid_size[1]))
        token_weights = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
        for indx, (features, mask) in enumerate(zip(visual_features, self.masks)):
            normality_and_abnormality_score = (100.0 * features @ self.text_features.T).softmax(dim=-1)
            normality_score = normality_and_abnormality_score[:, 0]
            abnormality_score = normality_and_abnormality_score[:, 1]
            normality_score = normality_score.cpu()

            mask = mask.reshape(-1)
            cur_token_anomaly_score = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
            if self.precision == "fp16":
                cur_token_anomaly_score = cur_token_anomaly_score.half()
            cur_token_anomaly_score[:, mask] = (1. / normality_score).unsqueeze(1)
            # cur_token_anomaly_score[:, mask] = (1. - normality_score).unsqueeze(1)
            cur_token_weight = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
            cur_token_weight[:, mask] = 1.

            if indx in self.scale_begin_indx[1:]:
                # deal with the first two scales
                token_anomaly_scores = token_anomaly_scores / token_weights
                scale_anomaly_scores.append(token_anomaly_scores)

                # another scale, calculate from scratch
                token_anomaly_scores = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
                token_weights = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))

            token_weights += cur_token_weight
            token_anomaly_scores += cur_token_anomaly_score

        # deal with the last one
        token_anomaly_scores = token_anomaly_scores / token_weights
        scale_anomaly_scores.append(token_anomaly_scores)

        scale_anomaly_scores = torch.stack(scale_anomaly_scores, dim=0)
        scale_anomaly_scores = torch.mean(scale_anomaly_scores, dim=0)
        scale_anomaly_scores = 1. - 1. / scale_anomaly_scores

        anomaly_map = scale_anomaly_scores.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)
        return anomaly_map

    def calculate_visual_anomaly_score(self, visual_features):
        N = visual_features[0].shape[0]
        scale_anomaly_scores = []
        token_anomaly_scores = torch.zeros((N,self.grid_size[0] * self.grid_size[1]))
        token_weights = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))

        cur_scale_indx = 0
        cur_visual_gallery = self.visual_gallery[cur_scale_indx]

        for indx, (features, mask) in enumerate(zip(visual_features, self.masks)):
            normality_score = 0.5 * (1 - (features @ cur_visual_gallery.T).max(dim=1)[0])
            normality_score = normality_score.cpu()

            mask = mask.reshape(-1)
            cur_token_anomaly_score = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
            if self.precision == "fp16":
                cur_token_anomaly_score = cur_token_anomaly_score.half()
            cur_token_anomaly_score[:, mask] = normality_score.unsqueeze(1)
            # cur_token_anomaly_score[:, mask] = (1. - normality_score).unsqueeze(1)
            cur_token_weight = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
            cur_token_weight[:, mask] = 1.

            if indx in self.scale_begin_indx[1:]:
                cur_scale_indx += 1
                cur_visual_gallery = self.visual_gallery[cur_scale_indx]
                # deal with the first two scales
                token_anomaly_scores = token_anomaly_scores / token_weights
                scale_anomaly_scores.append(token_anomaly_scores)

                # another scale, calculate from scratch
                token_anomaly_scores = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
                token_weights = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))

            token_weights += cur_token_weight
            token_anomaly_scores += cur_token_anomaly_score

        # deal with the last one
        token_anomaly_scores = token_anomaly_scores / token_weights
        scale_anomaly_scores.append(token_anomaly_scores)

        scale_anomaly_scores = torch.stack(scale_anomaly_scores, dim=0)
        scale_anomaly_scores = torch.mean(scale_anomaly_scores, dim=0)

        anomaly_map = scale_anomaly_scores.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)
        return anomaly_map

    def forward(self, images):

        visual_features = self.encode_image(images)
        textual_anomaly_map = self.calculate_textual_anomaly_score(visual_features)
        if self.visual_gallery is not None:
            visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
        else:
            visual_anomaly_map = textual_anomaly_map

        if self.fusion_version == 'visual':
            anomaly_map = visual_anomaly_map
        elif self.fusion_version == 'textual':
            anomaly_map = textual_anomaly_map
        else:
            anomaly_map = 1. / (1. / textual_anomaly_map + 1. / visual_anomaly_map)

        # anomaly_map = F.interpolate(anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear', align_corners=False)
        am_np = anomaly_map.squeeze(1).cpu().numpy()

        am_np_list = []

        for i in range(am_np.shape[0]):
            # am_np[i] = gaussian_filter(am_np[i], sigma=4)
            am_np_list.append(am_np[i])

        return am_np_list

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()
