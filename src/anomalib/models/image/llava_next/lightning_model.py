"""Llm: Zero-/Few-Shot Anomaly Classification and Segmentation.

Paper No paper
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List

import base64
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, InterpolationMode, Normalize, Resize, Transform

from anomalib import LearningType
from anomalib.data.predict import PredictDataset
from anomalib.models.components import AnomalyModule

from anomalib.models.image.llava_next.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from anomalib.models.image.llava_next.conversation import conv_templates, SeparatorStyle
from anomalib.models.image.llava_next.model.builder import load_pretrained_model
from anomalib.models.image.llava_next.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import copy




logger = logging.getLogger(__name__)

__all__ = ["Llavanext"]


class Llavanext(AnomalyModule):
    """Llmollama Lightning model.

    Args:
        openai_key(str): The key to interact with openai,
                         https://platform.openai.com/docs/quickstart/step-2-set-up-your-api-key .
    """

    def __init__(
        self,
        k_shot = 0,
        temperature = 0,
        model_path = "liuhaotian/llava-v1.6-vicuna-7b",
        load_8bits = False,
        load_4bits = False,
        model_base = None,
        conver_mode = None,
        max_new_tokens = 100, # max 1024
        device = "cuda",
    ) -> None:
        super().__init__()
        self.model_str = "Llava :" + model_path
        self.k_shot = k_shot
        self.model_base = model_base
        self.temperature = temperature
        self.load8bits = load_8bits
        self.load4bits = load_4bits
        self.model_path = model_path
        self.conv_mode = conver_mode
        self.max_new_tokens = max_new_tokens
        # Model
        # disable_torch_init()


        self.pretrained = "lmms-lab/llama3-llava-next-8b"
        self.model_name = "llava_llama3"
        #self.device = "cuda"
        self.device_map = "auto"
        # Add any other thing you want to pass in llava_model_args
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(self.pretrained, None, self.model_name, device_map=self.device_map, attn_implementation=None)

        self.model.eval()
        self.model.tie_weights()

    def _setup(self):
        dataloader = self.trainer.datamodule.train_dataloader()
        pre_images = self.collect_reference_images(dataloader)
        self.pre_images = pre_images

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> None:
        """train Step of LLM."""
        del args, kwargs  # These variables are not used.
        # no train on llm
        return batch

    @staticmethod
    def configure_optimizers() -> None:
        """WinCLIP doesn't require optimization, therefore returns no optimizers."""
        return

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> dict:
        """Validation Step of WinCLIP."""
        del args, kwargs  # These variables are not used.
        bsize = len(batch["image_path"])
        out_list: list[str] = []
        pred_list = []
        for x in range(bsize):
            o = "NO - default"
            if self.k_shot > 0:
                o = str(self.api_call_fewShot(self.pre_images ,"", batch["image_path"][x])).strip()
            else:
                o = str(self.api_call( "", batch["image_path"][x])).strip()
            print(o)
            p = 0.0 if o.startswith("N") else 1.0
            out_list.append(o)
            pred_list.append(p)

        batch["str_output"] = out_list
        # [api_call( "", batch["image_path"][0])]*bsize  # the first img of the batch
        batch["pred_scores"] = torch.tensor(pred_list).to('cuda')
        #batch["pred_scores"] = batch["pred_scores"].to('cpu')
        return batch

    @property
    def trainer_arguments(self) -> dict[str, int | float]:
        """Set model-specific trainer arguments."""
        return {}

    @property
    def learning_type(self) -> LearningType:
        """The learning type of the model.

        Llm is a zero-/few-shot model, depending on the user configuration. Therefore, the learning type is
        set to ``LearningType.FEW_SHOT`` when ``k_shot`` is greater than zero and ``LearningType.ZERO_SHOT`` otherwise.
        """
        return LearningType.FEW_SHOT if self.k_shot else LearningType.ZERO_SHOT

    def collect_reference_images(self, dataloader: DataLoader) -> torch.Tensor:
        """Collect reference images for few-shot inference.

        The reference images are collected by iterating the training dataset until the required number of images are
        collected.

        Returns:
            ref_images (Tensor): A tensor containing the reference images.
        """
        ref_images = []
        for batch in dataloader:
            images = batch["image_path"][: self.k_shot - len(ref_images)]
            ref_images.extend( images)
            if self.k_shot == len(ref_images):
                break
        return ref_images


    def load_image(self,image_file:str)-> Image:
        if image_file.startswith('http://') or image_file.startswith('https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        print(image.size)
        return image

    def configure_transforms(self, image_size: tuple[int, int] | None = None) -> Transform:
        """Configure the default transforms used by the model."""
        if image_size is not None:
            logger.warning("Image size is not used in WinCLIP. The input image size is determined by the model.")
        return Compose(
            [
                Resize((520, 520), antialias=True, interpolation=InterpolationMode.BICUBIC),
                Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ],
        )



    def api_call_Default(self, prompt, image_path) -> str:
        image = self.load_image(image_path)
        image_size = image.size
        # Similar operation in model_worker.py
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if type(image_tensor) is list:
            image_tensor = [imag.to(self.model.device, dtype=torch.float16) for imag in image_tensor]
        else:
           image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        prompt = "Describe me if this image has an obious anomaly or not. if yes say 'YES:', follow by a description, and if not say 'NO' and finish."

        inp = f"{self.roles[0]}: {prompt} "

        if image is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            image = None
        conv = self.conv.copy()

        conv.append_message(self.conv.roles[0], inp)
        conv.append_message(self.conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
                prompt,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors='pt',
                ).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        #print(image_tensor.shape)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=not self.temperature > 0,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                #streamer=streamer,
                use_cache=True)

        outputs = self.tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs
        return outputs


    def api_call_fewShot(self, preImages, prompt, image_path) -> str:
        images = []
        images_size = []


        for img_path in preImages:
            print(img_path)
            i = self.load_image(img_path)
            # images_size.append(i.size)
            # images.append(i)

        image = self.load_image(image_path)
        image_size = image.size
        images.append(image)
        images.append(image)
        images_size.append(image_size)
        images_size.append(image_size)
        print(image_path)


        # Similar operation in model_worker.py
        image_tensor = process_images(images, self.image_processor, self.model.config)
        if type(image_tensor) is list:
            image_tensor = [imag.to(dtype=torch.float16) for imag in image_tensor]
        else:
           image_tensor = image_tensor.to(dtype=torch.float16)

        preprompt = "This is an object without any abnormality or defect."

        prompt = "You are an expert in a production factory examining products with defects the Describe me if the object in the second image has an anomaly or not. if yes start the response with: 'YES:', follow by a description, and if not say 'NO' and finish."
        # conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
        conv_template = "qwen_1_5" # Make sure you use correct chat template for different models

        question = DEFAULT_IMAGE_TOKEN + "\n "
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question + preprompt)
        conv.append_message(conv.roles[0], question + prompt)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        print(len(images_size))

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        # image_sizes = [image.size]


        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=images_size,
            do_sample=True,
            temperature=0.3,
            max_new_tokens=256,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        return text_outputs[0]
        prompt = "Describe me if this image has an obious anomaly or not. if yes say 'YES:', follow by a description, and if not say 'NO' and finish."




    def api_call(self, prompt, image_path) -> str:
        image = self.load_image(image_path)
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]

        prompt = """You are given an image and need to identify whether it has any visible defects or anomalies. If the image contains any defects, irregularities, or anomalies, respond with "YES:description" where "description" explains the specific defect(s) found. If the image does not contain any defects or anomalies, respond with "NO". Consider defects such as blurriness, incorrect color, artifacts, missing parts, distortion, or any other noticeable issues."""

        prompt = "You are a an expert with the Describe me if this image has an anomaly or not. if yes start the response with: 'YES:', follow by a description, and if not say 'NO' and finish. Consider defects such as blurriness, incorrect color, artifacts, missing parts, distortion, or any other noticeable issues."
        conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
        conv_template = "qwen_1_5" # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + "\n "
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_sizes = [image.size]


        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=256,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        return text_outputs[0]
        prompt = "Describe me if this image has an obious anomaly or not. if yes say 'YES:', follow by a description, and if not say 'NO' and finish."



    def inference(self, prompt, image_path):


        url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
        image = Image.open(requests.get(url, stream=True).raw)
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

        conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [image.size]


        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=256,
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        return text_outputs
