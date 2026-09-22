from typing import Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transformers import AutoModel


class DINOv2ImageEncoder(nn.Module):
    def __init__(self, model_name: Literal[
        "facebook/dinov2-with-registers-large",
        "facebook/dinov2-large"
    ]):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.model.requires_grad_(False)
        self.model.eval()

        DINOv2_INPUT_MEAN = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32)[
            None, :, None, None
        ]
        DINOv2_INPUT_STD = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float32)[
            None, :, None, None
        ]
        self.register_buffer("DINOv2_INPUT_MEAN", DINOv2_INPUT_MEAN, persistent=False)
        self.register_buffer("DINOv2_INPUT_STD", DINOv2_INPUT_STD, persistent=False)
        self.max_size = 518
        self.hidden_size = self.model.config.hidden_size

    def preprocess(self, image: torch.Tensor):
        B, C, H, W = image.shape
        assert C == 3 and H <= self.max_size and W <= self.max_size
        image = (image - self.DINOv2_INPUT_MEAN.to(image)) / self.DINOv2_INPUT_STD.to(image)
        return image
    
    def forward(self, image: torch.Tensor):
        image = self.preprocess(image)
        features = self.model(image).last_hidden_state
        return features