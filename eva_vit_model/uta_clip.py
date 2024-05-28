import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import eva_vit
from .transformer import text_transformer

class CLIP(nn.Module):
    def __init__(
        self,
        vision_model: str = 'eva_base_p16',
    ):
        super().__init__()
        self.visual = eva_vit.__dict__[vision_model]()
        self.text = text_transformer()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        return image_features, text_features, self.logit_scale.exp()