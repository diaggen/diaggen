from typing import *
import torch


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        # kwargs['if_use_mask'] = True
        pred = super()._inference_model(model, x_t, t, cond, **kwargs)
        # kwargs['if_use_mask'] = False
        kwargs['ordered_mask_dino'] = torch.zeros_like(kwargs['ordered_mask_dino'])
        neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        return (1 + cfg_strength) * pred - cfg_strength * neg_pred
