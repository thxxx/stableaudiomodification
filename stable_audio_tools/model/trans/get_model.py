import torch
from torch import nn
import numpy as np
import typing as tp
from .conditioners import create_multi_conditioner_from_conditioning_config
from .dit import DiffusionTransformer
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from ..inference.generation import generate_diffusion_cond

class ConditionedDiffusionModelWrapper(nn.Module):
    """
    A diffusion model that takes in conditioning
    """
    def __init__(
            self,
            diffusion_model_config,
            conditioning_config, 
            pretransform_config,
            io_channels,
            sample_rate,
            diffusion_objective: tp.Literal["v", "rectified_flow"] = "v",
            cross_attn_cond_ids: tp.List[str] = [],
            global_cond_ids: tp.List[str] = [],
            input_concat_ids: tp.List[str] = [],
            prepend_cond_ids: tp.List[str] = [],
            ):
        super().__init__()

        self.model = DiffusionTransformer(**diffusion_model_config)
        self.conditioner = create_multi_conditioner_from_conditioning_config(conditioning_config)
        self.pretransform = create_pretransform_from_config(pretransform_config, sample_rate)

        # self.conditioner = conditioner # T5 encoder, start/end time encoder는 여기 위치한다.
        self.io_channels = io_channels # 64
        self.sample_rate = sample_rate # 44100
        self.diffusion_objective = diffusion_objective
        self.cross_attn_cond_ids = cross_attn_cond_ids
        self.global_cond_ids = global_cond_ids
        self.input_concat_ids = input_concat_ids
        self.prepend_cond_ids = prepend_cond_ids
        self.min_input_length = self.pretransform.downsampling_rate

    # 여기 컨디션 텐서가 들어와서 리턴값이 샘플링 함수에 사용된다. inference에서
    def get_conditioning_inputs(self, conditioning_tensors: tp.Dict[str, tp.Any]):
        cross_attention_input = None
        cross_attention_masks = None
        global_cond = None
        input_concat_cond = None
        prepend_cond = None
        prepend_cond_mask = None

        if len(self.cross_attn_cond_ids) > 0:
            # Concatenate all cross-attention inputs over the sequence dimension
            # Assumes that the cross-attention inputs are of shape (batch, seq, channels)
            cross_attention_input = []
            cross_attention_masks = []

            # 아하 inference에서는 여기만 들어가는구나..? ids 값은 ['prompt', 'seconds_start', 'seconds_total']
            for key in self.cross_attn_cond_ids:
                cross_attn_in, cross_attn_mask = conditioning_tensors[key]

                # Add sequence dimension if it's not there
                if len(cross_attn_in.shape) == 2:
                    cross_attn_in = cross_attn_in.unsqueeze(1)
                    cross_attn_mask = cross_attn_mask.unsqueeze(1)

                cross_attention_input.append(cross_attn_in)
                cross_attention_masks.append(cross_attn_mask)

            cross_attention_input = torch.cat(cross_attention_input, dim=1)
            cross_attention_masks = torch.cat(cross_attention_masks, dim=1)

        if len(self.global_cond_ids) > 0:
            # Concatenate all global conditioning inputs over the channel dimension
            # Assumes that the global conditioning inputs are of shape (batch, channels)
            global_conds = []
            for key in self.global_cond_ids:
                global_cond_input = conditioning_tensors[key][0]

                global_conds.append(global_cond_input)

            # Concatenate over the channel dimension
            global_cond = torch.cat(global_conds, dim=-1)

            if len(global_cond.shape) == 3:
                global_cond = global_cond.squeeze(1)

        if len(self.input_concat_ids) > 0:
            # Concatenate all input concat conditioning inputs over the channel dimension
            # Assumes that the input concat conditioning inputs are of shape (batch, channels, seq)
            input_concat_cond = torch.cat([conditioning_tensors[key][0] for key in self.input_concat_ids], dim=1)

        if len(self.prepend_cond_ids) > 0:
            # Concatenate all prepend conditioning inputs over the sequence dimension
            # Assumes that the prepend conditioning inputs are of shape (batch, seq, channels)
            prepend_conds = []
            prepend_cond_masks = []

            for key in self.prepend_cond_ids:
                prepend_cond_input, prepend_cond_mask = conditioning_tensors[key]
                prepend_conds.append(prepend_cond_input)
                prepend_cond_masks.append(prepend_cond_mask)

            prepend_cond = torch.cat(prepend_conds, dim=1)
            prepend_cond_mask = torch.cat(prepend_cond_masks, dim=1)

        return {
            "cross_attn_cond": cross_attention_input,
            "cross_attn_mask": cross_attention_masks,
            "global_cond": global_cond,
            # 아래는 inference에 들어가는 값이 없다. 텍스트랑 타이밍만 썼을 때는! None
            "input_concat_cond": input_concat_cond,
            "prepend_cond": prepend_cond,
            "prepend_cond_mask": prepend_cond_mask
        }

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: tp.Dict[str, tp.Any], **kwargs):
        conds = self.get_conditioning_inputs(cond)
        return self.model(x, t, **conds, **kwargs)

    def generate(self, *args, **kwargs):
        return generate_diffusion_cond(self, *args, **kwargs)
