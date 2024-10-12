import torch
from torch import nn
import numpy as np
import typing as tp
from .conditioners import MultiConditioner, create_multi_conditioner_from_conditioning_config
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
            conditioner: MultiConditioner,
            io_channels,
            sample_rate,
            min_input_length: int,
            diffusion_objective: tp.Literal["v", "rectified_flow"] = "v",
            pretransform: tp.Optional[Pretransform] = None,
            cross_attn_cond_ids: tp.List[str] = [],
            global_cond_ids: tp.List[str] = [],
            input_concat_ids: tp.List[str] = [],
            prepend_cond_ids: tp.List[str] = [],
            ):
        super().__init__()

        self.model = DiffusionTransformer(**diffusion_model_config)
        self.conditioner = conditioner # T5 encoder, start/end time encoder는 여기 위치한다.
        self.io_channels = io_channels # 64
        self.sample_rate = sample_rate # 44100
        self.diffusion_objective = diffusion_objective
        self.pretransform = pretransform
        self.cross_attn_cond_ids = cross_attn_cond_ids
        self.global_cond_ids = global_cond_ids
        self.input_concat_ids = input_concat_ids
        self.prepend_cond_ids = prepend_cond_ids
        self.min_input_length = min_input_length

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
        return self.model(x, t, **self.get_conditioning_inputs(cond), **kwargs)

    def generate(self, *args, **kwargs):
        return generate_diffusion_cond(self, *args, **kwargs)

class DiTWrapper(nn.Module):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__()

        self.model = DiffusionTransformer(*args, **kwargs)

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self,
                x,
                t,
                cross_attn_cond=None,
                cross_attn_mask=None,
                negative_cross_attn_cond=None,
                negative_cross_attn_mask=None,
                input_concat_cond=None,
                negative_input_concat_cond=None,
                global_cond=None,
                negative_global_cond=None,
                prepend_cond=None,
                prepend_cond_mask=None,
                cfg_scale=1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = True,
                rescale_cfg: bool = False,
                scale_phi: float = 0.0,
                **kwargs):

        assert batch_cfg, "batch_cfg must be True for DiTWrapper"
        #assert negative_input_concat_cond is None, "negative_input_concat_cond is not supported for DiTWrapper"

        return self.model(
            x,
            t,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=cross_attn_mask,
            negative_cross_attn_cond=negative_cross_attn_cond,
            negative_cross_attn_mask=negative_cross_attn_mask,
            input_concat_cond=input_concat_cond,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            cfg_scale=cfg_scale,
            cfg_dropout_prob=cfg_dropout_prob,
            scale_phi=scale_phi,
            global_embed=global_cond,
            **kwargs)

def create_diffusion_cond_from_config(config: tp.Dict[str, tp.Any]):
    model_config = config["model"]
    model_type = config["model_type"]

    diffusion_config = model_config.get('diffusion', None)
    assert diffusion_config is not None, "Must specify diffusion config"

    diffusion_model_type = diffusion_config.get('type', None)
    assert diffusion_model_type is not None, "Must specify diffusion model type"

    diffusion_model_config = diffusion_config.get('config', None)
    assert diffusion_model_config is not None, "Must specify diffusion model config"

    if diffusion_model_type == 'dit':
        diffusion_model = DiTWrapper(**diffusion_model_config) # 여기에 들어가는구나!

    io_channels = model_config.get('io_channels', None) # 64
    assert io_channels is not None, "Must specify io_channels in model config"

    sample_rate = config.get('sample_rate', None) # 44100
    assert sample_rate is not None, "Must specify sample_rate in config"

    diffusion_objective = diffusion_config.get('diffusion_objective', 'v')
    conditioning_config = model_config.get('conditioning', None)

    conditioner = None
    if conditioning_config is not None:
        # 입력 value의 key에 맞는 conditioner를 태우는 역할을 하는 Multi Conditioner를 반환한다. 그래서 무슨 컨디션이던지 여기 태우게 됨.
        conditioner = create_multi_conditioner_from_conditioning_config(conditioning_config)

    cross_attention_ids = diffusion_config.get('cross_attention_cond_ids', [])
    global_cond_ids = diffusion_config.get('global_cond_ids', [])
    input_concat_ids = diffusion_config.get('input_concat_ids', [])
    prepend_cond_ids = diffusion_config.get('prepend_cond_ids', [])

    pretransform = model_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)
        min_input_length = pretransform.downsampling_ratio
    else:
        min_input_length = 1

    if diffusion_model_type == "adp_cfg_1d" or diffusion_model_type == "adp_1d":
        min_input_length *= np.prod(diffusion_model_config["factors"])
    elif diffusion_model_type == "dit":
        min_input_length *= diffusion_model.model.patch_size

    # Get the proper wrapper class

    extra_kwargs = {}

    if model_type == "diffusion_cond" or model_type == "diffusion_cond_inpaint":
        wrapper_fn = ConditionedDiffusionModelWrapper

        extra_kwargs["diffusion_objective"] = diffusion_objective
            
    return wrapper_fn(
        generate_diffusion_cond,
        conditioner,
        min_input_length=min_input_length,
        sample_rate=sample_rate,
        cross_attn_cond_ids=cross_attention_ids,
        global_cond_ids=global_cond_ids,
        input_concat_ids=input_concat_ids,
        prepend_cond_ids=prepend_cond_ids,
        pretransform=pretransform,
        io_channels=io_channels,
        **extra_kwargs
    )
