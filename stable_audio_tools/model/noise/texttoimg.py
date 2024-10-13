import typing as tp
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from x_transformers import ContinuousTransformerWrapper, Encoder
from .blocks import FourierFeatures
from .transformer import ContinuousTransformer

class DiffusionTransformer(nn.Module):
    """
    전체 모듈, 모델들의 어느정도까지를 포함하고 있는걸까?
    일은 self.transformer가 한다.
    여기서는 Classifier guidance 적용, 차원 일치시켜서 넣어주기?
    """
    def __init__(self, 
        io_channels=64,
        patch_size=1,
        embed_dim=1536,
        cond_token_dim=768,
        project_cond_tokens=False,
        global_cond_dim=1536,
        project_global_cond=True,
        input_concat_dim=0,
        prepend_cond_dim=0,
        depth=24, # DiT block을 몇개를 통과하는지.
        num_heads=24,
        transformer_type: tp.Literal["x-transformers", "continuous_transformer"] = "x-transformers", # continuous transformer
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        **kwargs):
        super().__init__()
        
        self.cond_token_dim = cond_token_dim

        # Timestep embeddings
        timestep_features_dim = 256
        self.timestep_features = FourierFeatures(1, timestep_features_dim)
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

        if cond_token_dim > 0:
            # Conditioning tokens
            cond_embed_dim = cond_token_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False)
            )
        else:
            cond_embed_dim = 0

        if global_cond_dim > 0:
            # Global conditioning
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim # 미지정 디폴트값으로 True, embed_dim 1536이 됨.
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, global_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(global_embed_dim, global_embed_dim, bias=False)
            )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )
        self.input_concat_dim = input_concat_dim # is 0 at inference
        dim_in = io_channels + self.input_concat_dim
        self.patch_size = patch_size # !what is this?
        # Transformer
        self.transformer_type = transformer_type
        self.global_cond_type = global_cond_type # prepend
        
        if self.transformer_type == "continuous_transformer":
            global_dim = embed_dim if self.global_cond_type == "adaLN" else None # The global conditioning is projected to the embed_dim already at this point

            self.transformer = ContinuousTransformer(
                dim=embed_dim, # 1536
                dim_in=dim_in * patch_size, # io_channels=64 + input_concat_dim=0
                dim_out=io_channels * patch_size, # io_channels=64
                depth=depth,
                dim_heads=embed_dim // num_heads,
                cross_attend = cond_token_dim > 0,
                cond_token_dim = cond_embed_dim,
                global_cond_dim=global_dim,
                **kwargs
            )
        elif self.transformer_type == "x-transformers":
            pass
            # self.transformer = ContinuousTransformerWrapper(
            #     dim_in=dim_in * patch_size,
            #     dim_out=io_channels * patch_size,
            #     max_seq_len=0, #Not relevant without absolute positional embeds
            #     attn_layers = Encoder(
            #         dim=embed_dim,
            #         depth=depth,
            #         heads=num_heads,
            #         attn_flash = True,
            #         cross_attend = cond_token_dim > 0,
            #         dim_context=None if cond_embed_dim == 0 else cond_embed_dim,
            #         zero_init_branch_output=True,
            #         use_abs_pos_emb = False,
            #         rotary_pos_emb=True,
            #         ff_swish = True,
            #         ff_glu = True,
            #         **kwargs
            #     )
            # )
        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    def _forward(
        self, 
        x, 
        t, 
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        return_info=False,
        **kwargs):

        if cross_attn_cond is not None: # 현재 세팅에서는 차원 변화 없음. 768
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)
        if global_embed is not None: # embed_dim이 되지만 값은 같다. 1536
            # Project the global conditioning to the embedding dimension
            global_embed = self.to_global_embed(global_embed)

        prepend_inputs = None 
        prepend_mask = None
        prepend_length = 0

        # 보통 이게 두개 다 들어오나? inference에서는 두개 다 None으로 찍힘.. 시간은 어디로 들어가는거지?
        # if prepend_cond is not None:
        #     # Project the prepend conditioning to the embedding dimension
        #     prepend_cond = self.to_prepend_embed(prepend_cond)
            
        #     prepend_inputs = prepend_cond
        #     if prepend_cond_mask is not None:
        #         prepend_mask = prepend_cond_mask
        # if input_concat_cond is not None:
        #     # Interpolate input_concat_cond to the same length as x
        #     if input_concat_cond.shape[2] != x.shape[2]:
        #         input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')

        #     x = torch.cat([x, input_concat_cond], dim=1)

        # Get the batch of timestep embeddings
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None])) # (b, embed_dim)

        # Timestep embedding is considered a global embedding. Add to the global conditioning if it exists
        global_embed = global_embed + timestep_embed if global_embed is not None else timestep_embed

        # Add the global_embed to the prepend inputs if there is no global conditioning support in the transformer
        if self.global_cond_type == "prepend": # true
            if prepend_inputs is None:
                # Prepend inputs are just the global embed, and the mask is all ones
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                # Prepend inputs are the prepend conditioning + the global embed
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)

            prepend_length = prepend_inputs.shape[1]
        # extra_args = {}
        # if self.global_cond_type == "adaLN":
        #     extra_args["global_cond"] = global_embed

        x = self.preprocess_conv(x) + x # 이게 머지
        x = rearrange(x, "b c t -> b t c")

        if self.patch_size > 1: # 1이다.
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)

        if self.transformer_type == "continuous_transformer":
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, context_mask=cross_attn_cond_mask, mask=mask, prepend_mask=prepend_mask, return_info=return_info, **kwargs)
        # elif self.transformer_type == "x-transformers":
        #     output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, context_mask=cross_attn_cond_mask, mask=mask, prepend_mask=prepend_mask, **extra_args, **kwargs)
        # elif self.transformer_type == "mm_transformer":
        #     output = self.transformer(x, context=cross_attn_cond, mask=mask, context_mask=cross_attn_cond_mask, **extra_args, **kwargs)

        output = rearrange(output, "b t c -> b c t")[:,:,prepend_length:]

        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)

        output = self.postprocess_conv(output) + output
        
        return output

    def forward(
        self, 
        x, 
        t,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        scale_phi=0.0,
        mask=None,
        return_info=False,
        **kwargs):
        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = cross_attn_cond_mask.bool()
            cross_attn_cond_mask = None # Temporarily disabling conditioning masks due to kernel issue for flash attention
        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.bool()

        # CFG dropout
        if cfg_dropout_prob > 0.0: # inference시 classifier free guidance가 적용되게 하기 위해서 보통 10%를 drop해서 학습한다.
            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                # full -> bernoulli -> to.bool
                dropout_mask = torch.bernoulli(torch.full((cross_attn_cond.shape[0], 1, 1), cfg_dropout_prob, device=cross_attn_cond.device)).to(torch.bool)
                # 모든 값이 cfg_dropout_prob인 ( , 1, 1) 크기의 텐서를 만들고, 그 확률대로 1을 갖거나 0을 갖는 텐서를 만든다음 True/False 값으로 바꾼다.
                cross_attn_cond = torch.where(dropout_mask, null_embed, cross_attn_cond)
                # condition, x, y | condition이 True인 위치에서는 x를, False인 위치에서는 y의 값을 갖는 텐서를 반환한다.
                # -> dropoout이 0.1 확률로 적용됨.
            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                dropout_mask = torch.bernoulli(torch.full((prepend_cond.shape[0], 1, 1), cfg_dropout_prob, device=prepend_cond.device)).to(torch.bool)
                prepend_cond = torch.where(dropout_mask, null_embed, prepend_cond)


        if cfg_scale != 1.0 and (cross_attn_cond is not None or prepend_cond is not None):
            # Classifier-free guidance
            # Concatenate conditioned and unconditioned inputs on the batch dimension            
            batch_inputs = torch.cat([x, x], dim=0) # uncond, cond 두개로 만들기 위해서 이렇게 두개의 배치로 만들어서 넣는다. 만약 각각 배치여도 다 두배씩 만들어야 하기 때문에.
            batch_timestep = torch.cat([t, t], dim=0)
            # if global_embed is not None:
            #     batch_global_cond = torch.cat([global_embed, global_embed], dim=0)
            # else:
            #     batch_global_cond = None
            # if input_concat_cond is not None:
            #     batch_input_concat_cond = torch.cat([input_concat_cond, input_concat_cond], dim=0)
            # else:
            #     batch_input_concat_cond = None

            batch_cond = None
            batch_cond_masks = None
            
            # Handle CFG for cross-attention conditioning
            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0) # 두개를 동시에 넣어서 쓰기 때문에,,
                if cross_attn_cond_mask is not None:
                    batch_cond_masks = torch.cat([cross_attn_cond_mask, cross_attn_cond_mask], dim=0)
               
            # batch_prepend_cond = None
            # batch_prepend_cond_mask = None
            # if prepend_cond is not None:
            #     null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
            #     batch_prepend_cond = torch.cat([prepend_cond, null_embed], dim=0)
            #     if prepend_cond_mask is not None:
            #         batch_prepend_cond_mask = torch.cat([prepend_cond_mask, prepend_cond_mask], dim=0)

            if mask is not None:
                batch_masks = torch.cat([mask, mask], dim=0)
            else:
                batch_masks = None
            
            batch_output = self._forward(
                batch_inputs, 
                batch_timestep, 
                cross_attn_cond=batch_cond, 
                cross_attn_cond_mask=batch_cond_masks, 
                mask = batch_masks, 
                # input_concat_cond=batch_input_concat_cond, 
                # global_embed = batch_global_cond,
                # prepend_cond = batch_prepend_cond,
                # prepend_cond_mask = batch_prepend_cond_mask,
                return_info = return_info,
                **kwargs)

            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)
            cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale

            # CFG Rescale
            if scale_phi != 0.0:
                cond_out_std = cond_output.std(dim=1, keepdim=True)
                out_cfg_std = cfg_output.std(dim=1, keepdim=True)
                output = scale_phi * (cfg_output * (cond_out_std/out_cfg_std)) + (1-scale_phi) * cfg_output
            else:
                output = cfg_output
            
            return output
            
        # else:
        #     return self._forward(
        #         x,
        #         t,
        #         cross_attn_cond=cross_attn_cond, 
        #         cross_attn_cond_mask=cross_attn_cond_mask, 
        #         input_concat_cond=input_concat_cond, 
        #         global_embed=global_embed, 
        #         prepend_cond=prepend_cond, 
        #         prepend_cond_mask=prepend_cond_mask,
        #         mask=mask,
        #         return_info=return_info,
        #         **kwargs
        #     )