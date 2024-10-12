import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .utils import LayerNorm
from .Attn import Attention

class GLU(nn.Module):
    """
    Gated Linear Unit. 각 입력을 어느정도로 통과시킬지 결정.
    """
    def __init__(
        self,
        dim_in,
        dim_out,
        activation,
        use_conv = False,
        conv_kernel_size = 3,
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2) if not use_conv else nn.Conv1d(dim_in, dim_out * 2, conv_kernel_size, padding = (conv_kernel_size // 2))
        self.use_conv = use_conv

    def forward(self, x):
        if self.use_conv:
            x = rearrange(x, 'b n d -> b d n')
            x = self.proj(x)
            x = rearrange(x, 'b d n -> b n d')
        else:
            x = self.proj(x)

        x, gate = x.chunk(2, dim = -1)
        return x * self.act(gate)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        no_bias = False,
        zero_init_output = True,
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        # Default to SwiGLU
        activation = nn.SiLU()

        dim_out = dim if dim_out is None else dim_out # same as input you know

        self.linear_in = GLU(dim, inner_dim, activation)
        self.linear_out = nn.Linear(inner_dim, dim_out, bias = not no_bias)

        # init last linear layer to 0
        if zero_init_output:
            nn.init.zeros_(self.linear_out.weight)
            if not no_bias:
                nn.init.zeros_(self.linear_out.bias)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.linear_out(x)
        return x

class ConformerModule(nn.Module):
    def __init__(
        self,
        dim,
        norm_kwargs = {},
    ):     
        super().__init__()

        self.dim = dim
        
        self.in_norm = LayerNorm(dim, **norm_kwargs)
        self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.glu = GLU(dim, dim, nn.SiLU())
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=17, groups=dim, padding=8, bias=False)
        self.mid_norm = LayerNorm(dim, **norm_kwargs) # This is a batch norm in the original but I don't like batch norm
        self.swish = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.in_norm(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.glu(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.depthwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.mid_norm(x)
        x = self.swish(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv_2(x)
        x = rearrange(x, 'b d n -> b n d')

        return x

class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_heads = 64,
            cross_attend = False,
            dim_context = None,
            global_cond_dim = None,
            zero_init_branch_outputs = True,
            remove_norms = False,
            attn_kwargs = {},
            ff_kwargs = {},
            norm_kwargs = {}
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.cross_attend = cross_attend
        self.dim_context = dim_context

        self.pre_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
        self.self_attn = Attention(
            dim,
            dim_heads = dim_heads,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs
        )

        if cross_attend:
            self.cross_attend_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
            self.cross_attn = Attention(
                dim,
                dim_heads = dim_heads,
                dim_context=dim_context,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs
            )
        
        self.ff_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)

        # self.global_cond_dim = global_cond_dim
        # if global_cond_dim is not None:
        #     self.to_scale_shift_gate = nn.Sequential(
        #         nn.SiLU(),
        #         nn.Linear(global_cond_dim, dim * 6, bias=False)
        #     )
        #     nn.init.zeros_(self.to_scale_shift_gate[1].weight)

    def forward(
        self,
        x,
        context = None,
        global_cond=None,
        mask = None,
        context_mask = None,
        rotary_pos_emb = None
    ):
        if self.global_cond_dim is not None and self.global_cond_dim > 0 and global_cond is not None:
            # scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = self.to_scale_shift_gate(global_cond).unsqueeze(1).chunk(6, dim = -1)

            # # self-attention with adaLN
            # residual = x
            # x = self.pre_norm(x)
            # x = x * (1 + scale_self) + shift_self
            # x = self.self_attn(x, mask = mask, rotary_pos_emb = rotary_pos_emb)
            # x = x * torch.sigmoid(1 - gate_self)
            # x = x + residual

            # # cross-attention
            # if context is not None:
            #     x = x + self.cross_attn(self.cross_attend_norm(x), context = context, context_mask = context_mask)

            # # feedforward with adaLN
            # residual = x
            # x = self.ff_norm(x)
            # x = x * (1 + scale_ff) + shift_ff
            # x = self.ff(x)
            # x = x * torch.sigmoid(1 - gate_ff)
            # x = x + residual
            pass

        else:
            x = x + self.self_attn(self.pre_norm(x), mask = mask, rotary_pos_emb = rotary_pos_emb)
            if context is not None:
                x = x + self.cross_attn(self.cross_attend_norm(x), context = context, context_mask = context_mask)
            x = x + self.ff(self.ff_norm(x))

        return x