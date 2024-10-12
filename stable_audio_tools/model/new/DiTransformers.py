from torch import nn
import torch
from .TransformerBlock import TransformerBlock
from .utils import checkpoint
from .positional_encoding import RotaryEmbedding

class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in = None,
        dim_out = None,
        dim_heads = 64,
        cross_attend=False,
        cond_token_dim=None,
        global_cond_dim=None,
        rotary_pos_emb=True,
        zero_init_branch_outputs=True,
        conformer=False,
        use_sinusoidal_emb=False, # 
        use_abs_pos_emb=False, # 둘다 false야? 그렇네..
        abs_pos_emb_max_length=10000,
        **kwargs
        ):

        super().__init__()

        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32)) if rotary_pos_emb is not None else None

        for i in range(depth):
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads = dim_heads,
                    cross_attend = cross_attend,
                    dim_context = cond_token_dim,
                    global_cond_dim = global_cond_dim,
                    zero_init_branch_outputs = zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    **kwargs
                )
            )
        
    def forward(
        self,
        x,
        mask = None,
        prepend_embeds = None,
        prepend_mask = None,
        global_cond = None,
        **kwargs
    ):
        batch, seq, device = x.shape[0], x.shape[1], x.device
        x = self.project_in(x)

        if prepend_embeds is not None:
            bs, prepend_length, prepend_dim = prepend_embeds.shape
            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'
            x = torch.cat((prepend_embeds, x), dim = -2) # 여기서 드디어 prepend.
            if prepend_mask is not None or mask is not None:
                mask = mask if mask is not None else torch.ones((batch, seq), device = device, dtype = torch.bool)
                prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_length), device = device, dtype = torch.bool)
                mask = torch.cat((prepend_mask, mask), dim = -1)

        # Attention layers 
        rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1]) if self.rotary_pos_emb is not None else None

        if self.use_sinusoidal_emb or self.use_abs_pos_emb:
            x = x + self.pos_emb(x)

        # Iterate over the transformer layers
        for layer in self.layers:
            # global_cond는 현재 inference 세팅에서는 없다!
            # x = layer(x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs) # 을 해도 되는데, 아래가 효율적임.
            x = checkpoint(layer, x, rotary_pos_emb=rotary_pos_emb, global_cond=global_cond, **kwargs)

        x = self.project_out(x)

        return x
