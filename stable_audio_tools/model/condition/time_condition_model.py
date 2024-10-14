import torch.nn as nn
from torch import Tensor
import torch
from einops import rearrange
from typing import List, Union
from math import pi
import typing as tp
from .Conditioners import Conditioner

class LearnedPositionalEmbedding(nn.Module):
    """ Used for continuous time """
    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1) # (b, b) 차원이 된다.
        fouriered = torch.cat((x, fouriered), dim=-1) # (b, b+1) 차원이 된다.
        return fouriered

def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
    )

# seconds start, total에 사용된다.
class NumberEmbedder(nn.Module):
    def __init__(
        self,
        features: int, # cond_dim=768
        dim: int = 256,
    ):
        super().__init__()
        self.features = features
        self.embedding = TimePositionalEmbedding(dim=dim, out_features=features)

    # 입력이 왜 list float이냐?
    def forward(self, x: Union[List[float], Tensor]) -> Tensor:
        # 우선 텐서가 아니면 텐서로 바꾼다.
        if not torch.is_tensor(x):
            device = next(self.embedding.parameters()).device
            x = torch.tensor(x, device=device)
        assert isinstance(x, Tensor)

        shape = x.shape
        x = rearrange(x, "... -> (...)") # flatten된다.
        embedding = self.embedding(x)
        x = embedding.view(*shape, self.features) # 다시 원래 shape으로. 마지막 features는 뭔지 잘..
        return x  # type: ignore



class NumberConditioner(Conditioner):
    '''
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    '''
    def __init__(self, 
                output_dim: int,
                min_val: float=0,
                max_val: float=1
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val

        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: tp.List[float], device=None) -> tp.Any:
            # Cast the inputs to floats
            floats = [float(x) for x in floats]
            floats = torch.tensor(floats).to(device)
            floats = floats.clamp(self.min_val, self.max_val)
            normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

            # Cast floats to same type as embedder
            embedder_dtype = next(self.embedder.parameters()).dtype
            normalized_floats = normalized_floats.to(embedder_dtype)

            float_embeds = self.embedder(normalized_floats).unsqueeze(1)
    
            return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]
