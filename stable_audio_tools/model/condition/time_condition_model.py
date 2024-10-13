import torch.nn as nn
from torch import Tensor, einsum
import torch
from einops import rearrange
from typing import List, Union
from math import pi

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
