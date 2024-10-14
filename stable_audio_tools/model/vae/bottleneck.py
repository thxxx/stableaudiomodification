import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def create_bottleneck_from_config(bottleneck_config):
    bottleneck_type = bottleneck_config.get('type', None)

    assert bottleneck_type is not None, 'type must be specified in bottleneck config'

    if bottleneck_type == 'vae':
        from .bottleneck import VAEBottleneck
        bottleneck = VAEBottleneck()
    else:
        raise NotImplementedError(f'Unknown bottleneck type: {bottleneck_type}')
    
    requires_grad = bottleneck_config.get('requires_grad', True)
    if not requires_grad:
        for param in bottleneck.parameters():
            param.requires_grad = False

    return bottleneck


def vae_sample(mean, scale):
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return latents, kl


class Bottleneck(nn.Module):
    def __init__(self, is_discrete: bool = False):
        super().__init__()

        self.is_discrete = is_discrete

    def encode(self, x, return_info=False, **kwargs):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError

class VAEBottleneck(Bottleneck):
    def __init__(self):
        super().__init__(is_discrete=False)

    def encode(self, x, return_info=False, **kwargs):
        info = {}

        mean, scale = x.chunk(2, dim=1)

        x, kl = vae_sample(mean, scale)

        info["kl"] = kl

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x
