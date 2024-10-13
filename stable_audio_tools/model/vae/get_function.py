
from .bottleneck import create_bottleneck_from_config
from typing import Literal, Dict, Any
from .autoencoders import OobleckDecoder, OobleckEncoder, DACEncoderWrapper, DACDecoderWrapper, AudioAutoencoder

# AE factories
def create_encoder_from_config(encoder_config: Dict[str, Any]):
    encoder_type = encoder_config.get("type", None)

    if encoder_type == "oobleck":
        encoder = OobleckEncoder(
            **encoder_config["config"]
        )
    elif encoder_type == "dac":
        dac_config = encoder_config["config"]
        encoder = DACEncoderWrapper(**dac_config)
    else:
        raise ValueError(f"Unknown encoder type {encoder_type}")
    
    requires_grad = encoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder

def create_decoder_from_config(decoder_config: Dict[str, Any]):
    decoder_type = decoder_config.get("type", None)

    if decoder_type == "oobleck":
        decoder = OobleckDecoder(
            **decoder_config["config"]
        )
    elif decoder_type == "dac":
        dac_config = decoder_config["config"]
        decoder = DACDecoderWrapper(**dac_config)
    else:
        raise ValueError(f"Unknown decoder type {decoder_type}")
    
    requires_grad = decoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in decoder.parameters():
            param.requires_grad = False

    return decoder

def create_autoencoder_from_config(config: Dict[str, Any]):
    ae_config = config
    
    encoder = create_encoder_from_config(ae_config["encoder"])
    decoder = create_decoder_from_config(ae_config["decoder"])

    bottleneck = ae_config.get("bottleneck", None)

    latent_dim = ae_config.get("latent_dim", None)
    assert latent_dim is not None, "latent_dim must be specified in model config"
    downsampling_ratio = ae_config.get("downsampling_ratio", None)
    assert downsampling_ratio is not None, "downsampling_ratio must be specified in model config"
    io_channels = ae_config.get("io_channels", None)
    assert io_channels is not None, "io_channels must be specified in model config"
    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "sample_rate must be specified in model config"

    in_channels = ae_config.get("in_channels", None)
    out_channels = ae_config.get("out_channels", None)

    if bottleneck is not None:
        bottleneck = create_bottleneck_from_config(bottleneck)

    soft_clip = ae_config["decoder"].get("soft_clip", False)

    return AudioAutoencoder(
        encoder,
        decoder,
        io_channels=io_channels,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        sample_rate=sample_rate,
        bottleneck=bottleneck,
        pretransform=None,
        in_channels=in_channels,
        out_channels=out_channels,
        soft_clip=soft_clip
    )
