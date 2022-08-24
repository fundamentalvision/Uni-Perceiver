from uniperceiver.utils.registry import Registry
from torch import ModuleDict


ENCODER_REGISTRY = Registry("ENCODER")
ENCODER_REGISTRY.__doc__ = """
Registry for encoder
"""

def build_encoder(cfg):
    encoder = ENCODER_REGISTRY.get(cfg.MODEL.ENCODER)(cfg) if len(cfg.MODEL.ENCODER) > 0 else None
    return encoder

def build_unfused_encoders(cfg):
    from uniperceiver.config import  CfgNode
    encoder_dict = {}
    for config in cfg.ENCODERS:
        tmg_config = CfgNode(config)
        encoder = ENCODER_REGISTRY.get(
            tmg_config.TYPE)(tmg_config, cfg) if len(tmg_config.TYPE) > 0 else None
        encoder_dict[tmg_config.NAME] = encoder

    return encoder_dict


def add_encoder_config(cfg, tmp_cfg):
    if len(tmp_cfg.MODEL.ENCODER) > 0:
        ENCODER_REGISTRY.get(tmp_cfg.MODEL.ENCODER).add_config(cfg)