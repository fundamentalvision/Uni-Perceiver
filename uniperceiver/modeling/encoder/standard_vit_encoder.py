import torch
from torch import nn

from uniperceiver.config import configurable

from ..layers.transformer_encoder_layer import TransformerEncoderLayer
from .build import ENCODER_REGISTRY

import uniperceiver.utils.comm as comm

__all__ = ["StandardViT", "TextEncoder", "VisualEncoder"]



@ENCODER_REGISTRY.register()
class StandardViT(nn.Module):
    @configurable
    def __init__(self, *, num_hidden_layers: int, bert_layers, cfg):
        super(StandardViT, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layers = bert_layers
        self.cfg = cfg
        self.name = cfg.NAME

    @classmethod
    def from_config(cls, cfg, global_cfg):
        if cfg.DROP_PATH_PROB_FIXED:
            dpr = [cfg.DROP_PATH_PROB for _ in range(cfg.NUM_HIDDEN_LAYERS)]
        else:
            dpr = [x.item() for x in torch.linspace(0, cfg.DROP_PATH_PROB, cfg.NUM_HIDDEN_LAYERS)]

        layers = []
        for i in range(cfg.NUM_HIDDEN_LAYERS):
            layers.append(
                TransformerEncoderLayer(
                    d_model=cfg.HIDDEN_SIZE,
                    nhead=cfg.NUM_ATTENTION_HEADS,
                    dim_feedforward=cfg.INTERMEDIATE_SIZE,
                    dropout=cfg.HIDDEN_DROPOUT_PROB,
                    drop_path_ratio=dpr[i],
                    activation=cfg.HIDDEN_ACT,
                    layer_scale=global_cfg.MODEL.LAYER_SCALE,
                    ls_init_values=global_cfg.MODEL.LAYER_SCALE_INIT,
                    batch_first=True,
                    norm_first=True,
                    cfg=cfg,
                ))

        bert_layers = nn.ModuleList(
            layers
        )
        return {
            "num_hidden_layers": cfg.NUM_HIDDEN_LAYERS,
            "bert_layers": bert_layers,
            "cfg": cfg
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def _forward(self, x, attn_mask=None, key_padding_masks=None, history_states=None, *kwargs):

        for l, layer_module in enumerate(self.layers):
            x = layer_module(
                src=x, src_mask=attn_mask, src_key_padding_mask=key_padding_masks
            )

        return x


    def forward(self, batched_inputs, return_all=False):

        raise NotImplementedError

@ENCODER_REGISTRY.register()
class  VisualEncoder(StandardViT):

    @staticmethod
    def _construct_attention_masks( data, sample_info, task_info):

        return None

    def forward(self, data, invalid_mask, sample_info, task_info, **kwargs):
        #TODO: prepare attn mask for each task type
        # used for visual encoder
        attn_mask = self._construct_attention_masks(data, sample_info, task_info)
        history_states = kwargs.pop('history_states', None)
        out = self._forward(data,
                            attn_mask,
                            invalid_mask,
                            history_states=history_states,
                            **kwargs,
                            )

        return out


@ENCODER_REGISTRY.register()
class  TextEncoder(StandardViT):

    @staticmethod
    def _construct_attention_masks( data, sample_info, task_info):
        mask_type = torch.bool
        device = data.device

        attn_mask = None
        if isinstance(sample_info, list):
            sample_info = sample_info[0]
        if task_info['task_type'] in ['image_caption', 'video_caption'] and sample_info.get('text_spe_cat', False):
            total_length = data.shape[1]
            attn_mask  = torch.ones((total_length, total_length), dtype=mask_type, device=device)
            attn_mask[:total_length // 2, :total_length // 2] = torch.ones(
                (total_length // 2, total_length // 2),  dtype=mask_type, device=device).triu_(diagonal=1)
            attn_mask[total_length // 2:, : total_length // 2] = torch.ones(
                (total_length // 2, total_length // 2),
                dtype=mask_type,
                device=device).triu_(diagonal=0)
            attn_mask[total_length // 2:, total_length // 2:] = ~torch.ones(
                (total_length // 2),
                dtype=mask_type,
                device=device).diag()

        return  attn_mask

    def forward(self, data, invalid_mask, sample_info, task_info, **kwargs):
        #TODO: prepare attn mask for each task type
        # used for text encoder
        attn_mask = self._construct_attention_masks(data, sample_info, task_info)
        history_states = kwargs.pop('history_states', None)
        out = self._forward(data,
                            attn_mask,
                            invalid_mask,
                            history_states=history_states,
                            **kwargs)

        return out
