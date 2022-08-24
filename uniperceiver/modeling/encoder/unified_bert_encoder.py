import torch
from torch import nn

from uniperceiver.config import configurable
from ..layers.transformer_encoder_layer import TransformerEncoderLayer
from ..layers.transformer_encoder_moe_layer import MoETransformerEncoderLayer
from .build import ENCODER_REGISTRY
import uniperceiver.utils.comm as comm



__all__ = ["UnifiedBertEncoder"]

def _construct_attention_masks( data, sample_info, task_info):
    mask_type = torch.bool
    device = data.device

    attn_mask = None
    if isinstance(sample_info, list):
        sample_info = sample_info[0]
    if task_info['task_type'] in ['image_caption', 'video_caption'] and sample_info.get('text_spe_cat', False):

        # the extra 1 length for spe token
        spe_length, img_length, text_total_length = sample_info['data_length']
        text_length = text_total_length//2

        attn_mask = torch.ones((spe_length + img_length + text_total_length,
                                spe_length + img_length + text_total_length), dtype=mask_type, device=device)

        attn_mask[:spe_length + img_length + text_total_length, :spe_length+img_length] = False
        attn_mask[spe_length + img_length:spe_length + img_length + text_length, spe_length + img_length:spe_length + img_length + text_length] =  torch.ones(
                (text_length, text_length),  dtype=mask_type, device=device).triu_(diagonal=1)
        attn_mask[spe_length + img_length + text_length:, spe_length + img_length:spe_length + img_length + text_length] =  torch.ones(
                (text_length, text_length),
                dtype=mask_type,
                device=device).triu_(diagonal=0)
        attn_mask[spe_length + img_length + text_length:,
                  spe_length + img_length + text_length:] = ~torch.ones(
                      (text_length), dtype=mask_type,
                      device=device).diag()

    return attn_mask


@ENCODER_REGISTRY.register()
class UnifiedBertEncoder(nn.Module):
    @configurable
    def __init__(self, *, num_hidden_layers: int, bert_layers,
                 skip_target_encode, word_balance_losses,
                 bookswiki_word_alone, cfg):
        super(UnifiedBertEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layers = bert_layers
        self.skip_target_encode = skip_target_encode
        self.word_balance_losses = word_balance_losses
        self.bookswiki_word_alone = bookswiki_word_alone
        self.cfg = cfg



    @classmethod
    def from_config(cls, cfg):
        if cfg.MODEL.BERT.DROP_PATH_PROB_FIXED:
            dpr = [cfg.MODEL.BERT.DROP_PATH_PROB for _ in range(cfg.MODEL.BERT.NUM_HIDDEN_LAYERS)]
        else:
            dpr = [x.item() for x in torch.linspace(0, cfg.MODEL.BERT.DROP_PATH_PROB, cfg.MODEL.BERT.NUM_HIDDEN_LAYERS)]

        layers = []
        for layer_idx in range(cfg.MODEL.BERT.NUM_HIDDEN_LAYERS):
            if not cfg.MOE.MOE:
                layers.append(
                    TransformerEncoderLayer(
                        d_model=cfg.MODEL.BERT.HIDDEN_SIZE,
                        nhead=cfg.MODEL.BERT.NUM_ATTENTION_HEADS,
                        dim_feedforward=cfg.MODEL.BERT.INTERMEDIATE_SIZE,
                        dropout=cfg.MODEL.BERT.HIDDEN_DROPOUT_PROB,
                        drop_path_ratio=dpr[layer_idx],
                        activation=cfg.MODEL.BERT.HIDDEN_ACT,
                        layer_scale=cfg.MODEL.LAYER_SCALE,
                        ls_init_values=cfg.MODEL.LAYER_SCALE_INIT,
                        batch_first=True,
                        norm_first=True,
                        cfg = cfg,
                    ))
            else:
                attention_moe = False
                ffn_moe = False

                moe_layer_start_idx = cfg.MOE.MOE_LAYER_START_IDX
                moe_layer_end_idx = cfg.MOE.MOE_LAYER_END_IDX

                if cfg.MOE.MOE and cfg.MOE.MOE_EXPERT_LOCATION == 'odd':
                    if layer_idx % 2 == 0 and layer_idx >= moe_layer_start_idx and layer_idx < moe_layer_end_idx:
                        moe_layers = cfg.MOE.MOE_EXPERT_TYPE.split(',')
                        attention_moe = "SA" in moe_layers
                        ffn_moe = 'FFN' in moe_layers

                elif cfg.MOE.MOE and cfg.MOE.MOE_EXPERT_LOCATION == 'four':
                    if layer_idx % 4 == 0 and layer_idx >= moe_layer_start_idx and layer_idx < moe_layer_end_idx:
                        moe_layers = cfg.MOE.MOE_EXPERT_TYPE.split(',')
                        attention_moe = "SA" in moe_layers
                        ffn_moe = 'FFN' in moe_layers

                elif cfg.MOE.MOE and cfg.MOE.MOE_EXPERT_LOCATION == 'all':
                    if layer_idx >= moe_layer_start_idx and layer_idx < moe_layer_end_idx:
                        moe_layers = cfg.MOE.MOE_EXPERT_TYPE.split(',')
                        attention_moe = "SA" in moe_layers
                        ffn_moe = 'FFN' in moe_layers
                elif cfg.MOE.MOE and cfg.MOE.MOE_EXPERT_LOCATION == 'none':
                    attention_moe = None
                    ffn_moe = None


                elif cfg.MOE.MOE:
                    raise NotImplementedError('cfg.MOE.MOE_EXPERT_LOCATION')

                layers.append(
                    MoETransformerEncoderLayer(
                        d_model=cfg.MODEL.BERT.HIDDEN_SIZE,
                        nhead=cfg.MODEL.BERT.NUM_ATTENTION_HEADS,
                        dim_feedforward=cfg.MODEL.BERT.INTERMEDIATE_SIZE,
                        dropout=cfg.MODEL.BERT.HIDDEN_DROPOUT_PROB,
                        drop_path_ratio=dpr[layer_idx],
                        activation=cfg.MODEL.BERT.HIDDEN_ACT,
                        layer_scale=cfg.MODEL.LAYER_SCALE,
                        ls_init_values=cfg.MODEL.LAYER_SCALE_INIT,
                        batch_first=False,
                        norm_first=True,
                        cfg = cfg,
                        ffn_moe=ffn_moe,
                        attn_moe=attention_moe,
                    ))



        bert_layers = nn.ModuleList(
            layers
        )
        return {
            "num_hidden_layers": cfg.MODEL.BERT.NUM_HIDDEN_LAYERS,
            "skip_target_encode": cfg.MODEL.BERT.SKIP_TARGET_ENCODE,
            "bert_layers": bert_layers,
            "word_balance_losses": cfg.SOLVER.WORD_BALANCE_LOSSESS,
            "bookswiki_word_alone": cfg.MODEL.BW_WORD_ALONE,
            "cfg": cfg
        }

    @classmethod
    def add_config(cls, cfg):
        pass


    def forward(self, data, invalid_mask, sample_info, task_info, history_states=None, return_all=False, **kwargs):

        attn_mask = _construct_attention_masks(data, sample_info, task_info)
        kwargs.update({'sample_info': sample_info})
        data_type = kwargs.get('data_type', 'input')
        if data_type == 'target' and self.skip_target_encode:
            # used for debugging with single gpu sometimes
            return data 
        if return_all:
            data_all = [data]
        for l, layer_module in enumerate(self.layers):

            if history_states is None:
                data = layer_module(src=data, src_mask=attn_mask, src_key_padding_mask=invalid_mask, task_info=task_info, **kwargs)
            else:
                data = layer_module(src=data, src_mask=attn_mask, src_key_padding_mask=invalid_mask, history_states=history_states[l], task_info=task_info, **kwargs)

            if return_all:
                data_all.append(data)

        return data if not return_all else data_all
