import torch
from torch import nn

from uniperceiver.config import configurable
from ..layers.create_act import get_act_layer
from .build import EMBEDDING_REGISTRY
from .position_embedding import build_position_encoding
# from uniperceiver.modeling.layers import LayerNorm
from uniperceiver.utils import comm
import copy
from uniperceiver.modeling.layers import FP16LayerNorm


__all__ = ["TokenBaseEmbedding"]

@EMBEDDING_REGISTRY.register()
class TokenBaseEmbedding(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        dim: int,
        vocab_size: int, # include <BOS>/<EOS>
        **kwargs
    ):
        super(TokenBaseEmbedding, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, dim)
        self.embeddings_act = kwargs.pop("embeddings_act", None)
        self.embeddings_norm = kwargs.pop("embeddings_norm", None)
        self.embeddings_dropout = kwargs.pop("embeddings_dropout", None)
        self.embeddings_pos = kwargs.pop("embeddings_pos", None)
        self.embeddings_token_type = kwargs.pop('embeddings_token_type', None)
        self.embeddings_token_seg = kwargs.pop('embeddings_token_seg', None)
        self.bw_own_embed = kwargs.pop('bw_own_embed', False)
        self.pos_before = kwargs.pop('pos_before', True)
        self.cfg = kwargs.pop('cfg', None)

        if self.bw_own_embed:
            # only for debugging
            self.bw_embeddings = copy.deepcopy(self.embeddings)
            self.bw_embeddings_norm = copy.deepcopy(self.embeddings_norm)
            self.bw_embeddings_pos = copy.deepcopy(self.embeddings_pos)
            self.bw_embeddings_token_type = copy.deepcopy(self.embeddings_token_type)
        self.s_token_bias = None

    @classmethod
    def from_config(cls, cfg):
        kwargs = {
            "dim": cfg.MODEL.TOKEN_EMBED.DIM,
            "vocab_size": cfg.MODEL.VOCAB_SIZE
        }

        activation_name = (cfg.MODEL.TOKEN_EMBED.ACTIVATION).lower()
        if activation_name != "none":
            activation = get_act_layer(activation_name)
            assert activation is not None

            act_kwargs = {}
            if activation_name in { "elu", "celu" }:
                act_kwargs["alpha"] = cfg.MODEL.TOKEN_EMBED.ELU_ALPHA
            embeddings_act = activation(**act_kwargs)
            kwargs['embeddings_act'] = embeddings_act

        if cfg.MODEL.TOKEN_EMBED.DROPOUT > 0:
            embeddings_dropout = nn.Dropout(cfg.MODEL.TOKEN_EMBED.DROPOUT)
            kwargs['embeddings_dropout'] = embeddings_dropout

        if cfg.MODEL.TOKEN_EMBED.USE_NORM:
            if cfg.SOLVER.FORCE_LN_FP16:
                embeddings_norm = FP16LayerNorm(cfg.MODEL.TOKEN_EMBED.DIM)
            else:
                embeddings_norm = nn.LayerNorm(cfg.MODEL.TOKEN_EMBED.DIM)
            kwargs['embeddings_norm'] = embeddings_norm

        if (cfg.MODEL.TOKEN_EMBED.POSITION).lower() != 'none':
            embeddings_pos = build_position_encoding(cfg,
                cfg.MODEL.TOKEN_EMBED.DIM, cfg.MODEL.TOKEN_EMBED.POSITION_MAX_LEN)
            kwargs['embeddings_pos'] = embeddings_pos

        if cfg.MODEL.TOKEN_EMBED.TYPE_VOCAB_SIZE > 0:
            embeddings_token_type = nn.Embedding(
                cfg.MODEL.TOKEN_EMBED.TYPE_VOCAB_SIZE, cfg.MODEL.TOKEN_EMBED.DIM)
            kwargs['embeddings_token_type'] = embeddings_token_type

        if cfg.MODEL.TOKEN_EMBED.TYPE_SEG_SIZE > 0:
            embeddings_token_seg = nn.Embedding(
                cfg.MODEL.TOKEN_EMBED.TYPE_SEG_SIZE, cfg.MODEL.TOKEN_EMBED.DIM)
            kwargs['embeddings_token_seg'] = embeddings_token_seg

        # for debug
        kwargs['bw_own_embed'] = cfg.MODEL.BW_OWD_EMBED
        kwargs['pos_before'] = cfg.MODEL.POS_BEFORE
        kwargs['cfg'] = cfg
        return kwargs

    def get_time_step(self, data, sample_info, task_info=None):
        """
        data: Bs, L
        task_info: {
            task_type: str
        }
        """
        # TODO: the position embedding for caption text should be handled in a different way.  0,1, n/2,0,1, n/2,
        if task_info is None:
            task_type = ''
        else:
            task_type = task_info.get('task_type', None)
        time_step = None
        if isinstance(sample_info, list):
            sample_info = sample_info[0]
        if task_type in ['image_caption', 'video_caption'] and sample_info.get('text_spe_cat', False):
            text_length = data.shape[1]
            time_step = torch.cat([
                torch.arange(text_length // 2,
                             dtype=torch.long,
                             device=data.device) for _ in range(2)
            ])
        elif task_type == 'VQA' and sample_info.get('text_spe_cat', False):
            text_length = data.shape[1]
            time_step = torch.cat([
                torch.arange(text_length - 1,
                             dtype=torch.long,
                             device=data.device),
                torch.arange(1, dtype=torch.long, device=data.device)
            ])


        return time_step

    def forward(self, data, sample_info={}, task_info={}, **kwargs):


        time_step = kwargs.pop('time_step', None)
        if time_step is None:
            time_step = self.get_time_step(data, sample_info, task_info)

        if kwargs.pop("prompt_with_pos", False):
            raise NotImplementedError
        else:
            start_time = 0

        type_embed = kwargs.get('type_embed', True)
        pos_emb = kwargs.get('pos_embed', True)

        data = self._forward(data,
                            type_embed=type_embed,
                            pos_emb=pos_emb,
                            token_seg_ids=None,
                            time_step=time_step,
                            start_time=start_time)

        return data



    def set_s_token_bias(self, s_token_bias):
        self.s_token_bias = s_token_bias

    def _forward(self, input_ids, type_embed=True, token_seg_ids=None, time_step=None, pos_emb=True, start_time=0, ):

        embeddings = self.embeddings(input_ids)
        if self.cfg.SOLVER.FORCE_EMBED_FP16:
            embeddings = embeddings.half()

        if self.s_token_bias is not None:
            # learnable
            embeddings[input_ids == 49410] = embeddings[input_ids == 49410] + self.s_token_bias

        if self.embeddings_pos is not None and pos_emb and self.pos_before:
            pos_inputs = input_ids if time_step is None else time_step
            position_embeddings = self.embeddings_pos(pos_inputs, start_time=start_time)
            embeddings = embeddings + position_embeddings.to(embeddings.dtype)

        if self.embeddings_token_type is not None and type_embed:

            embeddings_token_type = self.embeddings_token_type.weight[0].unsqueeze(0).unsqueeze(1)
            embeddings = embeddings + embeddings_token_type.to(embeddings.dtype)

        if (self.embeddings_token_seg is not None) and (token_seg_ids is not None):
            embeddings_token_seg = self.embeddings_token_seg(token_seg_ids)
            embeddings = embeddings + embeddings_token_seg

        if self.embeddings_act is not None:
            embeddings = self.embeddings_act(embeddings)

        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(embeddings)

        if self.embeddings_pos is not None and pos_emb and not self.pos_before:
            pos_inputs = input_ids if time_step is None else time_step
            position_embeddings = self.embeddings_pos(pos_inputs, start_time=start_time)
            embeddings = embeddings + position_embeddings.to(embeddings.dtype)

        if self.embeddings_dropout is not None:
            embeddings = self.embeddings_dropout(embeddings)

        return embeddings
