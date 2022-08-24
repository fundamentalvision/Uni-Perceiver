import torch
from torch import nn
from einops import rearrange

from uniperceiver.config import configurable

from ..layers.create_act import get_act_layer
from .build import EMBEDDING_REGISTRY

__all__ = ["PrefixPromptEmbedding"]

@EMBEDDING_REGISTRY.register()
class PrefixPromptEmbedding(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            cfg,
            dim: int,
            query_size: int,  # include <BOS>/<EOS>
            label_prompt,
            label_size,
            **kwargs):
        super(PrefixPromptEmbedding, self).__init__()

        self.cfg = cfg
        self.input_prompt = cfg.MODEL.PROMPT_EMBED.INPUT_PROMPT and cfg.MODEL.PROMPT_EMBED.PROMPT_LENGTH > 0
        self.target_prompt = cfg.MODEL.PROMPT_EMBED.TARGET_PROMPT and cfg.MODEL.PROMPT_EMBED.TARGET_PROMPT_LENGTH > 0
        self.label_prompt = label_prompt

        if self.input_prompt:
            self.embeddings = nn.Embedding(cfg.MODEL.PROMPT_EMBED.PROMPT_LENGTH, cfg.MODEL.ENCODER_DIM)
        if self.target_prompt:
            if self.label_prompt:
                self.target_embedding = nn.Parameter(
                    torch.zeros((cfg.MODEL.PROMPT_EMBED.LABEL_SIZE, 1, cfg.MODEL.ENCODER_DIM)))
            else:
                self.target_embedding = nn.Embedding(cfg.MODEL.PROMPT_EMBED.TARGET_PROMPT_LENGTH, cfg.MODEL.ENCODER_DIM)

        self.embeddings_act = kwargs.pop("embeddings_act", None)
        self.embeddings_norm = kwargs.pop("embeddings_norm", None)
        self.embeddings_dropout = kwargs.pop("embeddings_dropout", None)
        self.prompt_with_pos = kwargs.pop('prompt_with_pos', None)

    @classmethod
    def from_config(cls, cfg):
        kwargs = {
            "dim": cfg.MODEL.ENCODER_DIM,
            "query_size": cfg.MODEL.PROMPT_EMBED.PROMPT_LENGTH,
            "label_prompt": cfg.MODEL.PROMPT_EMBED.LABLE_PROMPT,
            "label_size": cfg.MODEL.PROMPT_EMBED.LABEL_SIZE,
            "target_prompt": cfg.MODEL.PROMPT_EMBED.TARGET_PROMPT,
            "num_layers": cfg.MODEL.BERT.NUM_HIDDEN_LAYERS,
        }

        activation_name = (cfg.MODEL.PROMPT_EMBED.ACTIVATION).lower()
        if activation_name != "none":
            activation = get_act_layer(activation_name)
            assert activation is not None

            act_kwargs = {}
            if activation_name in {"elu", "celu"}:
                act_kwargs["alpha"] = cfg.MODEL.PROMPT_EMBED.ELU_ALPHA
            embeddings_act = activation(**act_kwargs)
            kwargs['embeddings_act'] = embeddings_act

        if cfg.MODEL.PROMPT_EMBED.DROPOUT > 0:
            embeddings_dropout = nn.Dropout(cfg.MODEL.PROMPT_EMBED.DROPOUT)
            kwargs['embeddings_dropout'] = embeddings_dropout

        if cfg.MODEL.PROMPT_EMBED.USE_NORM:
            embeddings_norm = nn.LayerNorm(cfg.MODEL.PROMPT_EMBED.DIM)
            kwargs['embeddings_norm'] = embeddings_norm

        kwargs['prompt_with_pos'] = cfg.MODEL.PROMPT_EMBED.WITH_POS
        kwargs['cfg'] = cfg

        return kwargs

    def forward(self, data_list):
        bs = data_list[0]['data'].shape[0]
        prompt_embed = self._forward(bs,  data_type=data_list[0]['data_type'])

        if prompt_embed is None:
            return


        insert_data = {
            'data': prompt_embed,
            'invalid_mask': None,
            'modality': None,
            'data_type': data_list[0]['data_type'],

        }
        data_list.insert(0, insert_data)

        #TODO label prompt




    def _forward(self, bs, data_type:str = None):
        assert data_type in ['input', 'target']
        if data_type == 'input' and self.input_prompt:
            embeddings = self.embeddings.weight.unsqueeze(0).repeat(bs, 1, 1)
        elif data_type == 'target' and self.target_prompt:
            if  not self.label_prompt:
                embeddings = self.target_embedding.weight.unsqueeze(0).repeat(bs, 1, 1)
            elif  self.label_prompt:
                embeddings = self.target_embedding
            else:
                # target will not have prompt_embedding
                return None
        else:
            return None

        if self.embeddings_act is not None:
            embeddings = self.embeddings_act(embeddings)

        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(embeddings)

        if self.embeddings_dropout is not None:
            embeddings = self.embeddings_dropout(embeddings)

        return embeddings
