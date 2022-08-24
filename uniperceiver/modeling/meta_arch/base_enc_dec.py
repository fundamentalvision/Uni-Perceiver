import copy
import numpy as np
import weakref
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod

from uniperceiver.config import configurable
from uniperceiver.config import CfgNode as CN
from uniperceiver.functional import pad_tensor, dict_to_cuda, flat_list_of_lists
from ..embedding import build_embeddings
from ..encoder import build_encoder, add_encoder_config
# from ..decoder import build_decoder, add_decoder_config
from ..predictor import build_predictor, add_predictor_config
from ..decode_strategy import build_beam_searcher, build_greedy_decoder

class BaseEncoderDecoder(nn.Module, metaclass=ABCMeta):
    @configurable
    def __init__(
        self,
        *,
        vocab_size,
        max_seq_len,
        token_embed,
        fused_encoder,
        decoder,
        greedy_decoder,
        beam_searcher,
        **kwargs,
    ):
        super(BaseEncoderDecoder, self).__init__()
        self.fused_encoder = fused_encoder
        self.decoder = decoder

        self.token_embed = token_embed
        self.greedy_decoder = greedy_decoder
        self.beam_searcher = beam_searcher
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


    @classmethod
    def add_config(cls, cfg, tmp_cfg):
        add_encoder_config(cfg, tmp_cfg)
        add_predictor_config(cfg, tmp_cfg)

    def forward(self, batched_inputs, use_beam_search=None, output_sents=False):
        if use_beam_search is None:
            return self._forward(batched_inputs)
        # elif use_beam_search == False or self.beam_searcher.beam_size == 1:
        elif use_beam_search == False:
            return self.greedy_decode(batched_inputs, output_sents)
        else:
            return self.decode_beam_search(batched_inputs, output_sents)

    @abstractmethod
    def _forward(self, batched_inputs):
        pass

    def bind_or_init_weights(self):
        pass


    def greedy_decode(self, batched_inputs, output_sents=False):
        return self.greedy_decoder(
            batched_inputs,
            output_sents,
            model=weakref.proxy(self)
        )

    def decode_beam_search(self, batched_inputs, output_sents=False):
        return self.beam_searcher(
            batched_inputs,
            output_sents,
            model=weakref.proxy(self)
        )
