'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import torch
import copy
from .gate import one_hot_with_dtype
from uniperceiver.utils import comm

import torch.nn.functional as F

from torch.cuda.amp import autocast


class FusedExperts(torch.nn.Module):
    def __init__(self, expert, cfg,  num_local_experts=1):
        super(FusedExperts, self).__init__()
        self.cfg = cfg

        self.deepspeed_experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        self.bias_merge = self.deepspeed_experts[0].bias is not None


    def top1_expert_forward(self, x, indice, gate, mode=None, **kwargs):
        assert  mode is None, "unified qkv inference is not supported for top1"
        if indice.size(0)== 1:
            #unimodal
            x = self.deepspeed_experts[indice[0]](x) * gate[0].to(x)
        elif indice.size(0) == 2:
            # mulmodal
            data1_length = kwargs['sample_info']['data_cum_length'][1]
            x = torch.cat([
                self.deepspeed_experts[indice[0]](x[:, :data1_length, :]) * gate[0].to(x),
                self.deepspeed_experts[indice[1]](x[:, data1_length:, :]) * gate[1].to(x)
            ],
                          dim=1)

        else:
            raise NotImplementedError('only support one or two modality')
        return x

    def mergelayer(self, x,  index1, index2, gate1, gate2, mode=None):
        
        if not self.cfg.SOLVER.FORCE_EXPERT_ADDING_FP16:
            if mode == 'q':
                # hidden_states
                _start = 0
                _end = self.deepspeed_experts[index1].weight.shape[0] // 3
                return F.linear(
                    x,
                    self.deepspeed_experts[index1].weight[_start:_end, :] * gate1 +
                    self.deepspeed_experts[index2].weight[_start:_end, :] * gate2,
                    bias=self.deepspeed_experts[index1].bias[_start:_end] * gate1 +
                    self.deepspeed_experts[index2].bias[_start:_end] * gate2
                    if self.bias_merge else None,
                )

            elif mode == 'kv':
                # history_states
                _start =  self.deepspeed_experts[index1].weight.shape[0] // 3

                return F.linear(
                    x,
                    self.deepspeed_experts[index1].weight[_start:, :] * gate1 +
                    self.deepspeed_experts[index2].weight[_start:, :] * gate2,
                    bias=self.deepspeed_experts[index1].bias[_start:] * gate1 +
                    self.deepspeed_experts[index2].bias[_start:] * gate2
                    if self.bias_merge else None,
                )

            else:

                return F.linear(
                    x,
                    self.deepspeed_experts[index1].weight * gate1 +
                    self.deepspeed_experts[index2].weight * gate2,
                    bias=self.deepspeed_experts[index1].bias * gate1 +
                    self.deepspeed_experts[index2].bias * gate2 if self.bias_merge else None,
                )
        else:
            if mode == 'q':
                # hidden_states
                _start = 0
                _end = self.deepspeed_experts[index1].weight.shape[0] // 3
                return F.linear(
                    x,
                    self.deepspeed_experts[index1].weight[_start:_end, :].half() * gate1 +
                    self.deepspeed_experts[index2].weight[_start:_end, :].half() * gate2,
                    bias=self.deepspeed_experts[index1].bias[_start:_end].half() * gate1 +
                    self.deepspeed_experts[index2].bias[_start:_end].half() * gate2 if self.bias_merge else None,
                )

            elif mode == 'kv':
                # history_states
                _start = self.deepspeed_experts[index1].weight.shape[0] // 3

                return F.linear(
                    x,
                    self.deepspeed_experts[index1].weight[_start:, :].half() * gate1 +
                    self.deepspeed_experts[index2].weight[_start:, :].half() * gate2,
                    bias=self.deepspeed_experts[index1].bias[_start:].half() * gate1 +
                    self.deepspeed_experts[index2].bias[_start:].half() * gate2 if self.bias_merge else None,
                )

            else:

                return F.linear(
                    x,
                    self.deepspeed_experts[index1].weight.half() * gate1 + self.deepspeed_experts[index2].weight.half() * gate2,
                    bias=self.deepspeed_experts[index1].bias.half() * gate1 +
                    self.deepspeed_experts[index2].bias.half() * gate2 if self.bias_merge else None,
                )
        

    def top2_expert_forward(self, x, indices, gates, mode=None, **kwargs):

        # caption eval mode
        if comm._CAPTION_GEN_MODE and x.shape[1] == 1:
            #
            return self.mergelayer(x,
                                   indices[0][1], indices[1][1],
                                   gates[0][1], gates[1][1], mode=mode)

        # unimodal
        if indices[0].size(0) == 1:
            x = self.mergelayer(x, indices[0][0], indices[1][0], gates[0][0], gates[1][0], mode=mode)
        elif indices[0].size(0) == 2:
            data1_length = kwargs['sample_info']['data_cum_length'][1]
            if mode == 'kv' and kwargs['sample_info'].get('pe_length', 0) > 0:
                # may have prompt embedding for kv embedding
                data1_length += kwargs['sample_info'].get('pe_length', 0)
            x = torch.cat([
                self.mergelayer(x[:, :data1_length, :], indices[0][0], indices[1][0], gates[0][0], gates[1][0], mode=mode),
                self.mergelayer(x[:, data1_length:, :], indices[0][1], indices[1][1], gates[0][1], gates[1][1], mode=mode)
            ],
                          dim=1)

        else:
            raise NotImplementedError('only support one or two modality')
        return x

    def forward(self, hidden_states, top_indices=None, gates=None, **kwargs):

        # top1
        if len(top_indices) == 1:
            out = self.top1_expert_forward(hidden_states, top_indices[0], gates[0], **kwargs)

        # top2
        elif len(top_indices) == 2:
            out = self.top2_expert_forward(hidden_states, top_indices, gates, **kwargs)

        else:
            raise NotImplementedError("only support top1 and top2 ")



        assert out.shape[1] == hidden_states.shape[1]

        return out
