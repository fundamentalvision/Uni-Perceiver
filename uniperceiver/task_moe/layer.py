'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import torch.nn.init as init
import torch
from torch import nn
import torch.distributed as dist




from .gate import TopKGate
import copy
import typing

from .experts import FusedExperts as Experts


class TaskMoE(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 expert,
                 num_experts=1,
                 k=1,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 noisy_gate_policy: typing.Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts=True,
                 use_tutel: bool = False,
                 cfg=None):
        """Initialize an MoE layer.

        Arguments:
            hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.

            expert (torch.nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).

            num_experts (int, optional): default=1, the total number of experts per layer.

            k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.

            capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.

            eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.

            min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.

            noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.

            drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to infinite capacity).

            use_rts (bool, optional): default=True, whether to use Random Token Selection.

            use_tutel (bool, optional): default=False, whether to use Tutel optimizations (if installed).
        """

        super().__init__()


        self.num_experts = num_experts

        if isinstance(expert, nn.Linear):
            self.expert_type = 'linear'
        elif isinstance(expert, nn.MultiheadAttention):
            self.expert_type = 'attention'
        else:
            raise NotImplementedError('please check expert type')

        experts = Experts(expert, cfg, num_experts)

        self.gate = TopKGate(hidden_size,
                             num_experts,
                             k,
                             noisy_gate_policy,
                             cfg,
                             moe_type=self.expert_type)


        self.experts = experts



    def forward(self, hidden_states, gate_decision=None, **kwargs):
        """ MoE forward
        Arguments:
            hidden_states (Tensor): input to the layer
        Returns:
            A tuple including output
            * output (Tensor): output of the model
        """


        if  gate_decision is not None:
            top_indices, gates = gate_decision
        else:
            top_indices, gates = self.gate(hidden_states, **kwargs)

        expert_output = self.experts(hidden_states, top_indices, gates, **kwargs)

        return expert_output, [top_indices, gates]
