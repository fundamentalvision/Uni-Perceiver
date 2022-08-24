'''
Copyright 2021 The Microsoft DeepSpeed Team
'''
# The file has been adapted from two fairscale files:
# (1) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/moe_layer.py
# (2) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/top2gate.py
# Git commit hash: 34df606902a240567a0d898037ece55c2f1336cf
# We retain the following license from the original files:

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import time
from time import perf_counter
import torch
from torch import nn
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from uniperceiver.utils.events import get_event_storage
from torch.cuda.amp import autocast



if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}
normal_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}


import torch.distributed.nn
from uniperceiver.utils import comm
from uniperceiver.modeling.layers import FP16LayerNorm



def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0 - epsilon, device=device),
            high=torch.tensor(1.0 + epsilon,
                              device=device)).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero,
                                                   one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


def normal_rsample(shape: Tuple, device: torch.device, num_expert: int) -> Tensor:
    normal = normal_map.get(device)
    if normal is None:
        std = torch.tensor(1.0/num_expert, device=device)
        mean = torch.tensor(0.0, device=device)
        normal = torch.distributions.normal.Normal(mean, std).rsample  # type: ignore
        normal_map[device] = normal
    return normal(shape)


def one_hot_with_dtype(data, num_classes, dtype):
    result = torch.zeros([data.size(0), num_classes],
                         device=data.device,
                         dtype=dtype)
    result.scatter_(1, data.unsqueeze(-1), 1)
    return result

@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]


@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()




class TopKGate(nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    # wg: torch.nn.Linear

    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 k: int = 1,
                 noisy_gate_policy: Optional[str] = None,
                 cfg: dict = None,
                 moe_type: str = None,
                 **kwargs):
        super().__init__( )

        if k != 1 and k != 2:
            raise ValueError('Only top-1 and top-2 gatings are supported.')
        self.model_dim = model_dim
        self.k = k

        self.cfg = cfg

        self.noisy_gate_policy = noisy_gate_policy
        self.noise_std = self.cfg.MOE.NOISE_STD

        self.batch_prioritized_routing = self.cfg.MOE.BATCH_PRIO
        self.gate = self.cfg.MOE.GATE_TYPE



        self.layer_type = kwargs.pop('moe_type', 'ffn')

        self.tag_transform_enable = self.cfg.MOE.TAG_Transform

        self.moe_type = moe_type

        if self.cfg.SOLVER.FORCE_LN_FP16:
            LayerNormModule = FP16LayerNorm
        else:
            LayerNormModule = torch.nn.LayerNorm
        if self.tag_transform_enable and self.cfg.MOE.TAG_Transform_ACT:
            self.tag_transform = torch.nn.Sequential(torch.nn.Linear(self.cfg.MOE.ATTRIBUTE_LENGTH, self.model_dim), torch.nn.GELU(),
                                                     LayerNormModule(self.model_dim))
        else:
            self.tag_transform = torch.nn.Sequential(torch.nn.Linear(self.cfg.MOE.ATTRIBUTE_LENGTH, self.model_dim), LayerNormModule(self.model_dim))

        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float()





        pass



    def tag_gate(self, x, data_type=None, moe_embedding:torch.Tensor = None, **kwargs):
        if self.cfg.MODEL.TAG_TRANSFORM_FP32:
            with autocast(enabled=False):
                gate_embed = self.tag_transform.float()(moe_embedding.float())
        else:
            gate_embed = self.tag_transform(moe_embedding)


        return gate_embed




    def forward(
        self,
        input,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore


        if self.tag_transform_enable:
            input = self.tag_gate(input, **kwargs)
        if self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
        input_fp32 = input.float()
        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
        with autocast(enabled=not self.cfg.MODEL.GATE_FP32):
            if self.cfg.SOLVER.FORCE_WG_RECAST:
                # used for dbeugging only
                logits = self.wg.half().float()(input_fp32)
            else:
                logits = self.wg(input_fp32)

        if self.k == 1 and self.gate == 'deepspeed':
            gate_output = self.top1gating(
                logits,
                self.noisy_gate_policy if self.training else None,
                **kwargs)

        # tutel gate function
        else:
            gate_output = self.top2gating(
                logits,
                self.noisy_gate_policy if self.training else None,
                **kwargs )


            return gate_output

    def load_balance(self, gates, mask1, num_experts, data_type=None):
        # Compute l_aux
        if self.balance_loss and self.training:
            # TODO: for retrieval task, these maybe some gpu do not have  this input

            if data_type == 'INPUT':
                if comm._LOCAL_IMAGE_LENGTH > 0 and not comm._LOCAL_UTOKEN_LENGTH + comm._LOCAL_GTOKEN_LENGTH > 0:
                    # input image features only
                    me = gates.sum(dim=0)
                    ce = mask1.sum(dim=0)

                    # maybe has retrieval pair
                    if comm._MOE_TARGET_MECE_LIST.get(str(comm._LOCAL_CURRENT_LAYER)+'_'+self.layer_type, None) is not None:
                        # if len(comm._MOE_TARGET_MECE_LIST) > 0:
                        me_t, ce_t = comm._MOE_TARGET_MECE_LIST[
                            str(comm._LOCAL_CURRENT_LAYER) + '_' +
                            self.layer_type]
                        me = me + me_t
                        ce = ce + ce_t

                    me = me * self.task_weights[comm._LOCAL_CURRENT_TASK]
                    ce = ce * self.task_weights[comm._LOCAL_CURRENT_TASK]

                elif comm._LOCAL_IMAGE_LENGTH > 0 and comm._LOCAL_UTOKEN_LENGTH + comm._LOCAL_GTOKEN_LENGTH > 0:

                    # sum of these two distribution from two modalities
                    me = gates.sum(dim=0)
                    ce = mask1.sum(dim=0)

                    me = me * self.task_weights[comm._LOCAL_CURRENT_TASK]
                    ce = ce * self.task_weights[comm._LOCAL_CURRENT_TASK]

                elif comm._LOCAL_IMAGE_LENGTH <= 0 and comm._LOCAL_UTOKEN_LENGTH + comm._LOCAL_GTOKEN_LENGTH > 0:

                    me = gates.sum(
                        dim=0) * self.task_weights[comm._LOCAL_CURRENT_TASK]
                    ce = mask1.sum(
                        dim=0) * self.task_weights[comm._LOCAL_CURRENT_TASK]
                    # raise NotImplementedError
                else:

                    raise NotImplementedError

            elif data_type == 'TARGET':
                # the retrieval embedding

                # only remove the padding contributions

                comm._MOE_TARGET_MECE_LIST[str(comm._LOCAL_CURRENT_LAYER) + '_' +self.layer_type] = [gates.sum(dim=0), mask1.sum(dim=0)]

            elif data_type == 'IN_LABEL':
                # remove paddings contributions

                me = gates.sum(dim=0)
                ce = mask1.sum(dim=0)

            elif data_type == 'WORD_VOCAB':
                # do not need padding mask
                me = gates.sum(dim=0)
                ce = mask1.sum(dim=0)
            else:
                raise NotImplementedError

            # debug left

            if not data_type == 'TARGET':
                me = torch.distributed.nn.all_reduce(
                    me) / comm.get_world_size()
                ce = torch.distributed.nn.all_reduce(
                    ce) / comm.get_world_size()

                if data_type not in comm._MOE_LOSSES_COLLECTIONS[
                        'exp_balance']:
                    comm._MOE_LOSSES_COLLECTIONS['exp_balance'][
                        data_type] = []
                comm._MOE_LOSSES_COLLECTIONS['exp_balance'][
                    data_type].append([me, ce])


    def top1gating(
            self,
            logits: Tensor,
            noisy_gate_policy: Optional[str] = None,
            **kwargs,
            ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Implements Top1Gating on logits."""

        logits_w_noise = None
        if noisy_gate_policy == 'RSample':
            logits_w_noise = logits + gumbel_rsample(logits.shape,
                                                     device=logits.device)
        elif noisy_gate_policy == 'vmoe':
            num_experts = int(logits.shape[-1])
            logits_w_noise = logits + normal_rsample(logits.shape,
                                                     device=logits.device,
                                                     num_expert=num_experts/self.noise_std)

        # everything is in fp32 in this function
        gates = F.softmax(logits, dim=1)
        # Create a mask for 1st's expert per token
        # noisy gating
        indices1_s = torch.argmax(logits_w_noise if logits_w_noise is not None else gates, dim=1)

        num_experts = int(gates.shape[1])
        mask1 = F.one_hot(indices1_s, num_classes=num_experts)

        # gating decisions
        exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

        self.load_balance(gates, mask1, num_experts)

        self.tb_output(
            mask1,
            exp_counts,
            gates=None
        )

        gates = (gates*mask1).sum(dim=1)
        self.tb_output(mask1=None, exp_counts=None, gates=[gates])

        return [indices1_s], [gates]




    def top2gating(
        self,
        logits: Tensor,
        noisy_gate_policy: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Implements Top2Gating on logits."""
        # everything is in fp32 in this function

        num_experts = int(logits.shape[-1])

        logits_w_noise = None
        if noisy_gate_policy == 'RSample':
            logits_w_noise = logits + gumbel_rsample(logits.shape,
                                                     device=logits.device) * self.noise_std
        elif noisy_gate_policy == 'vmoe':
            logits_w_noise = logits + normal_rsample(logits.shape,
                                                     device=logits.device,
                                                     num_expert=num_experts/self.noise_std)

        # topk_indices = torch.topk(logits, self.k, dim=1).indices
        topk_indices = torch.topk(
            logits_w_noise
            if logits_w_noise is not None else logits,
            self.k,
            dim=1).indices

        indices_s = [x.view(-1) for x in topk_indices.chunk(self.k, dim=1)]
        masks_se = [
            one_hot_with_dtype(x, num_classes=num_experts, dtype=x.dtype)
            for x in indices_s
        ]


        if noisy_gate_policy == 'vmoe':
            gates = F.softmax(logits_w_noise, dim=1)

        else:
            gates = F.softmax(logits, dim=1)

        # self.load_balance(gates, masks_se[0], num_experts)
        gates_s = [(gates * x).sum(dim=1) for x in masks_se]

        # gating decisions
        exp_counts = torch.sum(masks_se[0], dim=0).detach().to('cpu')

        # self.tb_output(masks_se[0], exp_counts, gates=None)
        # if self.k>1:
        #     for k in range(1, self.k):
        #         self.tb_output(masks_se[k], torch.sum(masks_se[k], dim=0).detach().to('cpu'), None, postfix='_top{}'.format(k+1))



        if self.k > 1:

            # Normalize Gate
            denom_s = torch.clamp(sum(gates_s),
                                  min=torch.finfo(gates_s[0].dtype).eps)
            gates_s = [x / denom_s for x in gates_s]

        # self.tb_output(mask1=None, exp_counts=None, gates=gates_s)

        return indices_s, gates_s


    def tb_output(self, data_type=None, mask1=None, exp_counts=None, gates=None, postfix=''):
        if self.training:
            storage = get_event_storage()
        else:
            return

        if not (comm._LOCAL_CURRENT_TASK == 'imagenet' or comm._LOCAL_CURRENT_TASK.startswith('bookswiki') or comm._LOCAL_CURRENT_TASK.startswith('cc3m') or comm._LOCAL_CURRENT_TASK.startswith('cc12m') or comm._LOCAL_CURRENT_TASK.startswith('tqa')):
            # to save time
            return

        if  (storage._iter+1)%(comm._EXPERT_LOG_INTERVAL//10) != 0:
            # to save time
            return


        if storage is not None and comm.is_main_process():
            # pass
            # for each expert

            if gates is not None:
                if data_type == "INPUT" and comm._LOCAL_IMAGE_LENGTH > 0:


                    gate_logs = {
                        "logits_layer{}_expert_{}/top{}_{}_{}_v".format(
                            comm._LOCAL_CURRENT_LAYER, self.layer_type,
                            e_id+1, comm._LOCAL_CURRENT_TASK,
                            data_type): ratio[0]
                        for e_id, ratio in enumerate(gates)
                    }
                    storage.put_scalars(**gate_logs, avg_hint=True)


                    if gates[0].shape[0] > 1:
                        gates_t_logs = {
                            "logits_layer{}_expert_{}/top{}_{}_{}_t".
                            format(comm._LOCAL_CURRENT_LAYER,
                                    self.layer_type, e_id+1,
                                    comm._LOCAL_CURRENT_TASK,
                                    data_type): ratio[1]
                            for e_id, ratio in enumerate(gates)
                        }
                        storage.put_scalars(**gates_t_logs, avg_hint=True)

                elif data_type in ['IN_LABEL', 'WORD_VOCAB']:

                    gates_logs = {
                        "logits_layer{}_expert_{}/top{}_{}".format(
                            comm._LOCAL_CURRENT_LAYER, self.layer_type,
                            e_id+1, data_type): ratio[0]
                        for e_id, ratio in enumerate(gates)
                    }
                    storage.put_scalars(**gates_logs, avg_hint=True)

                else:

                    gates_logs = {
                        "layer{}_expert_{}/top{}_{}_{}".format(
                            comm._LOCAL_CURRENT_LAYER, self.layer_type,
                            e_id+1, comm._LOCAL_CURRENT_TASK,
                            data_type): ratio[0]
                        for e_id, ratio in enumerate(gates)
                    }
                    storage.put_scalars(**gates_logs, avg_hint=True)

            else:

                if data_type == "INPUT" and comm._LOCAL_IMAGE_LENGTH > 0:

                    exp_counts_v = mask1[0]
                    exp_count_logs = {
                        "layer{}_expert_{}/E{}_{}_{}_v{}".format(
                            comm._LOCAL_CURRENT_LAYER, self.layer_type, e_id,
                            comm._LOCAL_CURRENT_TASK, data_type,
                            postfix): ratio
                        for e_id, ratio in enumerate((exp_counts_v /
                                                    exp_counts_v.sum()).tolist())
                    }
                    storage.put_scalars(**exp_count_logs, avg_hint=True)

                    if mask1.size(0)>1:
                        exp_counts_t = mask1[1]
                        exp_count_logs = {
                            "layer{}_expert_{}/E{}_{}_{}_t{}".format(
                                comm._LOCAL_CURRENT_LAYER, self.layer_type, e_id,
                                comm._LOCAL_CURRENT_TASK,
                                data_type, postfix): ratio
                            for e_id, ratio in enumerate((
                                exp_counts_t / exp_counts_t.sum()).tolist())
                        }
                        storage.put_scalars(**exp_count_logs, avg_hint=True)



                elif data_type in ['IN_LABEL', 'WORD_VOCAB']:
                    exp_count_logs = {
                        "layer{}_expert_{}/E{}_{}{}".format(
                            comm._LOCAL_CURRENT_LAYER, self.layer_type, e_id,
                            data_type, postfix): ratio
                        for e_id, ratio in enumerate((exp_counts /
                                                    exp_counts.sum()).tolist())
                    }
                    storage.put_scalars(**exp_count_logs, avg_hint=True)

                else:
                    exp_count_logs = {
                        "layer{}_expert_{}/E{}_{}_{}{}".format(
                            comm._LOCAL_CURRENT_LAYER, self.layer_type, e_id,
                            comm._LOCAL_CURRENT_TASK, data_type,
                            postfix): ratio
                        for e_id, ratio in enumerate((exp_counts /
                                                    exp_counts.sum()).tolist())
                    }
                    storage.put_scalars(**exp_count_logs, avg_hint=True)
