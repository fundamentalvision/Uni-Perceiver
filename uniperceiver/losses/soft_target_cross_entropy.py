import torch
import torch.nn as nn
import torch.nn.functional as F

from uniperceiver.config import configurable
from .build import LOSSES_REGISTRY

class CrossEntropyWithSoftTarget(nn.Module):

    def __init__(self, loss_fp32):
        super(CrossEntropyWithSoftTarget, self).__init__()
        self.loss_fp32 = loss_fp32

    def forward(self, x, target):
        if self.loss_fp32 and  x.dtype != torch.float32:
            loss = torch.sum(-target * F.log_softmax(x, dim=-1, dtype=torch.float32), dim=-1).to(x.dtype)
        else:
            loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


@LOSSES_REGISTRY.register()
class SoftTargetCrossEntropy(nn.Module):
    @configurable
    def __init__(self, loss_weight=1.0, loss_fp32=False):
        super(SoftTargetCrossEntropy, self).__init__()
        self.criterion = CrossEntropyWithSoftTarget(loss_fp32)
        if not isinstance(loss_weight, float):
            self.loss_weight = 1.0
        else:
            self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg):
        return {
            'loss_weight' :  getattr(cfg.LOSSES, 'LOSS_WEIGHT', None),
            'loss_fp32' :  getattr(cfg.LOSSES, 'LOSS_FP32', False),
        }

    def forward(self, outputs_dict):
        ret  = {}
        for logit, target, loss_identification in zip(outputs_dict['logits'],
                                            outputs_dict['targets'],
                                            outputs_dict['loss_names']):

            loss = self.criterion(logit, target)
            if self.loss_weight != 1.0:
                loss *= self.loss_weight
            loss_name = 'SoftTargetCrossEntropy_Loss'
            if len(loss_identification) > 0:
                loss_name = loss_name+ f' ({loss_identification})'
            ret.update({ loss_name: loss })


        return ret
