import torch
import torch.nn as nn
from uniperceiver.config import configurable
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class CrossEntropy(nn.Module):
    @configurable
    def __init__(self, loss_weight=1.0, reduction='mean', loss_fp32=False):
        super(CrossEntropy, self).__init__()
        if reduction is None:
            reduction = 'mean'
        self.criterion_func = nn.CrossEntropyLoss(ignore_index=-1, reduction=reduction)
        if not isinstance(loss_weight, float):
            self.loss_weight = 1.0
        else:
            self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss_fp32 = loss_fp32

    def criterion(self, x, target):
        if self.loss_fp32 and x.dtype != torch.float32:
            loss = self.criterion_func(x.to(torch.float32), target).to(x.dtype)
        else:
            loss = self.criterion_func(x, target)
        return loss.mean()

    @classmethod
    def from_config(cls, cfg):
        return {
            'loss_weight': getattr(cfg.LOSSES, 'LOSS_WEIGHT', None),
            'reduction': getattr(cfg.LOSSES, 'REDUCTION', 'mean'),
            'loss_fp32': getattr(cfg.LOSSES, 'LOSS_FP32', False),
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.LOSSES.LOSS_WEIGHT = None
        cfg.LOSSES.REDUCTION = 'mean'

    def forward(self, outputs_dict):
        ret  = {}

        for logit, target, loss_identification in zip(outputs_dict['logits'],
                                            outputs_dict['targets'],
                                            outputs_dict['loss_names']):

            loss = self.criterion(logit, target)
            if self.loss_weight != 1.0:
                loss *= self.loss_weight
            loss_name = 'CrossEntropy_Loss'
            if len(loss_identification) > 0:
                loss_name = loss_name+ f' ({loss_identification})'
            ret.update({ loss_name: loss })


        return ret
