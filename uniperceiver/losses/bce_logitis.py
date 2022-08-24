import torch
import torch.nn as nn
from uniperceiver.config import configurable
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class BCEWithLogits(nn.Module):
    @configurable
    def __init__(self, loss_weight=1.0):
        super(BCEWithLogits, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        if not isinstance(loss_weight, float):
            self.loss_weight = 1.0
        else:
            self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg):
        return {
            'loss_weight' :  getattr(cfg.LOSSES, 'LOSS_WEIGHT', 1.0)
        }

    def forward(self, outputs_dict):
        ret  = {}
        for logit, target, loss_identification in zip(outputs_dict['logits'],
                                            outputs_dict['targets'],
                                            outputs_dict['loss_names']):

            loss = self.criterion(logit, target)
            if self.loss_weight != 1.0:
                loss *= self.loss_weight
            loss_name = 'BCEWithLogits_Loss'
            if len(loss_identification) > 0:
                loss_name = loss_name+ f' ({loss_identification})'
            ret.update({ loss_name: loss })
            
        return ret
    