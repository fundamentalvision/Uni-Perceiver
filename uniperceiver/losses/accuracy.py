import torch
import torch.nn as nn
import torch.nn.functional as F
from uniperceiver.config import configurable
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class Accuracy(nn.Module):
    @configurable
    def __init__(
        self
    ):
        super(Accuracy, self).__init__()

    @classmethod
    def from_config(cls, cfg):
        return {
        }

    def Forward(self, logits, targets):
        pred = torch.argmax(logits.view(-1, logits.shape[-1]), -1)
        targets = targets.view(-1)
        return torch.mean((pred == targets).float())

    def forward(self, outputs_dict):

        ret = {}
        for logit, target, loss_identification in zip(outputs_dict['logits'],
                                            outputs_dict['targets'],
                                            outputs_dict['loss_names']):
            if logit.shape == target.shape:
                # for mixup
                target = torch.argmax(target, dim=-1)
            acc = self.Forward(logit, target)
            loss_name = 'Accuracy'
            if len(loss_identification) > 0:
                loss_name = loss_name + f' ({loss_identification})'
            ret.update({loss_name: acc})

        return ret