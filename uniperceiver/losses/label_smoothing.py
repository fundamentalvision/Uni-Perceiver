import torch
import torch.nn as nn
import torch.nn.functional as F
from uniperceiver.config import configurable
from .build import LOSSES_REGISTRY




@LOSSES_REGISTRY.register()
class LabelSmoothingCrossEntropy(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        label_smoothing,
        loss_weight,
        loss_fp32,
    ):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.label_smoothing = label_smoothing
        self.confidence = 1.0 - self.label_smoothing
        if not isinstance(loss_weight, float):
            self.loss_weight = 1.0
        else:
            self.loss_weight = loss_weight
        self.loss_fp32 = loss_fp32

    @classmethod
    def from_config(cls, cfg):
        return {
            "label_smoothing": cfg.LOSSES.LABELSMOOTHING,
            'loss_weight': getattr(cfg.LOSSES, 'LOSS_WEIGHT', None),
            'loss_fp32': getattr(cfg.LOSSES, 'LOSS_FP32', False),
        }

    def Forward(self, x, target):
        if self.loss_fp32 and x.dtype != torch.float32:
            logprobs = F.log_softmax(x, dim=-1,
                                     dtype=torch.float32).to(x.dtype)
        else:
            logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.label_smoothing * smooth_loss
        return loss.mean()

    def forward(self, outputs_dict):
        ret = {}

        for logit, target, loss_identification in zip(outputs_dict['logits'],
                                            outputs_dict['targets'],
                                            outputs_dict['loss_names']):


            loss = self.Forward(logit, target)
            if self.loss_weight != 1.0:
                loss *= self.loss_weight
            loss_name = 'LabelSmoothing'
            if len(loss_identification) > 0:
                loss_name = loss_name + f' ({loss_identification})'
            ret.update({loss_name: loss})


        return ret
