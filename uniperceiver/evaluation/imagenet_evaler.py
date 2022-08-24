
import os
import sys
import tempfile
import json
from json import encoder

import torch
from uniperceiver.config import configurable
from .build import EVALUATION_REGISTRY

# from timm.utils import accuracy

from uniperceiver.utils import comm

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

@EVALUATION_REGISTRY.register()
class ImageNetEvaler(object):
    def __init__(self, cfg, annfile, output_dir):
        super(ImageNetEvaler, self).__init__()
        self.ann_file = annfile
        with open(self.ann_file, 'r') as f:
            img_infos = f.readlines()
        
        target = [int(info.replace('\n', '').split(' ')[1]) for info in img_infos]
        self.target = torch.tensor(target)
        

    def eval(self, results, epoch):
        
        # sort the result for multi-gpu evaluation
        results = {res['image_id']: res['cls_logits'] for res in results}
        results = [results[i] for i in sorted(results.keys())]
        
        results = torch.stack(results)
        
        acc1, acc5 = accuracy(results, self.target.to(device=results.device), topk=(1, 5))
        # acc1, acc5 = accuracy(results, self.target[:results.size(0)].to(device=results.device), topk=(1, 5))
        return {'Acc@1': acc1, 'Acc@5': acc5}