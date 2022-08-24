
import os
import sys
import numpy as np
import torch
from uniperceiver.config import configurable
from .build import EVALUATION_REGISTRY

@EVALUATION_REGISTRY.register()
class RetrievalEvaler(object):
    def __init__(self, cfg, annfile, output_dir,):
        super(RetrievalEvaler, self).__init__()
        self.eval_bs = cfg.INFERENCE.EVAL_BS
        pass

    def eval(self, vfeats, tfeats, labels, prefix=None):
        count = 0
        batch_size = self.eval_bs
        batch_num = tfeats.size(0) // batch_size
        rank_matrix = np.ones((tfeats.size(0))) * vfeats.size(0)
        for i in range(batch_num):
            if i == batch_num - 1:
                b_tfeats = tfeats[i*batch_size:]
            else:
                b_tfeats = tfeats[i*batch_size:(i+1)*batch_size]

            with torch.no_grad():
                scores = (b_tfeats.unsqueeze(1) * vfeats.unsqueeze(0)).sum(dim=-1).cpu().numpy()
            for score in scores:
                # rank = np.where((np.argsort(-score) == np.where(labels[count]==1)[0][0]) == 1)[0][0]
                rank = min([10] + [np.where((np.argsort(-score) == index) == 1)[0][0] for index in np.where(labels[count]==1)[0]])
                rank_matrix[count] = rank
                count += 1

        r1 = 100.0 * np.sum(rank_matrix < 1) / len(rank_matrix)
        r5 = 100.0 * np.sum(rank_matrix < 5) / len(rank_matrix)
        r10 = 100.0 * np.sum(rank_matrix < 10) / len(rank_matrix)

        rmean = (r1+r5+r10)/3

        # medr = np.floor(np.median(rank_matrix) + 1)
        # meanr = np.mean(rank_matrix) + 1
        if prefix is None:
            return {
                "r1": r1,
                "r5": r5,
                "r10": r10,
                "rmean": rmean,
                # "meanr": meanr
            }
        else:
            return {
                prefix+ "-r1": r1,
                prefix+ "-r5": r5,
                prefix+ "-r10": r10,
                prefix+ "-rmean": rmean,
                # prefix+ "-meanr": meanr
            }
