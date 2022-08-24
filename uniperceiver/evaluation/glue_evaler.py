import os
import sys
import pickle
import json
from json import encoder
from uniperceiver.config import configurable
from .build import EVALUATION_REGISTRY

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

@EVALUATION_REGISTRY.register()
class GLUEEvaler(object):
    def __init__(self,  cfg, *args, **kwargs):
        super(GLUEEvaler, self).__init__()
        self.task_name = cfg.DATASETS.DATASET_NAME
        self.tasks = [""]



    def eval(self, results, epoch):
        preds = []
        labels = []
        for result in results:
            # cls task
            if self.task_name != 'STS-B':
                preds.append(result["pred"].argmax().item())
                labels.append(int(result["label"]))

            else:
                # regression task
                preds.append(float(result["pred"].sigmoid().item()))
                labels.append(float(result["label"]))

        preds = np.array(preds)
        labels = np.array(labels)

        if self.task_name == 'CoLA':
            acc = simple_accuracy(preds, labels)
            matthewscorr = matthews_corrcoef(labels, preds)
            result = {
                "accuracy": acc,
                "matthews_corrcoef": matthewscorr,
            }
        elif self.task_name in [ 'QNLI', 'RTE', 'SST-2'] or self.task_name.startswith("MNLI"):
            acc = simple_accuracy(preds, labels)
            result = {
                "accuracy": acc,
            }
        elif self.task_name in ['MRPC', 'QQP']:
            acc = simple_accuracy(preds, labels)
            f1 = f1_score(y_true=labels, y_pred=preds)
            result = {
                "accuracy": acc,
                "f1_score": f1,
            }
        elif self.task_name in ['STS-B']:
            pearson_corr = pearsonr(preds, labels)[0]
            spearman_corr = spearmanr(preds, labels)[0]
            result ={
                "pearson_corr": pearson_corr,
                "spearman_corr": spearman_corr,
            }
        else:
            raise NotImplementedError

        return result
