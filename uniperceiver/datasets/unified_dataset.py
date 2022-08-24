import os
import copy
import torch

from .build import DATASETS_REGISTRY
from uniperceiver.config import configurable



import numpy as np
"""

    only standard dataset support now 
    class dataset:
        def __len__(self):
            pass 

        def __getitem__(self, index):
            pass 

"""

class UnifiedDataset:

    def __init__(self, cfg, task_cfg, stage, **kwargs):
        self.cfg = cfg
        self.task_cfg =task_cfg
        assert stage == 'train', f'now only training dataset is supported'

        datasets = dict()
        for name, new_cfg in self.task_cfg.items():
            datasets[name] = self.build_unit_dataset(new_cfg,
                                                     new_cfg.DATASETS.TRAIN,
                                                     stage=stage)
        self.datasets = datasets

        self.dataset_name = list(self.datasets.keys())
        self.dataset_list = list(self.datasets.values())


        self.dataset_length = np.array([len(ds) for ds in self.datasets.values()])

        self.dataset_scale = np.array([0] + np.cumsum(self.dataset_length).tolist()[:-1])
        # [0, 876, 128000, ....]


        pass

    def build_unit_dataset(self, cfg, name, stage):
        dataset_mapper = DATASETS_REGISTRY.get(name)(cfg, stage)
        return dataset_mapper

    def __len__(self):
        return np.cumsum(self.dataset_length).tolist()[-1]


    def __getitem__(self, index):
        dataset_index = (index >= self.dataset_scale).sum() - 1 # the dataset index 
        offset = self.dataset_scale[dataset_index]
        ret = self.dataset_list[dataset_index][index-offset]
        ret.update({"task_name": self.dataset_name[dataset_index]})
        return ret 
