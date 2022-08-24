import numpy as np
import os
import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
import random
from uniperceiver.utils import comm
import itertools

from .sampler import TrainingSampler, NaiveSampler, NodeDistributedSampler

from uniperceiver.datasets.unified_dataset import UnifiedDataset
try:
    import deepspeed.utils.groups as groups
    DEEPSPEED_INSTALLED = True
except:

    DEEPSPEED_INSTALLED = False
    




class WeightedBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self,
                 dataset: UnifiedDataset,
                 cfg,
                 task_cfg,
                 stage='train',
                 shuffle=True,
                 drop_last=True):
        self.dataset = dataset
        self.cfg = cfg
        self.task_cfg = task_cfg

        self._tasks = list(self.task_cfg.keys())

        # dataset_names = self.dataset.dataset_name
        # dataset_list = self.dataset.dataset_list

        unit_sampler = dict()
        for name, new_cfg in self.task_cfg.items():
            if new_cfg.DATASETS.DATASET_NAME in [
                    "MSCOCO", "FLICKR", "ImageNet22k", "ImageNet1k", "VG", "VideoDataSet", "K700", 'K400', 'MiT', 'MSVDDataset', 'MSRVTTDataset',
                    "RTE", "CoLA", "SST-2", "MRPC", "QQP", "QNLI", "MNLI", "MNLI_Match", "VQA"
            ]:
                sampler = TrainingSampler(self.dataset.datasets[name])
            elif new_cfg.DATASETS.DATASET_NAME in ["BooksWiki"]:
                # block cache
                sampler = NaiveSampler(self.dataset.datasets[name])
            elif new_cfg.DATASETS.DATASET_NAME in [
                    # "ImageTextPairDataset", 'SBUDataset', 'TQAPretrain'
                    'YFCC', 'CC12M', 'CC3M', 'SBU', 'TQAPretrain'
            ]:
                sampler = NodeDistributedSampler(
                    self.dataset.datasets[name],
                    shuffle=True,
                    num_replicas=comm.get_world_size(),
                    rank=comm.get_rank(),
                    local_rank=comm.get_local_rank(),
                    local_size=comm.get_local_size())
            else:
                raise NotImplementedError(
                    f'please check the sampler used for this dataset {new_cfg.DATASETS.DATASET_NAME}'
                )
            unit_sampler[name] = sampler
        self.unit_sampler = unit_sampler

        self.unit_sampler_iter = {
            k: iter(v)
            for k, v in self.unit_sampler.items()
        }

        self.sampling_weights = {
            k: v.DATALOADER.SAMPLING_WEIGHT
            for k, v in self.task_cfg.items()
        }

        self._weights = [self.sampling_weights[k] for k in self._tasks]

        self.stage = stage
        if self.stage == 'train':
            self.task_batch_size =  {
            k: v.DATALOADER.TRAIN_BATCH_SIZE
            for k, v in self.task_cfg.items()
            }
        else:
            raise NotImplementedError('only train dataset supportted now!')



        self.len = [ len_ds//bs for len_ds, bs in zip([len(ds) for ds in self.dataset.dataset_list], self.task_batch_size.values())]

        self.special_strategy = cfg.DATALOADER.STRATEGY

        self.count = 0

        self.task_index_offset = {
            k: v
            for k, v in zip(self.task_cfg.keys(),self.dataset.dataset_scale.tolist())
        }


    def __len__(self):
        return sum(self.len)

    def __iter__(self):

        batch = []
        while True:

            if self.special_strategy == 'uniform':
                task = self._tasks[comm.get_local_rank() % len(self._tasks)]
            elif self.special_strategy == 'uniformv2':
                task = self._tasks[(self.count + comm.get_rank()) %
                                len(self._tasks)]
                self.count = (self.count + 1) % len(self._tasks)
            elif self.special_strategy == 'turn':
                task = self._tasks[self.count % len(self._tasks)]
                self.count = (self.count + 1) % len(self._tasks)
            else:
                task = random.choices(self._tasks,
                                    weights=self._weights)[0]

            if self.cfg.MOE.MOE and DEEPSPEED_INSTALLED and groups.expert_parallel_is_initialized(
            ) and groups.get_expert_data_parallel_world_size() > 1:
                task = comm.broadcast_object(
                    task,
                    src=comm.get_rank() -
                    comm.get_rank() % groups.get_expert_parallel_world_size(),
                    group=groups.get_expert_parallel_group())

            """
                all sampler are infinite stream
            """
            sample_index_offset = self.task_index_offset[task]
            for i in range(self.task_batch_size[task]):
                try:
                    batch.append(
                        next(self.unit_sampler_iter[task]) + sample_index_offset)
                except:
                    self.unit_sampler_iter[task] = iter(self.unit_sampler[task])
                    batch.append(
                        next(self.unit_sampler_iter[task]) + sample_index_offset)

            assert len(batch) == self.task_batch_size[task]
            yield batch
            batch = []
