import itertools
import logging
import numpy as np
import operator
import pickle
from tabulate import tabulate
from termcolor import colored
import torch
import torch.utils.data
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler

from uniperceiver.config import configurable
from uniperceiver.utils.comm import get_world_size, get_rank
from uniperceiver.utils.env import seed_all_rng
from uniperceiver.utils.file_io import PathManager
from uniperceiver.utils.logger import log_first_n
from uniperceiver.utils.registry import Registry
from .common import DatasetFromList, MapDataset

from uniperceiver.functional import pad_tensor, dict_to_cuda, flat_list_of_lists
from .sampler import NodeDistributedSampler
from uniperceiver.utils import comm
from .sampler import TrainingSampler, NaiveSampler
from .moe_embeddings import get_moe_embedding, get_embed_with_task_type, get_embed_with_shared_tagert_name



from functools import partial

DATASETS_REGISTRY = Registry("DATASETS")  # noqa F401 isort:skip
DATASETS_REGISTRY.__doc__ = """
Registry for datasets, i.e. the whole model
"""

from uniperceiver.datasets.unified_dataset import UnifiedDataset
from .batch_sampler import WeightedBatchSampler


def build_dataset_mapper(cfg, name, stage):
    dataset_mapper = DATASETS_REGISTRY.get(name)(cfg, stage)
    return dataset_mapper

def trivial_batch_collator(batch):
    return batch

def preprocess_batch_collator(batched_inputs, cfg=dict(), shared_targets=dict()):

    ret = {}
    if cfg.MOE.MOE:
        moe_type =  cfg.MOE.MOE_TYPE
    else:
        moe_type = None
    # sample lists
    for data_name in ['input_sample', 'target_sample']:
        ret[(data_name + '_list')] = []
        num_data = len(batched_inputs[0][data_name])
        for i in range(num_data):
            # All samples in data_list can be either be Tensors or groups (i.e., list of Tensors, [Tensors]).
            # If the samples in data_list are groups, each element in each group will be padded individually, and then all elements in the same group will be concatenated along axis 1.
            data_list = [sample[data_name][i]['data'] for sample in batched_inputs]
            # valid_mask_list = [sample[data_name][i]['valid_mask'] for sample in batched_inputs]
            modality = batched_inputs[0][data_name][i]['modality']
            data_type = batched_inputs[0][data_name][i]['data_type']
            sample_info_list = [sample[data_name][i]['sample_info'] for sample in batched_inputs]
            padding_value = sample_info_list[0].get('padding_value', 0)

            if isinstance(data_list[0], list):
                if not batched_inputs[0][data_name][i]['sample_info'].get('sample_alone', False):
                    # some data are concatenated inside one sample, e.g. the caption text part during the training.
                    data_group_size = len(data_list[0])
                    # padding individually for each element in each group
                    data_, valid_mask_ = zip(*[pad_tensor(
                        tensor=[data_group[idx] for data_group in data_list],
                        padding_value=padding_value,
                        use_mask=True) for idx in range(data_group_size)])

                    # concatenate all elements in the same group along axis 1
                    data = torch.cat(data_, dim=1)
                    valid_mask = torch.cat(valid_mask_, dim=1)
                else:
                    # image-text retrieval may have multi-caption for one image when inference, e.g., MSCOCO caption dataset.
                    data_list = flat_list_of_lists(data_list)
                    data, valid_mask = pad_tensor(tensor=data_list, padding_value=padding_value, use_mask=True)

            elif isinstance(data_list[0], torch.Tensor):
                if sample_info_list[0].get('cat_along_first_dim', False):
                    # concatenate data along the first dimention, e.g.: video data
                    data = torch.cat(data_list, dim=0)
                    valid_mask = None
                else:
                    data, valid_mask = pad_tensor(tensor=data_list, padding_value=padding_value, use_mask=True) # Do we have valid mask that is not caused by padding? AND 1/0 for what?

            else:
                raise TypeError

            if valid_mask is not None and valid_mask.all():
                valid_mask = None

            ret[(data_name + '_list')].append({
                'data':
                data,
                'invalid_mask':
                1 - valid_mask if valid_mask is not None else None,
                'modality':
                modality,
                'data_type':
                data_type,
                'sample_info':
                sample_info_list,
                'moe_embedding':
                get_embed_with_task_type(moe_type, batched_inputs[0]['task_info']['task_type'], data_type)
            })


    # target sets
    num_target_sets = len(batched_inputs[0]['target_idx'])
    # change value to -1 for padding location
    ret['target_idx_list'] = [ pad_tensor(tensor=[sample['target_idx'][i] for sample in batched_inputs], padding_value=-1, use_mask=False)   if isinstance(batched_inputs[0]['target_idx'][i], torch.Tensor) else torch.tensor([sample['target_idx'][i] for sample in batched_inputs] )  for i in range(num_target_sets) ]
    ret['target_set_list'] = [batched_inputs[0]['target_set'][i] for i in range(num_target_sets)]

    # shared target sets
    ret['shared_target_sets'] = {}
    for k in shared_targets:
        padding_value = shared_targets[k]['sample_info'].get('padding_value', 0)
        if isinstance(shared_targets[k]['data'][0], list):
            data_list = [d[np.random.randint(0, len(d))] for d in shared_targets[k]['data']] # Randomly choose one for each list
        else:
            data_list = shared_targets[k]['data']

        data, valid_mask = pad_tensor(tensor=data_list, padding_value=padding_value, use_mask=True)
        if valid_mask.all():
            valid_mask = None
        ret['shared_target_sets'][k] = [{
            'data': data,
            'invalid_mask': 1 - valid_mask if valid_mask is not None else None,
            'modality': shared_targets[k]['modality'],
            'data_type': 'target',
            'sample_info': shared_targets[k]['sample_info'],
            'moe_embedding': get_embed_with_shared_tagert_name(moe_type, k)
        }]

    # task info
    ret['task_info'] = batched_inputs[0]['task_info'] # should task_name be put into task_info?

    ret['task_info']['task_name'] = batched_inputs[0].get('task_name', None)


    return ret



def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)

def load_pkl_file(filepath):
    return pickle.load(open(filepath, 'rb'), encoding='bytes') if len(filepath) > 0 else None

def load_shared_targets(cfg, stage='train'):
    shared_targets_cfg = cfg.SHARED_TARGETS
    shared_targets = {}
    for shared_target_cfg in shared_targets_cfg:
        name = shared_target_cfg['NAME']

        if (stage != 'train') and (name not in cfg.DATASETS.TARGET_SET):
            # For validation and test, we build a dataloader for each task / dataset.
            # Therefore, the dataloader only needs to load its corresponding shared target set.
            continue

        # For validation and test, we do not distribute the shared targets
        distributed = shared_target_cfg['SHARED_TARGETS_CFG']['DISTRIBUTED'] and (stage == 'train')

        shared_targets[name] = load_pkl_file(shared_target_cfg['SHARED_TARGETS_CFG']['FILE_PATH'])

        data = shared_targets[name]['data']
        if isinstance(data[0], list):
            max_len = max([len(t) for tl in data for t in tl])
        else:
            max_len = max([len(t) for t in data])
        shared_targets[name]['sample_info'] = {'distributed': distributed, 'max_len': max_len}

        if distributed:
            world_size = get_world_size()
            rank = get_rank()
            total_num = len(shared_targets[name]['data'])
            local_num = int(np.ceil(total_num / world_size))

            # we pad the shared_targets to a value that can be divided by WORLD_SIZE with no remainer, and then slice it
            if local_num * world_size > total_num:
                data = data + [data[0] for _ in range(local_num * world_size - total_num)]
            shared_targets[name]['data'] = data[rank * local_num : (rank + 1) * local_num]

            # compute the real (unpadded) length of the local slice
            start_idx = min(rank * local_num, total_num)
            end_idx = min((rank + 1) * local_num, total_num)

            shared_targets[name]['sample_info'].update({
                'total_num': total_num,
                'local_num': end_idx - start_idx,
                'world_size': world_size,
                'rank': rank
            })

    return shared_targets



def build_unified_train_loader(cfg, task_cfg, model=None):
    dataset = UnifiedDataset(cfg, task_cfg, stage="train")
    batchsampler = WeightedBatchSampler(dataset, cfg, task_cfg)
    shared_targets = load_shared_targets(cfg)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=batchsampler,
        # sampler=sampler,
        # batch_size=cfg.DATALOADER.TRAIN_BATCH_SIZE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=partial(preprocess_batch_collator, shared_targets=shared_targets, cfg=cfg),
        pin_memory=cfg.DATALOADER.PIN_MEM,
        worker_init_fn=worker_init_reset_seed,
        # drop_last=True,
        prefetch_factor=cfg.DATALOADER.PREFETCH_FACTOR, # default: 2
        persistent_workers=cfg.DATALOADER.NUM_WORKERS>0)


    return dataloader


def build_standard_train_loader(cfg, model=None):
    dataset = build_dataset_mapper(cfg, name=cfg.DATASETS.TRAIN, stage="train")
    if cfg.DATASETS.TRAIN in [ "ImageTextPairDataset", "ImageNet22KDataset", "ImageNetDataset", "VGPretrain", "VideoDataSet", "VQADataset" ]:
        sampler = TrainingSampler(dataset)
    elif cfg.DATASETS.TRAIN in ["GeneralCorpusDataset"]:
        sampler = NaiveSampler(dataset)
    else:
        sampler = NodeDistributedSampler(
                    dataset, shuffle=True,
                    num_replicas=comm.get_world_size(), rank=comm.get_rank(),
                    local_rank=comm.get_local_rank(), local_size=comm.get_local_size())
    # sampler = TrainingSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=cfg.DATALOADER.TRAIN_BATCH_SIZE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=partial(preprocess_batch_collator, model=model),
        pin_memory=cfg.DATALOADER.PIN_MEM,
        worker_init_fn=worker_init_reset_seed,
        drop_last=True,
        persistent_workers=True)
    return dataloader


def _single_modal_dataset(cfg, dataset_mapper=None, *, datalist=None, sampler=None):
    if len(cfg.DATASETS.TRAIN) > 0:
        if dataset_mapper is None:
            dataset_mapper = build_dataset_mapper(cfg, name=cfg.DATASETS.TRAIN, stage="train")
        if datalist is None:
            datalist = dataset_mapper.load_data(cfg)
    else:
        dataset_mapper = None
        datalist = None
    return datalist, dataset_mapper


def _train_loader_from_config(cfg,
                              dataset_mapper=None,
                              *,
                              datalist=None,
                              sampler=None,
                              model=None):
    # xiaoshi: mscoco image captioning: called from defaulttainer, only cfg passed
    datalist, dataset_mapper = _single_modal_dataset(
        cfg, dataset_mapper=dataset_mapper, datalist=datalist, sampler=sampler)

    return {
        "datalist": datalist,
        "dataset_mapper": dataset_mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "batch_size": cfg.DATALOADER.TRAIN_BATCH_SIZE,
        "cfg": cfg,
        "model": model,
    }



def _valtest_loader_from_config(cfg, dataset_mapper=None, *, datalist=None, sampler=None, stage="val"):
    dataset_names = {
        "val": cfg.DATASETS.VAL,
        "test": cfg.DATASETS.TEST,
    }
    dataset_name = dataset_names[stage]
    if len(dataset_name) > 0:
        if dataset_mapper is None:
            dataset_mapper = build_dataset_mapper(cfg, name=dataset_name, stage=stage)
        if datalist is None:
            datalist = dataset_mapper.load_data(cfg)
    else:
        dataset_mapper = None
        datalist = None

    if dataset_name in ['Flickr30kDatasetForSingleStreamVal', 'Flickr30kDatasetForSingleStreamValV2']:
        multi_gpu_eval = True
        batch_size = 1
    else:
        multi_gpu_eval = False
        batch_size = cfg.DATALOADER.TEST_BATCH_SIZE

    return {
        "datalist": datalist,
        "dataset_mapper": dataset_mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "batch_size": batch_size,
        "multi_gpu_eval": multi_gpu_eval,
        "cfg": cfg,
        "stage": stage
    }


def build_standard_valtest_loader(cfg, task_cfg, stage, multi_gpu_eval):
    dataset_names = {
        "val": cfg.DATASETS.VAL,
        "test": cfg.DATASETS.TEST,
    }
    dataset_name = dataset_names[stage]
    if len(dataset_name) > 0:
        dataset = build_dataset_mapper(cfg, name=dataset_name, stage=stage)
    else:
        return None

    shared_targets = load_shared_targets(cfg, stage=stage)

    if multi_gpu_eval and get_world_size() > 1:
        # multi-gpu-eval for single stream retrieval
        sampler = DistributedSampler(dataset, shuffle=True)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.DATALOADER.TEST_BATCH_SIZE,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=False,
            sampler=sampler,
            collate_fn=partial(preprocess_batch_collator, shared_targets=shared_targets, cfg=cfg),
            pin_memory=cfg.DATALOADER.PIN_MEM,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.DATALOADER.TEST_BATCH_SIZE,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=False,
            shuffle=False,
            collate_fn=partial(preprocess_batch_collator, shared_targets=shared_targets, cfg=cfg),
            pin_memory=cfg.DATALOADER.PIN_MEM,
        )
    return data_loader