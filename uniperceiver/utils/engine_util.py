import uniperceiver.utils.comm as comm
import torch
import numpy as np
from uniperceiver.utils.events import get_event_storage
from typing import Dict
from uniperceiver.datasets import (
    build_standard_valtest_loader,
    build_unified_train_loader,
)
import weakref

def write_metrics(loss_dict: Dict[str, torch.Tensor],
                    data_time: float,
                    prefix: str = "",
                    ):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        metrics_dict = {}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                metrics_dict.update({k: v.detach().cpu().item()})
            else:
                metrics_dict.update({k: v})
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = [metrics_dict]
        if comm.is_main_process():
            # print(all_metrics_dict)
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)
            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            storage.put_scalar("{}total_loss".format(prefix),
                               total_losses_reduced)
            if len(metrics_dict) > 1:
                for k, v in metrics_dict.items():
                    if k != 'null_loss':
                        storage.put_scalar(f'{prefix}{k}', v)

def build_writers(cfg, max_iter):
        from uniperceiver.engine.defaults import default_writers
        return default_writers(cfg.OUTPUT_DIR, max_iter)

def build_train_loader(cfg, task_cfg, model):
    loader = dict()
    if cfg.DATALOADER.UNIFIED_DATASET:
        loader = build_unified_train_loader(cfg, task_cfg, model=weakref.proxy(comm.unwrap_model(model)) if cfg.DATALOADER.LOAD_INLABEL else None)
        return loader
    else:
        raise NotImplementedError('please use unified dataset.')

def build_test_loader(cfg, task_cfg):
        loaders = dict()
        #TODO: move multi-gpu eval in config file
        for name, new_cfg in task_cfg.items():
            multi_gpu = name in [
                'K400_retrieve', 'imagenet', 'vqa', 'mscoco_caption',
                'flickr30k_caption', 'K700_retrieve', 'imagenet_caption'
            ]
            loaders[name] = build_standard_valtest_loader(new_cfg, task_cfg, stage='test', multi_gpu_eval=multi_gpu)
        return loaders

def build_val_loader(cfg, task_cfg):
    loaders = dict()
    for name, new_cfg in task_cfg.items():
        #TODO: move multi-gpu eval in config file
        multi_gpu = name in [
            'K400_retrieve', 'imagenet', 'vqa', 'mscoco_caption',
            'flickr30k_caption', 'K700_retrieve', 'imagenet_caption'
        ]
        loaders[name] = build_standard_valtest_loader(new_cfg, task_cfg, stage='val', multi_gpu_eval=multi_gpu)
    return loaders

def get_batch_data(cfg, train_data_loader_iter, train_data_loader):
    if not cfg.DATALOADER.FAKE_DATA:
        try:
            data = next(train_data_loader_iter)
        except StopIteration:
            train_data_loader_iter = iter(train_data_loader)
            data = next(train_data_loader_iter)
    else:
        # fake data
        bs = 32
    return data
