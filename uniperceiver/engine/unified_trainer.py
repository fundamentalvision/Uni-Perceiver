import time
import tqdm
import os
import json
import pickle
import sys
import copy
import numpy as np
import itertools
import random
import torch
import io
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast
from .unified_tester import tester, dict_to_cuda, list_to_cuda, move_to_cuda
from collections import OrderedDict
from uniperceiver.evaluation import build_evaluation
import uniperceiver.utils.comm as comm
from uniperceiver.utils.engine_util import *
from .build import ENGINE_REGISTRY
from uniperceiver.datasets import (
    build_standard_valtest_loader,
    build_unified_train_loader,
)

from uniperceiver.utils.events import get_event_storage
from uniperceiver.utils.events import EventStorage
from omegaconf import DictConfig
from uniperceiver.losses import build_losses
from uniperceiver.optim import build_optimizer
from uniperceiver.modeling import build_model
from uniperceiver.lr_scheduler import build_lr_scheduler
from torch.cuda.amp import autocast
from uniperceiver.checkpoint import TorchCheckpointer

import logging
import math
import weakref

from uniperceiver.config import  CfgNode


from . import hooks


from timm.data import Mixup
from timm.utils import ModelEma
from uniperceiver.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from uniperceiver.utils.misc import ApexScalerWithGradNormCount as ApexScaler

from collections import defaultdict
from .train_loop import TrainerBase
from uniperceiver.utils.logger import setup_logger

try:
    from apex import amp
    APEX_INSTALLED = True
except:
    print('apex has not been installed.')
    APEX_INSTALLED = False

__all__ = ['UnifiedTrainer']


@ENGINE_REGISTRY.register()
class UnifiedTrainer(TrainerBase):
    def __init__(self, cfg):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        if not self.logger.isEnabledFor(
                logging.INFO):  # setup_logger is not called for d2
            setup_logger()

        self.task_cfg = dict()
        self.task_names = []
        for task in cfg.TASKS:
            name = task['NAME']
            self.task_names.append(name)

            # self.task_cfg[name] = new_cfg
            self.task_cfg[name] = CfgNode(task)

        self.cfg = cfg

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        self.logger.info("Model Creation Done")

        self.apex_need_reload = False

        self.optimizer = self.build_optimizer(cfg, model)

        if cfg.SOLVER.APEX_FP16 and APEX_INSTALLED:
            self.apex_fp16 = True

            model, self.optimizer = amp.initialize(model,
                                                   self.optimizer,
                                                   opt_level=self.cfg.SOLVER.APEX_OPT_LEVEL,
                                                   master_weights=self.cfg.SOLVER.APEX_MASTER_WEIGHTS,
                                                   min_loss_scale=self.cfg.SOLVER.MIN_LOSS_SCLE,
                                                   loss_scale="dynamic")

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model,
                find_unused_parameters=cfg.find_unused_parameters,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False)
        self.model = model


        self.model.train()

        self.train_data_loader = build_train_loader(cfg, self.task_cfg, self.model)
        self.val_data_loader = build_val_loader(cfg, self.task_cfg)
        self.test_data_loader = build_test_loader(cfg, self.task_cfg)

        if isinstance(self.train_data_loader, list):
            self.iters_per_epoch_list = [
                len(loader) for loader in self.train_data_loader
            ]
            self._train_data_loader_iter_list = [
                iter(loader) for loader in self.train_data_loader
            ]

            self.iters_per_epoch = len(self.train_data_loader[0])
            self._train_data_loader_iter = iter(self.train_data_loader[0])
        else:
            self.iters_per_epoch = len(self.train_data_loader)
            self._train_data_loader_iter = iter(self.train_data_loader)

        if self.val_data_loader is not None:
            self.val_evaluator = build_evaluation(cfg,
                                                  cfg.INFERENCE.VAL_ANNFILE,
                                                  None)
        else:
            self.val_evaluator = None

        if self.test_data_loader is not None:
            self.test_evaluator = build_evaluation(cfg,
                                                   cfg.INFERENCE.TEST_ANNFILE,
                                                   cfg.OUTPUT_DIR)
        else:
            self.test_evaluator = None

        self.ss_prob = 0.0


        self.model_ema = None
        if cfg.MODEL.MODEL_EMA:
            self.model_ema = ModelEma(
                self.model,
                decay=cfg.MODEL.MODEL_EMA_DECAY,
                device='cpu' if cfg.MODEL.MODEL_EMA_FORCE_CPU else '',
                resume='')

        self.checkpointer = TorchCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            self.model_ema,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
            checkpoint_mapping=cfg.SOLVER.CHECKPOINT_MAPPING,
            mapping=cfg.SOLVER.CHECKPOINT_MAP,
            resume_tau=cfg.SOLVER.RESUME_TAU,
            ceph_save=cfg.SOLVER.CHECKPOINT_CEPH_SAVE,
            ceph_config=cfg.DATALOADER.get("TCS_CONF_PATH",
                                            "petreloss.config"),
        )
        self.checkpointer.add_checkpointable('optimizer', self.optimizer)

        if cfg.MODEL.MODEL_EMA:
            self.checkpointer.add_checkpointable('ema_model',self.model_ema.ema)

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.EPOCH * self.iters_per_epoch
        self.register_hooks(self.build_hooks())

        if cfg.SOLVER.AMP_FP16:
            # Creates a GradScaler once at the beginning of training.
            self.amp_scaler = NativeScaler(enabled=True, growth_interval=cfg.SOLVER.LOSS_SCALE_WINDOW)
            self.amp_fp16=True
        else:
            self.amp_scaler = NativeScaler(enabled=False)
            self.amp_fp16=False

        if cfg.SOLVER.APEX_FP16 and APEX_INSTALLED:

            self.amp_scaler = ApexScaler(enabled=True)

        else:
            self.apex_fp16 = False

        self.fp16 = cfg.SOLVER.AMP_FP16 or cfg.SOLVER.APEX_FP16
        self.bf16 = cfg.SOLVER.BF16
        if self.fp16:
            assert not self.bf16

        if self.amp_scaler is not None:
            self.checkpointer.add_checkpointable('amp_scaler', self.amp_scaler)


        self.val_evaluator = dict()
        self.test_evaluator = dict()
        self.mixup_fn = dict()
        for name, new_cfg in self.task_cfg.items():
            if self.val_data_loader[name]:
                self.val_evaluator[name] = build_evaluation(
                    new_cfg, new_cfg.INFERENCE.VAL_ANNFILE, cfg.OUTPUT_DIR)
            else:
                self.val_evaluator[name] = None
            if self.test_data_loader[name]:
                self.test_evaluator[name] = build_evaluation(new_cfg, new_cfg.INFERENCE.TEST_ANNFILE, cfg.OUTPUT_DIR)
            else:
                self.test_evaluator[name] = None

            if new_cfg.DATALOADER.MIXUP > 0 or new_cfg.DATALOADER.CUTMIX > 0:
                self.mixup_fn[name] = Mixup(
                    mixup_alpha=new_cfg.DATALOADER.MIXUP, cutmix_alpha=new_cfg.DATALOADER.CUTMIX, cutmix_minmax=None,
                    prob=new_cfg.DATALOADER.MIXUP_PROB, switch_prob=new_cfg.DATALOADER.MIXUP_SWITCH_PROB, mode=new_cfg.DATALOADER.MIXUP_MODE,
                    label_smoothing=new_cfg.DATALOADER.MIXUP_LABEL_SMOOTHING, num_classes=new_cfg.MODEL.LABELS_NUM)
            else:
                self.mixup_fn[name] = None

        if cfg.DATALOADER.USE_WEIGHTED_SAMPLER:
            # this is to avoid strange behaviors.
            self.iters_per_epoch = 1
            # override the previous scheduler

        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer, self.iters_per_epoch)
        self.checkpointer.add_checkpointable('scheduler', self.scheduler)

        self.accum_iter  = max(1, cfg.SOLVER.ACCUM_ITER)
        self.step_index = 0

        self.grad_print = getattr(cfg.SOLVER, "GRAD_PRINT", False)

        if self.cfg.SOLVER.GradHistogram:
            assert self.cfg.SOLVER.TORCH_OPTIMIZER and self.cfg.SOLVER.PARAMS_SEPERATE

    def resume_or_load(self, resume=True):

        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS,
                                                resume=resume,
                                                resume_optmizer=self.cfg.SOLVER.RESUME_OPTIMIZER)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = self.iter + 1
            # make apex resume work
            if self.apex_fp16:
                self.apex_need_reload = True

    @classmethod
    def build_losses(cls, cfg):
        losses = {}
        for task_config in cfg.TASKS:
            task_config = DictConfig(task_config)
            losses[task_config.NAME] = build_losses(task_config)

        return losses

    def build_hooks(self):

        self.max_iter = self.cfg.SOLVER.MAX_ITER
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.ModelWeightsManipulating()
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD,
                                                    max_to_keep= cfg.SOLVER.CHECKPOINT_MAX_SAVE ))

        def test_and_save_results(epoch):
            eval_results = self.test(self.cfg, self.model, self.test_data_loader, self.test_evaluator, epoch)
            return eval_results

        def val_and_save_results(epoch):
            eval_results = self.test(self.cfg, self.model, self.val_data_loader, self.val_evaluator, epoch)
            return eval_results

        if self.model_ema is not None:

            def test_and_save_results_ema(epoch):
                eval_results = self.test(self.cfg, self.model_ema.ema,
                                            self.test_data_loader,
                                            self.test_evaluator, epoch)
                ema_results = {}
                for taskname, taskresults in eval_results.items():
                    if isinstance(taskresults, dict):
                        taskresults = {
                            f'{k}_ema': v
                            for k, v in taskresults.items()
                        }
                    ema_results[taskname] = taskresults

                return ema_results

            def val_and_save_results_ema(epoch):
                eval_results = self.test(self.cfg, self.model_ema.ema,
                                            self.val_data_loader,
                                            self.val_evaluator, epoch)
                ema_results = {}
                for taskname, taskresults in eval_results.items():
                    if isinstance(taskresults, dict):
                        taskresults = {f'{k}_ema': v for k, v in taskresults.items()}
                    ema_results[taskname] = taskresults

                return ema_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        if self.val_data_loader is not None:
            ret.append(
                hooks.IterEvalHook(
                    eval_period = cfg.SOLVER.EVAL_PERIOD,
                    eval_start = cfg.INFERENCE.VAL_EVAL_START,
                    eval_function = val_and_save_results,
                    stage = 'val',
                    multi_gpu_eval=True
                ))
            if self.model_ema is not None:
                ret.append(
                hooks.IterEvalHook(
                    eval_period = cfg.SOLVER.EVAL_PERIOD,
                    eval_start = cfg.INFERENCE.VAL_EVAL_START,
                    eval_function = val_and_save_results_ema,
                    stage = 'val',
                    multi_gpu_eval=True
                ))

        if self.test_data_loader is not None:
            ret.append(
                hooks.IterEvalHook(
                    eval_period = cfg.SOLVER.EVAL_PERIOD,
                    eval_start = cfg.INFERENCE.TEST_EVAL_START,
                    eval_function = test_and_save_results,
                    stage = 'test',
                    multi_gpu_eval=True
                ))
            if self.model_ema is not None:
                ret.append(
                    hooks.IterEvalHook(
                        eval_period=cfg.SOLVER.EVAL_PERIOD,
                        eval_start=cfg.INFERENCE.TEST_EVAL_START,
                        eval_function=test_and_save_results_ema,
                        stage='test',
                        multi_gpu_eval=True))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(build_writers(cfg, self.max_iter), period=cfg.SOLVER.WRITE_PERIOD))

        return ret

    def train(self):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        start_iter = self.start_iter
        max_iter = self.max_iter
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:

                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()

                    self.run_step_torch()

                    self.after_step()
                    
                    if self.apex_need_reload:
                        optimizer_state_dict = torch.load(self.checkpointer.get_checkpoint_file())['optimizer']
                        self.optimizer.load_state_dict(optimizer_state_dict)
                        self.apex_need_reload = False

                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        logger = logging.getLogger(__name__)
        logger.info("building optimizer...")
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer, iters_per_epoch):
        logger = logging.getLogger(__name__)
        logger.info("building lr_scheduler...")
        return build_lr_scheduler(cfg, optimizer, iters_per_epoch)

    def run_step_torch(self):
        if self.accum_iter > 1:
            for micro_step in range(self.accum_iter):
                self.micro_step = micro_step
                self.run_min_batch()
        else:
            self.micro_step = 0
            self.run_min_batch()

    def run_min_batch(self):
        timer_fn = time.perf_counter
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        torch.cuda.synchronize()

        start = timer_fn()
        data = get_batch_data(self.cfg, self._train_data_loader_iter, self.train_data_loader)
        data_time = time.perf_counter() - start

        task = data['task_info']['task_name']
        data = move_to_cuda(data)

        #TODO: task specifix code, move into model
        if self.mixup_fn[task] is not None:
            # imagenet
            data['input_sample_list'][0]["data"], data[
                'target_idx_list'][0] = self.mixup_fn[task](
                    data['input_sample_list'][0]["data"], data["target_idx_list"][0])

        if not self.amp_fp16:
            losses_dict = self.model(data)

        else:
            with autocast(self.amp_fp16):
                losses_dict = self.model(data)

        losses = sum(losses_dict.values())

        # for accum iter
        losses /= self.accum_iter

        total_grad = self.amp_scaler(losses, self.optimizer, clip_grad=self.cfg.SOLVER.GRAD_CLIP,
                    parameters=self.model.parameters(), create_graph=False,
                    update_grad=(self.micro_step + 1 == self.accum_iter), fp16=self.fp16, iter=self.iter,
                    min_loss_scale=self.cfg.SOLVER.MIN_LOSS_SCLE,
                    loss_scale_window=self.cfg.SOLVER.LOSS_SCALE_WINDOW)

        if  self.micro_step + 1 != self.accum_iter:
            return

        if self.micro_step + 1 == self.accum_iter:
            write_metrics(losses_dict, data_time, task + '/')

        if comm.is_main_process():
            storage = get_event_storage()
            if torch.logical_or(total_grad.isnan(), total_grad.isinf()):
                logger = logging.getLogger(__name__)
                logger.info('grad to nan or inf in task {} {}'.format(task, total_grad))
            storage.put_scalar("total_grad", total_grad, smoothing_hint=False)

        if self.apex_need_reload:
            pass
        else:
            self.amp_scaler.step(self.optimizer)

        if comm.is_main_process():
            storage.put_scalar("amp_scale", self.amp_scaler.get_scale(), smoothing_hint=False)
            if hasattr(comm.unwrap_model(self.model).loss_prepare, 'temperature_dict'):
                if isinstance(comm.unwrap_model(self.model).loss_prepare, torch.nn.ModuleList):
                    temperature_dict = comm.unwrap_model(self.model).loss_prepare[-1].temperature_dict
                else:
                    temperature_dict = comm.unwrap_model(self.model).loss_prepare.temperature_dict
                storage.put_scalars(**temperature_dict, smoothing_hint=False)

        if self.amp_fp16:
            self.amp_scaler.update()


        self.optimizer.zero_grad()
        if self.model_ema is not None:
            self.model_ema.update(self.model)
        torch.cuda.synchronize()

    def cast_layers(self):
        logger = self.logger
        if self.cfg.MODEL.LN_FP32:
            logger.info("cast LN to fp32")

            def cast_ln_fp32(module):
                if isinstance(module, CustomLayernorm):
                    module.float()

            self.model_engine.module.apply(cast_ln_fp32)

        if self.iter == 0:
            comm.unwrap_model(self.model).operatedweight()



    def test(self, cfg, model, test_data_loader, evaluator, epoch):
        return tester(self.task_cfg, model, test_data_loader, evaluator, epoch, self.amp_fp16, self.apex_fp16)
