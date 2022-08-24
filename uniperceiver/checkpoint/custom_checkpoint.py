# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import pickle
import torch
# from typing import Any
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer, _IncompatibleKeys
from fvcore.common.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from torch.nn.parallel import DistributedDataParallel

import uniperceiver.utils.comm as comm
from uniperceiver.utils.env import TORCH_VERSION
from uniperceiver.utils.file_io import PathManager
from collections import defaultdict
import copy
import io

from .c2_model_loading import align_and_update_state_dicts
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

# import deepspeed
# from deepspeed.runtime.engine import DeepSpeedEngine
import shutil
from timm.utils import ModelEma


class PeriodicEpochCheckpointer(PeriodicCheckpointer):
    def step(self, iteration: int, epoch: int, **kwargs: Any) -> None:
        """
        Perform the appropriate action at the given iteration.

        Args:
            iteration (int): the current iteration, ranged in [0, max_iter-1].
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        iteration = int(iteration)
        epoch = int(epoch)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)

        if (iteration + 1) % self.period == 0:

            self.checkpointer.save(
                "{}_Epoch_{:05d}_Iter_{:07d}".format(self.file_prefix, epoch,
                                                        iteration),
                **additional_state)

            if self.max_to_keep is not None:
                self.recent_checkpoints.append(
                    self.checkpointer.get_checkpoint_file())
                # pyre-fixme[58]: `>` is not supported for operand types `int` and
                #  `Optional[int]`.
                if len(self.recent_checkpoints) > self.max_to_keep:
                    file_to_delete = self.recent_checkpoints.pop(0)
                    if self.path_manager.exists(
                            file_to_delete) and not file_to_delete.endswith(
                                f"{self.file_prefix}_final.pth"):
                        self.path_manager.rm(file_to_delete)



class TorchCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to handle models in uniperceiver
    model zoo, and apply conversions for legacy models.
    """
    def __init__(
        self,
        model,
        model_ema: ModelEma,
        save_dir="",
        *,
        save_to_disk=None,
        checkpoint_mapping=None,
        mapping=False,
        resume_tau=True,
        ceph_save=False,
        ceph_config=None,
        **checkpointables,
    ):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process
            if save_to_disk is None else save_to_disk,
            **checkpointables,
        )
        self.path_manager = PathManager

        if checkpoint_mapping is None:
            self.checkpoint_mapping = None
        else:
            self.checkpoint_mapping = defaultdict(list)
            for mapping_pair in checkpoint_mapping:
                self.checkpoint_mapping[mapping_pair['ORIGIN']].append(
                    mapping_pair['DEST'])
        self.mapping = mapping
        self.resume_tau = resume_tau
        self.ceph_save = ceph_save
        if self.ceph_save:
            self.path_prefix = 's3://checkpoints_zjg/'
            self.client = PetrelBackend(path_mapping={},
                                        tcs_conf_path=ceph_config)
        # if self.ceph_save and is_main_process:
        # # for local machine debug
        # if os.path.relpath(self.save_dir, os.getcwd()).startswith('outputs'):
        #     self.client.remove(self.save_dir)

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(
                    data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {
                    k: v
                    for k, v in data.items() if not k.endswith("_momentum")
                }
                return {
                    "model": data,
                    "__author__": "Caffe2",
                    "matching_heuristics": True
                }
        if self.ceph_save:
            relpath = os.path.relpath(filename, os.getcwd())
            s3url = os.path.join(self.path_prefix, relpath)
            with io.BytesIO(self.client.get(s3url)) as buffer:
                loaded = torch.load(buffer, map_location=torch.device("cpu"))
        else:
            loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded

    def save(self, name: str, **kwargs: Any) -> None:
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)

        if self.ceph_save:
            local_save_file = os.path.join(self.save_dir, basename)
            relpath = os.path.relpath(local_save_file, os.getcwd())
            save_file = os.path.join(self.path_prefix, relpath)
            assert os.path.basename(save_file) == basename, basename
            self.logger.info("Saving checkpoint to {}".format(save_file))
            with io.BytesIO() as f:
                torch.save(data, f)
                self.client.put(f.getvalue(), save_file)
        else:
            save_file = os.path.join(self.save_dir, basename)
            assert os.path.basename(save_file) == basename, basename
            self.logger.info("Saving checkpoint to {}".format(save_file))
            with self.path_manager.open(save_file, "wb") as f:
                torch.save(data, f)
        self.tag_last_checkpoint(basename)

    def load(self,
             path: str,
             checkpointables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load from the given checkpoint.

        Args:
            path (str): path or url to the checkpoint. If empty, will not load
                anything.
            checkpointables (list): List of checkpointable names to load. If not
                specified (None), will load all the possible checkpointables.
        Returns:
            dict:
                extra data loaded from the checkpoint that has not been
                processed. For example, those saved with
                :meth:`.save(**extra_data)`.
        """
        if not path:
            # no checkpoint provided
            self.logger.info(
                "No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("[Checkpointer] Loading from {} ...".format(path))
        if not self.ceph_save:
            if not os.path.isfile(path):
                path = self.path_manager.get_local_path(path)
                assert os.path.isfile(path), "Checkpoint {} not found!".format(
                    path)
        else:
            relpath = os.path.relpath(path, os.getcwd())
            s3url = os.path.join(self.path_prefix, relpath)
            #TODO:  dev branch is needed
            # if not self.client.exists(s3url):
            #     assert self.client.exists(s3url), "Checkpoint {} not found!".format(s3url)

        checkpoint = self._load_file(path)
        incompatible = self._load_model(checkpoint)
        if (incompatible is not None
            ):  # handle some existing subclasses that returns None
            self._log_incompatible_keys(incompatible)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:
                self.logger.info("Loading {} from {} ...".format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))

        # return any further checkpoint data
        return checkpoint

    def _convert_checkpoint(self, checkpoint):
        # for multitask pretrain and fintune
        if self.checkpoint_mapping is not None and self.mapping:
            pretrain_checkpoint = checkpoint["model"]
            for origin_task in self.checkpoint_mapping.keys():
                for k in list(pretrain_checkpoint.keys()):
                    if origin_task in k:
                        # mapping to downstrean task
                        state_dict_temp = copy.deepcopy(
                            pretrain_checkpoint.pop(k))
                        for subtask in self.checkpoint_mapping[origin_task]:
                            new_key = k.replace(origin_task, subtask)
                            pretrain_checkpoint[new_key] = state_dict_temp
            checkpoint["model"] = pretrain_checkpoint

        if not self.resume_tau:
            pretrain_checkpoint = checkpoint["model"]
            for k in list(pretrain_checkpoint.keys()):
                if "logit_scale" in k:
                    pretrain_checkpoint.pop(k)
            checkpoint["model"] = pretrain_checkpoint
        return checkpoint

    def _load_model(self, checkpoint):
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            model_state_dict = self.model.state_dict()
            align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            checkpoint["model"] = model_state_dict

        # convert checkpoint for pretrained model between different tasks
        checkpoint = self._convert_checkpoint(checkpoint)

        # for non-caffe2 models, use standard ways to load it
        incompatible = super()._load_model(checkpoint)
        if incompatible is None:  # support older versions of fvcore
            return None

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            # Ignore missing key message about pixel_mean/std.
            # Though they may be missing in old checkpoints, they will be correctly
            # initialized from config anyway.
            if k in model_buffers:
                try:
                    incompatible.missing_keys.remove(k)
                except ValueError:
                    pass
        return incompatible

    def _log_incompatible_keys(self, incompatible: _IncompatibleKeys) -> None:
        """
        Log information about the incompatible keys returned by ``_load_model``.
        """
        for k, shape_checkpoint, shape_model in incompatible.incorrect_shapes:
            self.logger.warning(
                "Skip loading parameter '{}' to the model due to incompatible "
                "shapes: {} in the checkpoint but {} in the "
                "model! You might want to double check if this is expected.".
                format(k, shape_checkpoint, shape_model))
        if incompatible.missing_keys:
            self.logger.info(
                get_missing_parameters_message(incompatible.missing_keys))
        if incompatible.unexpected_keys:
            self.logger.info(
                get_unexpected_parameters_message(
                    incompatible.unexpected_keys))

    def resume_or_load(self, path, resume: bool = True, **kwargs):
        super().resume_or_load(path, resume=resume)
