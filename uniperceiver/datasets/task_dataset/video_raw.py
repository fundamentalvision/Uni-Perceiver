import os
import random
import numpy as np
import torch
import pickle
from PIL import Image
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.transforms import Compose, RandomApply, ToTensor, Normalize, CenterCrop, Lambda, RandomHorizontalFlip, ColorJitter, Resize, RandomCrop
import json
import av
from torchvision.transforms.transforms import RandomResizedCrop
from uniperceiver.tokenization import  ClipTokenizer
from uniperceiver.config import  configurable
from ..build import DATASETS_REGISTRY
from uniperceiver.functional import dict_as_tensor
from .video_transform import random_short_side_scale_jitter, uniform_crop
import pyarrow as pa
from uniperceiver.utils import comm
import copy

import io

__all__ = ["VideoDataSet", "random_clip"]


def load_pkl_file(filepath):
    return pickle.load(open(filepath, 'rb'), encoding='bytes') if len(filepath) > 0 else None


def random_clip(video_frames, sampling_rate, frames_per_clip, fixed_offset=False):
    """
    Args:
        video_frames (int): total frame number of a video
        sampling_rate (int): sampling rate for clip, pick one every k frames
        frames_per_clip (int): number of frames of a clip
        fixed_offset (bool): used with sample offset to decide the offset value deterministically.
    Returns:
        list[int]: frame indices (started from zero)
    """
    new_sampling_rate = sampling_rate
    highest_idx = video_frames - int(new_sampling_rate * (frames_per_clip - 1) + 1)
    if highest_idx <= 0:
        random_offset = 0
    else:
        if fixed_offset:
            random_offset = (video_frames - int(new_sampling_rate * frames_per_clip)) // 2
        else:
            random_offset = int(np.random.randint(0, highest_idx, 1))
    frame_idx = [int(random_offset + int(i * sampling_rate)) % video_frames for i in range(frames_per_clip)]
    frame_idx = [x for x in frame_idx if x < video_frames]
    return frame_idx

@DATASETS_REGISTRY.register()
class VideoDataSet(data.Dataset):

    @configurable
    def __init__(self, cfg, stage, root_path, s3_path, list_file, category_file, use_ceph,  tcs_conf_path,
                 tokenizer, tokenizer_name, data_percentage,
                 frames_per_clip=64, interval=4, num_clips=1,
                 is_train=True, test_mode=False, num_classes=None, target_fps=30, timesformer_aug=False, minibatches=1):
        """
        Args:
            root_path (str): the file path to the root of video folder
            list_file (str): the file list, each line with folder_path, start_frame, end_frame, label_id
            frames_per_clip (int): number of frames per data sample
            interval (int): interval between frames
            is_train (bool): shuffle the video but keep the causality
            test_mode (bool): testing mode, no label
        """

        self.cfg = cfg
        self.stage = stage
        self.root_path = root_path
        self.s3_path = s3_path
        self.list_file = list_file
        self.category_file = category_file
        self.frames_per_clip = frames_per_clip
        self.interval = interval
        self.num_clips = num_clips
        self.is_train = is_train
        self.test_mode = test_mode
        self.num_classes = num_classes
        self.target_fps = target_fps
        self.minibatches = minibatches
        self.data_percentage = data_percentage

        # self.class_names = class_names if (class_names is not None) else None
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name

        self.transform = self._timesformer_transform() if timesformer_aug else self._transform()

        self.use_ceph = use_ceph
        if self.use_ceph:
            # get dataset
            # dataset_name = self.root_path.split('/')[-2]
            self.data_path = self.s3_path
            print('debug info for {} {} '.format(self.cfg.DATASETS.DATASET_NAME, self.data_path))
            from uniperceiver.datasets import TCSLoader

            self.tcs_loader = TCSLoader(tcs_conf_path)
        else:
            self.data_path = self.root_path

        _temp_list =self.load_data(self.cfg)
        self.video_list = pa.array(_temp_list)
        if comm.is_main_process():
            import sys
            print(f"!!! Dataset {self.cfg.DATASETS.DATASET_NAME} with task {self.cfg.DATASETS.TASK_TYPE}:")
            print('!!! length of _temp_list: ', len(_temp_list))
            print('!!! size of _temp_list: ', sys.getsizeof(_temp_list))
            print('!!! size of pa database: ', sys.getsizeof(self.video_list))
        del _temp_list

        self.testing_multi_view = self.cfg.DATALOADER.get('MULTI_VEIW', 'v0')
        self.temporal_num_view = self.cfg.DATALOADER.get('MULTI_VEIW_NUM', 1)

        self.random_stride = self.cfg.DATALOADER.get('RANDON_STRIDE', False)

        if self.test_mode:
            self.frames_per_clip = int(self.frames_per_clip*self.temporal_num_view)
            self.interval = int(self.interval/self.temporal_num_view)

        self.task_info = {
                'task_type'      : self.cfg.DATASETS.TASK_TYPE,
                'dataset_name'   : self.cfg.DATASETS.DATASET_NAME,
                'batch_size'     : self.cfg.DATALOADER.TRAIN_BATCH_SIZE if self.stage == 'train' else self.cfg.DATALOADER.TEST_BATCH_SIZE,
                'sampling_weight': self.cfg.DATALOADER.SAMPLING_WEIGHT,

            }

        self.target_set = self.cfg.DATASETS.TARGET_SET


    def _transform(self):
        assert False, 'use timesformer augmentation'
        transforms = [
            Lambda(lambda frames: torch.stack([ToTensor()(frame.convert("RGB")) for frame in frames])),
        ]
        if self.test_mode:
            transforms.extend([
                RandomResizedCrop(224, scale=(0.75, 0.75), ratio=(1.0, 1.0)),
                # CenterCrop(224)
                # RandomApply(torch.nn.ModuleList([ColorJitter(0.4, 0.4, 0.4)]), 0.8),
            ])
        else:
            transforms.extend([
                # scale jitter as in vivit: (0.9, 1.33)
                RandomResizedCrop(224, scale=(0.56, 0.95), ratio=(1.0, 1.0)),
                RandomHorizontalFlip(),
                # only p=0.8 is specified in vivit paper, using deit default parameters
                RandomApply(torch.nn.ModuleList([ColorJitter(0.4, 0.4, 0.4)]), 0.8),
            ])
        transforms.append(
            # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            # change to imagenet default value to keep consistency with pretrained parameters
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        return Compose(transforms)

    def _timesformer_transform(self):
        transforms = [
            Lambda(lambda frames: torch.stack([ToTensor()(frame.convert("RGB")) for frame in frames])),
        ]
        if self.test_mode:
            test_scale = self.cfg.MODEL.IMG_INPUT_SIZE
            transforms.extend([
                Lambda(lambda frames: random_short_side_scale_jitter(frames, test_scale, test_scale)[0]),
                Lambda(lambda images: torch.stack([uniform_crop(images, test_scale, i)[0] for i in range(3)], 0))
            ])
        else:
            min_scale = int((256 / 224)*self.cfg.MODEL.IMG_INPUT_SIZE)
            max_scale = int((320 / 224)*self.cfg.MODEL.IMG_INPUT_SIZE)
            transforms.extend([
                # Lambda(lambda frames: random_short_side_scale_jitter(frames, 256, 320)[0].unsqueeze(0)),
                Lambda(lambda frames: random_short_side_scale_jitter(frames, min_scale, max_scale)[0].unsqueeze(0)),
                RandomHorizontalFlip(),
                RandomCrop(self.cfg.MODEL.IMG_INPUT_SIZE)
            ])
        transforms.append(
            # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            # change to imagenet default value to keep consistency with pretrained parameters
            # Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        )
        return Compose(transforms)


    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        if 'SLURM_PROCID' in os.environ:
            tcs_conf_path = cfg.DATALOADER.get("TCS_CONF_PATH", "petreloss.config")
        else:
            # dev machine
            tcs_conf_path = "slurm_tools/petreloss_local.config"
        ret = {
            "cfg": cfg,
            "stage": stage,
            "list_file": os.path.join(cfg.DATALOADER.ANNO_FOLDER, cfg.DATALOADER.ANNO_FILE),
            "category_file": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "category_mapping.txt"),
            "root_path": os.path.join(cfg.DATALOADER.FEATS_FOLDER, "training" if stage == "train" else "validation"),
            "s3_path": os.path.join(cfg.DATALOADER.S3_PATH, "training" if stage == "train" else "validation"),
            "frames_per_clip": cfg.DATALOADER.FRAMES_PER_CLIP,
            "interval": cfg.DATALOADER.STRIDE,
            "num_clips": 1 if stage == 'train' else cfg.INFERENCE.NUM_VIEWS,
            "is_train": stage == 'train',
            "test_mode": stage != 'train',
            "num_classes": cfg.MODEL.NUM_CLASSES,
            "timesformer_aug": cfg.DATALOADER.TIMESFORMER_AUG,
            "minibatches": cfg.DATALOADER.MINI_BATCHES,
            "use_ceph": getattr(cfg.DATALOADER, 'USE_CEPH', False),
            "tcs_conf_path": tcs_conf_path,
            "data_percentage": cfg.DATALOADER.DATA_PERCENTAGE,
        }


        ret['tokenizer'] = ClipTokenizer()
        ret['tokenizer_name']  = "clip"


        return ret

    def load_data(self, cfg):
        # usualy it is [video_id, num_frames, class_idx]
        # or [video_id, start_frame, end_frame, list of class_idx]
        self.cls2idx = dict()
        self.idx2cls = dict()
        self.class_names = list()
        with open(self.category_file, 'r') as f:
            for line in f.readlines():
                class_name, idx = line.strip().split('\t')
                # for annotations
                class_name = class_name.replace(" ", "_") #  replace(" ", "_") for kinetics dataset
                self.cls2idx[class_name] = int(idx)
                self.idx2cls[int(idx)] = class_name
                # processed_name = class_name.replace("_", " ").lower()
                # if cfg.NAME in ["K700", "K400"]:
                #     processed_name = processed_name.replace("american football", "football").replace("(", "").replace(")", "")
                # self.class_names.append(processed_name)

        # self.class_name_tokens = [np.array(self.tokenizer.encode(x + " <|endoftext|>"), dtype=np.int64) for x in self.class_names]

        # self.class_name_type_tokens = [np.zeros(len(x), dtype=np.int64) for x in self.class_name_tokens]

        # load the exclude list
        # TODO: move this to the config file
        exclude_list = list()
        if os.path.exists(os.path.join(os.path.dirname(self.list_file), "exclude_list.txt")):
            with open(os.path.join(os.path.dirname(self.list_file), "exclude_list.txt"), 'r') as f:
                exclude_list = list(f)
                exclude_list = [t.strip() for t in exclude_list]

        video_list = []
        count = 0
        with open(self.list_file) as f:
            data_file = json.load(f)
            for name, info in data_file['database'].items():
                # if count > 1000:
                #     break
                # else:
                #     count =+ 1
                video_path = os.path.join(self.data_path, info["annotations"]['label'], name+cfg.DATALOADER.FILE_EXTENSION)
                # program will stop if there isn't an exclude list!
                if os.path.basename(video_path) in exclude_list:
                    continue
                if (self.is_train and info['subset'] == "training") or (not self.is_train and info['subset'] == "validation") :
                    inst = {
                        "video_path" : video_path,
                        "id": name
                    }
                    # if not self.test_mode:
                    label = info['annotations']['label']
                    inst["target_label"] = label
                    assert label in self.cls2idx
                    video_list.append(inst)

        if self.is_train and self.data_percentage < 1.0:
            video_dict = dict()
            for video in video_list:
                if video["target_label"] not in video_dict:
                    video_dict[video["target_label"]] = list()
                video_dict[video["target_label"]].append(video)
            new_list = list()
            for k, v in video_dict.items():
                new_list.extend(random.sample(v, k=int(len(v)*self.data_percentage)+1))
            video_list = new_list

        num = len(video_list)
        print("The number of videos is {}".format(num), flush=True)
        assert (num > 0)
        return video_list

    def _sample_indices(self, total_frames, fps):
        """
        Used for training.
        Args:
            - record (VideoRecord):
        Returns:
            list: frame index, index starts from 1.
        """
        if self.random_stride:
            interval = random.sample([8, 16, 32], k=1)[0]
        else:
            interval = self.interval
        frame_idx = np.asarray(random_clip(total_frames, interval * fps / self.target_fps , self.frames_per_clip))
        return frame_idx

    def _get_val_indices(self, total_frames, fps):
        max_frame_idx = total_frames - 1
        sample_pos = max(0, 1 + max_frame_idx - int(self.interval * fps / self.target_fps * self.frames_per_clip))
        start_list = np.linspace(0, sample_pos - 1, num=self.num_clips, dtype=int)
        frame_idx = []
        for start_idx in start_list.tolist():
            # ! changed by zhujinguo for torch.cat multi-views
            ids = [int(idx * self.interval * fps / self.target_fps + start_idx)%total_frames for idx in range(self.frames_per_clip)]
            ids = [x for x in ids if x < total_frames]
            frame_idx.append(ids)
        return frame_idx

    def __getitem__(self, index):
        for i_try in range(100):
            try:
                record = self.video_list[index].as_py()
                if self.use_ceph:
                    container = av.open(io.BytesIO(self.tcs_loader.client.get(record["video_path"])))
                else:
                    container = av.open(record["video_path"])
                # container.streams.video[0].thread_type = "AUTO"
                stream = container.streams.video[0]
                total_frames = stream.frames
                fps = float(container.streams.video[0].average_rate)

                if total_frames == 0:
                    # it returns 0 if not know, but that doesn't mean the video is null
                    for frame in container.decode(stream):
                        total_frames += 1
                    container.close()
                    container = av.open(record["video_path"])
                    stream = container.streams.video[0]
            except Exception as e:
                print(
                    "Failed to load video from {} with error {} ; trial {}".format(
                        record["video_path"], e, i_try
                    )
                )

                # let's try another one
                index = random.randint(0, len(self.video_list) - 1)
                continue


            if self.is_train:
                indices = [self._sample_indices(total_frames, fps)]
            else:
                indices = self._get_val_indices(total_frames, fps)

            all_index = set()
            for index in indices:
                all_index = all_index.union(set(index))

            start_index = min(all_index)
            num_frames = len(all_index)

            images = dict()

            fetched = 0

            for frame in container.decode(stream):
                if frame.index not in all_index or frame.index in images:
                    continue
                images[frame.index] = frame.to_rgb().to_image()
                last = frame.index
                fetched += 1
                if fetched == num_frames:
                    break

            container.close()

            video_data = list()
            for ind in indices:
                seq = list()
                for i in ind:
                    if i in images:
                        seq.append(images[i])
                    else:
                        seq.append(images[last])
                video_data.append(self.transform(seq))
            video_data = torch.cat(video_data, dim=0)
            # num_views, num_frames, 3, 224, 224
            if not self.is_train:
                if self.testing_multi_view == 'v1' and self.temporal_num_view > 1:
                    video_data = video_data.reshape(video_data.shape[0] * self.temporal_num_view, -1, *video_data.shape[-3:])
                    num_frames = num_frames // self.temporal_num_view
                elif self.testing_multi_view == 'v2' and self.temporal_num_view > 1:
                    video_data = video_data.reshape(video_data.shape[0], -1, self.temporal_num_view,
                                                    *video_data.shape[-3:]).transpose(1, 2).reshape(video_data.shape[0] * self.temporal_num_view, -1,
                                                                                                    *video_data.shape[-3:])
                    num_frames = num_frames // self.temporal_num_view


            ret = {
                'input_sample':[
                    {
                        'data': video_data, 'invalid_mask': None, 'modality': 'video', 'data_type': 'input',
                        'sample_info':{
                            'id': record['id'],
                            'path': record['video_path'],
                            'num_frames': num_frames,
                            'num_views': video_data.shape[0],
                            'cat_along_first_dim': True,
                            }
                    }
                ],
                'target_sample': [],
                'target_idx': [self.cls2idx[record['target_label']]],
                'target_set':  copy.deepcopy(self.target_set),
                'task_info':  copy.deepcopy(self.task_info)

            }

            # dict_as_tensor(ret)
            return ret

    def __len__(self):
        return len(self.video_list)
