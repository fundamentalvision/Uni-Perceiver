import os
import copy
import pickle
from PIL import Image
import torch
from torchvision import transforms
import random
from torchvision.transforms.transforms import ToTensor
from tqdm import tqdm
import numpy as np
from uniperceiver.config import configurable
from uniperceiver.functional import read_np, dict_as_tensor, boxes_to_locfeats
from ..build import DATASETS_REGISTRY
import glob
import json
from collections import defaultdict

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import pyarrow as pa
from uniperceiver.utils import comm

__all__ = ["ImageNetDataset", "ImageNet22KDataset"]


def load_pkl_file(filepath):
    return pickle.load(open(filepath, 'rb'), encoding='bytes') if len(filepath) > 0 else None

@DATASETS_REGISTRY.register()
class ImageNetDataset:
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: str,
        s3_path: str,
        feats_folder: str,
        class_names: list,
        use_ceph: bool,
        tcs_conf_path,
        data_percentage,
        task_info,
        target_set,
        cfg,
    ):
        self.stage = stage
        self.ann_file = anno_file
        self.feats_folder = feats_folder
        self.class_names = class_names if (class_names is not None) else None
        self.data_percentage = data_percentage

        self.initialized = False

        self.cfg = cfg

        self.task_info = task_info
        self.target_set = target_set
        # for index_maping
        self.idx2info = dict()

        self.use_ceph = use_ceph
        if self.use_ceph:
            self.feats_folder = s3_path
            print('debug info for imagenet{}  {}'.format(self.ann_file, self.feats_folder))
            from uniperceiver.datasets import TCSLoader
            self.tcs_loader = TCSLoader(tcs_conf_path)
           
        self.transform = build_transform(is_train=(self.stage == 'train'),
                                         input_size=cfg.MODEL.IMG_INPUT_SIZE)

        _temp_list =self.load_data(self.cfg)
        self.datalist = pa.array(_temp_list)
        if comm.is_main_process():
            import sys
            print("ImageNet1K Pretrain Dataset:")
            print('!!! length of _temp_list: ', len(_temp_list))
            print('!!! size of _temp_list: ', sys.getsizeof(_temp_list))
            print('!!! size of pa database: ', sys.getsizeof(self.datalist))
        del _temp_list

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        if 'SLURM_PROCID' in os.environ:
            tcs_conf_path = cfg.DATALOADER.get("TCS_CONF_PATH", "slurm_tools/petreloss.config")
        else:
            # dev machine
            tcs_conf_path = "slurm_tools/petreloss_local.config"
        ann_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "train.txt"),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "val.txt"),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "test.txt")
        }

        task_info = {
            'task_type'      : cfg.DATASETS.TASK_TYPE,
            'dataset_name'   : cfg.DATASETS.DATASET_NAME,
            'batch_size'     : cfg.DATALOADER.TRAIN_BATCH_SIZE if stage == 'train' else cfg.DATALOADER.TEST_BATCH_SIZE,
            'sampling_weight': cfg.DATALOADER.SAMPLING_WEIGHT
        }


        ret = {
            "cfg"            : cfg,
            "stage"          : stage,
            "anno_file"      : ann_files[stage],
            "feats_folder"   : cfg.DATALOADER.FEATS_FOLDER,
            's3_path'        : cfg.DATALOADER.S3_PATH,
            "class_names"    : load_pkl_file(cfg.DATALOADER.CLASS_NAME_FILE) if cfg.DATALOADER.CLASS_NAME_FILE else None,
            "use_ceph"       : getattr(cfg.DATALOADER, 'USE_CEPH', False),
            "tcs_conf_path"  : tcs_conf_path,
            "data_percentage": cfg.DATALOADER.DATA_PERCENTAGE,
            "task_info"      : task_info,
            "target_set"     : cfg.DATASETS.TARGET_SET
        }

        return ret

    def _preprocess_datalist(self, datalist):
        return datalist

    def load_data(self, cfg):
        datalist = []

        # local file reading
        with open(self.ann_file, 'r') as f:
            img_infos = f.readlines()

        if self.stage == "train" and self.data_percentage < 1.0:
            id2img = dict()
            for idx, l in enumerate(img_infos):
                name = int(l.replace('\n', '').split(' ')[1])
                if name not in id2img:
                    id2img[name] = list()
                id2img[name].append(idx)
                self.idx2info[idx] = l.replace('\n', '').split(' ')[0]

            datalist = list()
            for k, v in id2img.items():
                for idx in random.sample(v, k=int(len(v)*self.data_percentage)+1):
                    datalist.append({
                        'image_id': idx,
                        'class_id': k,
                        "file_path": self.idx2info[idx],
                    })
        else:
            datalist = [{
                'image_id': idx,
                'class_id': int(l.replace('\n', '').split(' ')[1]),
                "file_path": l.replace('\n', '').split(' ')[0],
            } for idx, l in enumerate(img_infos)]

        datalist = self._preprocess_datalist(datalist)
        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        for i_try in range(100):
            try:
                dataset_dict =self.datalist[index].as_py()
                image_id = dataset_dict['image_id']
                class_id = dataset_dict['class_id']
                image_name = dataset_dict['file_path']

                # load image
                image_path = os.path.join(self.feats_folder, self.stage, image_name)

                if self.use_ceph:
                    img = self.tcs_loader(image_path).convert('RGB')

                else:
                    img = Image.open(image_path).convert("RGB")


            except Exception as e:
                print(
                    "Failed to load image from {} with error {} ; trial {}".format(
                        image_path, e, i_try
                    )
                )

                # let's try another one
                index = random.randint(0, len(self.datalist) - 1)
                continue


            img = self.transform(img)


            ret = {
                'input_sample' : [{
                        'data'        : img, 
                        'invalid_mask': None,
                        'modality'    : 'image', 
                        'data_type': 'input',
                        'sample_info' : {
                            'id'  : image_id,
                            'path': image_path
                            }
                    }],
                'target_sample': [],
                'target_idx'   : [class_id],
                'target_set'   : copy.deepcopy(self.target_set),
                'task_info'    : copy.deepcopy(self.task_info)

            }
            return ret




@DATASETS_REGISTRY.register()
class ImageNet22KDataset:
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: str,
        s3_path: str, 
        feats_folder: str,
        use_ceph: bool,
        tcs_conf_path: str,
        cfg: str,
        task_info,
        target_set,
    ):
        self.cfg = cfg
        self.stage = stage
        self.ann_file = anno_file
        self.feats_folder = feats_folder
        self.task_info = task_info
        self.target_set = target_set
        self.initialized = False

        self.use_ceph = use_ceph
        if self.use_ceph:
            self.feats_folder = s3_path
            print('debug info for imagenet22k {}  {}'.format(self.ann_file, self.feats_folder))
            from uniperceiver.datasets import TCSLoader
            self.tcs_loader = TCSLoader(tcs_conf_path)


        self.transform = build_transform(is_train=(self.stage == 'train'),
                                         input_size=cfg.MODEL.IMG_INPUT_SIZE)

        _temp_list = self.load_data(self.cfg)
        self.datalist = pa.array(_temp_list)
        if comm.is_main_process():
            import sys
            print("ImageNet22K Pretrain Dataset:")
            print('!!! length of _temp_list: ', len(_temp_list))
            print('!!! size of _temp_list: ', sys.getsizeof(_temp_list))
            print('!!! size of pa database: ', sys.getsizeof(self.datalist))
        del _temp_list


    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
    
        ann_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "imagenet_22k_filelist_short.txt"),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "imagenet_22k_filelist_short.txt"),
        }


        if 'SLURM_PROCID' in os.environ:
            tcs_conf_path = cfg.DATALOADER.get("TCS_CONF_PATH", "slurm_tools/petreloss.config")
        else:
            # dev machine
            tcs_conf_path = "slurm_tools/petreloss_local.config"

        task_info = {
            'task_type'      : cfg.DATASETS.TASK_TYPE,
            'dataset_name'   : cfg.DATASETS.DATASET_NAME,
            'batch_size'     : cfg.DATALOADER.TRAIN_BATCH_SIZE if stage == 'train' else cfg.DATALOADER.TEST_BATCH_SIZE,
            'sampling_weight': cfg.DATALOADER.SAMPLING_WEIGHT
        }

        ret = {
            "cfg"          : cfg,
            "stage"        : stage,
            "anno_file"    : ann_files[stage],
            's3_path'      : cfg.DATALOADER.S3_PATH,
            "feats_folder" : cfg.DATALOADER.FEATS_FOLDER,
            "use_ceph"     : getattr(cfg.DATALOADER, 'USE_CEPH', False),
            "tcs_conf_path": tcs_conf_path,
            "task_info"    : task_info,
            "target_set"   : cfg.DATASETS.TARGET_SET
        }

        return ret

    def _preprocess_datalist(self, datalist):
        return datalist

    def load_data(self, cfg):
        datalist = []

        # local file reading
        with open(self.ann_file, 'r') as f:
            img_infos = f.readlines()

        datalist = []
        for idx, l in enumerate(img_infos):
            info_strip = l.replace('\n', '').split(' ')
            wn_id = info_strip[0]
            class_id = info_strip[2]
            file_path = wn_id + '/' + wn_id + '_' + info_strip[1] + '.JPEG'  # n01440764/n01440764_10074.JPEG

            datalist.append(
                {
                    'image_id': idx,
                    'file_path': file_path,
                    'class_id': int(class_id)
                }
            )

        datalist = self._preprocess_datalist(datalist)
        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        for i_try in range(100):
            try:
                dataset_dict =self.datalist[index].as_py()
                image_id = dataset_dict['image_id']
                class_id = dataset_dict['class_id']
                image_name = dataset_dict['file_path']

                # load image
                image_path = os.path.join(self.feats_folder, image_name)

                if self.use_ceph:
                    img = self.tcs_loader(image_path).convert('RGB')

                else:
                    img = Image.open(image_path).convert("RGB")
               

            except Exception as e:
                print(
                    "Failed to load image from {} with error {} ; trial {}".format(
                        image_path, e, i_try
                    )
                )

                # let's try another one
                index = random.randint(0, len(self.datalist) - 1)
                continue

            img = self.transform(img)

            ret = {
                'input_sample': [{
                        'data'        : img, 
                        'invalid_mask': None, 
                        'modality'    : 'image', 
                        'data_type': 'input',
                        'sample_info' : {
                            'id'  : image_id, 
                            'path': image_path
                            }
                    }],
                'target_sample': [],
                'target_idx'   : [class_id],
                'target_set'   : copy.deepcopy(self.target_set),
                'task_info'    : copy.deepcopy(self.task_info)
            }

            return ret



def build_transform(is_train,
                    input_size=224,
                    color_jitter=0.4,
                    auto_augment='rand-m9-mstd0.5-inc1',
                    train_interpolation='bicubic',
                    re_prob=0.25,
                    re_mode='pixel',
                    re_count=1
                   ):
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            interpolation=train_interpolation,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count
        )

        return transform

    t = []
    size = int((256 / 224) * input_size)
    t.append(
        transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
