import random
import os
import time
import json
from tqdm import trange
# import jsonlines
from PIL import Image, ImageFile
import copy

# ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import base64
import numpy as np
import pyarrow as pa
import logging
# import spacy
import glob
from io import BytesIO
import jsonlines

import torch
from torch.utils.data import Dataset
from uniperceiver.functional import read_np, dict_as_tensor, boxes_to_locfeats
from collections import defaultdict

from uniperceiver.datasets.zipreader import ZipReader
import errno
from uniperceiver.datasets.circular_cached_loader import CircularCachedInputIterator

from uniperceiver.tokenization import ClipTokenizer

from ..build import DATASETS_REGISTRY
# from uniperceiver.config import kfg
from uniperceiver.config import configurable
import pickle
from uniperceiver.utils import comm

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torchvision import transforms
from uniperceiver.datasets.custom_transforms import clip_transforms

__all__ = ["ImageTextPairDataset"]

memorycache = False



def makedirsExist(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory not created.')
        else:
            raise

def _smart_join(str_or_list, delim):
    if isinstance(str_or_list, str):
        return str_or_list
    else:
        return delim.join(str_or_list)

@DATASETS_REGISTRY.register()
class ImageTextPairDataset(Dataset):

    @configurable
    def __init__(self, cfg, stage, ann_file, image_set, root_path, data_path, s3_path,
                 feats_folder,
                 dataset_name,
                 data_percentage,
                 seq_per_img,
                 tokenizer, tokenizer_name,
                 seq_len=64,
                 mask_prob=(0.15, 0.8), repl_prob=0.1,
                 task_type=True,
                 transform=None, test_mode=False,
                 zip_mode=False,
                 cache_mode=False,
                 cache_origin_image=False,
                 cache_local_rank=0, cache_local_size=1,
                 circular_cache_mode=False,
                 ignore_db_cache=True,
                 aspect_grouping=False,
                 use_ceph=False,
                 tcs_conf_path='',
                 random_caption=False,
                 max_length=-1,
                 as_numpy_as_possible=False,
                 use_node_distirbuted_sampler=False,
                 **kwargs):
        """
        Conceptual Captions Dataset

        :param ann_file: annotation jsonl file
        :param image_set: image folder name, e.g., 'vcr1images'
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to vcr dataset
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param aspect_grouping: whether to group images via their aspect
        :param kwargs:
        """
        super(ImageTextPairDataset, self).__init__()

        # assert not cache_mode, 'currently not support cache mode!'
        assert not test_mode
        assert not (cache_mode and circular_cache_mode)

        self.mask_prob = mask_prob
        self.repl_prob = repl_prob
        self.seq_len = seq_len
        self.task_type = task_type
        self.cfg = cfg
        self.stage = stage
        self.dataset_name = dataset_name
        self.feats_folder = feats_folder
        self.seq_per_img = seq_per_img
        assert self.seq_per_img == 1
        self.data_percentage = data_percentage


        self.data_path = data_path
        self.root_path = root_path
        self.ann_file = ann_file
        self.image_set = image_set
        self.transform = transform
        self.test_mode = test_mode
        self.zip_mode = zip_mode
        self.cache_mode = cache_mode
        self.cache_origin_image = cache_origin_image
        self.cache_local_rank = cache_local_rank
        self.cache_local_size = cache_local_size
        self.circular_cache_mode = circular_cache_mode
        self.ignore_db_cache = ignore_db_cache
        self.aspect_grouping = aspect_grouping
        self.cache_dir = os.path.join(self.data_path, 'cache')
        self.use_node_distirbuted_sampler = (use_node_distirbuted_sampler or cache_mode)
        if not os.path.exists(self.cache_dir):
            makedirsExist(self.cache_dir)

        self.initialized = False

        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.use_clip_tokenizer = tokenizer_name == 'clip'

        self.zipreader = ZipReader()

        self.use_ceph = use_ceph
        self.tcs_conf_path = tcs_conf_path
        if use_ceph:
            self.data_path = s3_path
            from uniperceiver.datasets.tcsreader import TCSLoader
            self.tcs_loader = TCSLoader(tcs_conf_path)
        else:
            self.data_path = feats_folder
            
        if comm.is_main_process():
            print(f"data_path for Dataset {self.dataset_name} with task {self.task_type}: {self.data_path}")
            
        self.random_caption = random_caption


        if self.dataset_name == 'VG':
            self.load_VG(self.cfg)
        elif self.dataset_name in ['MSCOCO', 'FLICKR']:
            self.load_COCO_flickr(self.cfg)
        else:
            self.load_database()

        if self.circular_cache_mode:
            chunk_dir = os.path.join(self.data_path, '{}_chunks'.format(image_set))
            self.chunk_path_list = glob.glob(os.path.join(chunk_dir, '*.pa'))

        if self.aspect_grouping:
            assert False, "not support aspect grouping currently!"
            self.group_ids = self.group_aspect(self.database)

        self.as_numpy_as_possible = as_numpy_as_possible
        self.max_length = max_length

        self.task_info = {
                'task_type'      : self.cfg.DATASETS.TASK_TYPE,
                'dataset_name'   : self.cfg.DATASETS.DATASET_NAME,
                'batch_size'     : self.cfg.DATALOADER.TRAIN_BATCH_SIZE if self.stage == 'train' else self.cfg.DATALOADER.TEST_BATCH_SIZE,
                'sampling_weight': self.cfg.DATALOADER.SAMPLING_WEIGHT
            }

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):

        if 'SLURM_PROCID' in os.environ:
            tcs_conf_path = cfg.DATALOADER.get("TCS_CONF_PATH", "petreloss.config")
        else:
            # dev machine
            tcs_conf_path = "slurm_tools/petreloss_local.config"
        anno_filename = cfg.DATALOADER.get("ANNO_FILENAME", "train_spacy.json")
        if cfg.DATALOADER.USE_CEPH and cfg.DATALOADER.S3_ANNO_FOLDER is not None:
            anno_folder = cfg.DATALOADER.S3_ANNO_FOLDER
        else:
            anno_folder = cfg.DATALOADER.ANNO_FOLDER
        if cfg.DATASETS.DATASET_NAME == 'MSCOCO':
            anno_files = {
                "train": [os.path.join(anno_folder, "captions_train113k.json"), os.path.join(anno_folder, "captions_val5k.json")],
                # no validation
                "test": os.path.join(anno_folder, "captions_test5k.json")
            }
        elif cfg.DATASETS.DATASET_NAME == 'FLICKR':
            anno_files = {
                "train": [os.path.join(anno_folder, "all_data_final_train_2014.jsonline"), os.path.join(anno_folder, "all_data_final_val_set0_2014.jsonline")],
                # no val
                # "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "all_data_final_val_set0_2014.jsonline"),
                "test": os.path.join(anno_folder, "all_data_final_test_set0_2014.jsonline")
            }
        else:
            anno_files = {
                "train":  os.path.join(anno_folder, anno_filename),
                "val": os.path.join(anno_folder, anno_filename),
                "test": os.path.join(anno_folder, anno_filename),
            }
        if getattr(cfg.DATALOADER, 'TRANSFORM', None) == 'clip_transforms':
            transform = clip_transforms(stage, img_size=cfg.MODEL.IMG_INPUT_SIZE)
        else:
            # same as imagenet
            transform = build_transform(is_train=(stage=='train'))

        ret = {
            'cfg': cfg,
            'stage': stage,
            'ann_file' : anno_files[stage],
            "seq_per_img": 1,
            'image_set' : stage,
            'root_path' : cfg.DATALOADER.ANNO_FOLDER,
            'data_path' : cfg.DATALOADER.FEATS_FOLDER,
            's3_path': cfg.DATALOADER.S3_PATH,
            'feats_folder': cfg.DATALOADER.FEATS_FOLDER,
            'dataset_name': cfg.DATASETS.DATASET_NAME,
            "data_percentage": cfg.DATALOADER.DATA_PERCENTAGE,
            'seq_len':  cfg.MODEL.MAX_SEQ_LEN,
            'task_type': cfg.DATASETS.TASK_TYPE,
            'transform': transform,
            'zip_mode': cfg.DATALOADER.ZIP_MODE,
            "cache_mode": cfg.DATALOADER.CACHE_MODE,
            'cache_origin_image': cfg.DATALOADER.CACHE_ORIGIN_IMAGE,
            "cache_local_rank": comm.get_local_rank(),
            "cache_local_size": comm.get_local_size(),
            "circular_cache_mode": cfg.DATALOADER.CIRCULAR_CACHE_MODE,
            "use_ceph": getattr(cfg.DATALOADER, 'USE_CEPH', False),
            "tcs_conf_path": tcs_conf_path,
            "random_caption": cfg.DATALOADER.RANDOM_CAPTION,
            "as_numpy_as_possible": cfg.DATALOADER.AS_NUMPY_AS_POSSIBLE,
            "use_node_distirbuted_sampler": cfg.DATALOADER.SAMPLER == 'NodeDistributed',
            'tokenizer': ClipTokenizer(),
            'tokenizer_name': "clip",

        }

     
            
        return ret

    def _init_memcached(self):
        pass

    def load_img_info(self, anno_file):
        id2path = {}
        with jsonlines.open(anno_file) as reader:
            for annotation in reader:
                image_id =  annotation["id"]
                id2path[image_id] = annotation["img_path"]

        return id2path
        
    def load_COCO_flickr(self, cfg):
        # for index_mapping
        self.idx2name = dict()
        self.name2idx = dict()
        if isinstance(self.ann_file, list):
            imageinfo = list()
            self.id2path = dict()
            for anno_file in self.ann_file:
                if self.dataset_name == 'MSCOCO':
                    imageinfo.extend(json.load(open(anno_file))["images"])
                else:
                    id2path = self.load_img_info(anno_file)
                    self.id2path.update(id2path)
        else:
            if self.dataset_name == 'MSCOCO':
                imageinfo = json.load(open(self.ann_file))["images"]
            else:
                self.id2path = self.load_img_info(self.ann_file)

        if self.dataset_name == 'MSCOCO':
            for info in imageinfo:
                self.idx2name[info['id']] = {
                    "split": info['file_path'],
                    "name": info['file_name']}
                self.name2idx[info['file_name']] = info['id']

        if self.stage == "test":
            if self.dataset_name == 'MSCOCO':
                cache_path = os.path.join(
                    os.path.dirname(self.ann_file), "cache",
                    "mscoco_caption_w_testcap_%s.pkl" % ( self.stage)
                )
            else:
                cache_path = os.path.join(
                    self.root_path, "cache",
                    "RetrievalFlickr30k_raw_%s_%s_%d.pkl" % (self.tokenizer_name, self.stage, self.seq_len)
                )

            if not os.path.exists(os.path.dirname(cache_path)):
                os.makedirs(os.path.dirname(cache_path))
            if not os.path.exists(cache_path):
                datalist = self.load_raw_data(cfg, self.ann_file)
                pickle.dump(datalist, open(cache_path, "wb"))
            datalist = pickle.load(open(cache_path, "rb"))
        else:
            datalist = list()
            assert self.stage == "train", "no validation now"
            for i, stage in enumerate(["train", "val"]):
                if self.dataset_name == 'MSCOCO':
                    cache_path = os.path.join(
                        os.path.dirname(self.ann_file[i]), "cache",
                        "mscoco_caption_w_testcap_%s.pkl" % ( stage)
                    )
                else:
                    cache_path = os.path.join(
                        self.root_path, "cache",
                        "RetrievalFlickr30k_raw_%s_%s_%d.pkl" % (self.tokenizer_name, stage, self.seq_len)
                    )
                if not os.path.exists(os.path.dirname(cache_path)):
                    os.makedirs(os.path.dirname(cache_path))
                if not os.path.exists(cache_path):
                    datalist_part = self.load_raw_data(cfg, self.ann_file[i])
                    pickle.dump(datalist_part, open(cache_path, "wb"))
                datalist_part = pickle.load(open(cache_path, "rb"))
                datalist.extend(datalist_part)

        if self.data_percentage < 1.0 and self.stage == 'train':
            datalist = random.sample(datalist, k = int(self.data_percentage* len(datalist) )  )

        self.database = pa.array(datalist)
        if comm.is_main_process():
            import sys
            print(f"!!! Dataset {self.dataset_name} with task {self.task_type}:")
            print('!!! length of _temp_list: ', len(datalist))
            print('!!! size of _temp_list: ', sys.getsizeof(datalist))
            print('!!! size of pa database: ', sys.getsizeof(self.database))
        del datalist

    def load_raw_data(self, cfg, anno_file):
        datalist = []
        if self.dataset_name == 'MSCOCO':
            annoinfo = json.load(open(anno_file))
            captions_train = sorted( annoinfo['annotations'], key=lambda x: x['id'])
            image_caption_info = defaultdict(list)
            for cap_info in captions_train:
                image_caption_info[cap_info['image_id']].append(cap_info['caption'])

            for im_id, caps in image_caption_info.items():
                datalist.append(
                    {
                        "image_id": im_id,
                        "captions": caps,
                    }
                )
        else:
            with jsonlines.open(anno_file) as reader:
                for annotation in reader:
                    sentences = annotation["sentences"]
                    image_id = annotation["id"]
                    datalist.append({ "image_id": image_id, "imagename": annotation["img_path"], "captions": sentences })


        return datalist

    def load_VG(self, cfg):
        cache_path = os.path.join(
            os.path.dirname(self.ann_file), "cache",
            "vg_caption_spe_raw_%s.pkl" % (self.stage)
        )
        if not os.path.exists(os.path.dirname(cache_path)):
            os.makedirs(os.path.dirname(cache_path))
        if not os.path.exists(cache_path):
            _temp_list = []
            if self.use_ceph:
                anno_file = os.path.join('s3://visual_genome/annotations', os.path.basename(self.ann_file))
                annotations = json.load(BytesIO(self.tcs_loader.client.get(anno_file)))  
            else:
                annotations = json.load(open(self.ann_file))

            for im_id, annoinfo in annotations['phrase'].items():
                _temp_list.append(
                    {
                        "image_id": im_id,
                        "captions": annoinfo,
                        'path': annotations['subset'][im_id],
                    }
                )
            pickle.dump(_temp_list, open(cache_path, "wb"))
        else:
            _temp_list = pickle.load(open(cache_path, "rb"))
        self.database = pa.array(_temp_list)

        if comm.is_main_process():
            import sys
            print(f"!!! Dataset {self.dataset_name} with task {self.task_type}:")
            print('!!! length of _temp_list: ', len(_temp_list))
            print('!!! size of _temp_list: ', sys.getsizeof(_temp_list))
            print('!!! size of pa database: ', sys.getsizeof(self.database))
        del _temp_list

    def load_database(self):

        if self.random_caption:
            cache_filename  =  'spe_cache_random_caption_' + os.path.basename(self.ann_file).replace('.', "_") + "_" + str(self.cache_local_rank) + "_" + str(self.cache_local_size) + '.pkl'
        else:
            cache_filename  =  'spe_cache_' + os.path.basename(self.ann_file).replace('.', "_") + "_" + str(self.cache_local_rank) + "_" + str(self.cache_local_size) + '.pkl'


        cache_file = os.path.join(self.cache_dir, cache_filename)

        if not os.path.exists((cache_file)):
            _temp_list = []
            self.img_path_to_index = {}
            if self.use_ceph:
                f =  BytesIO(self.tcs_loader.client.get(self.ann_file))
            else:
                f = open(self.ann_file, 'r')
            if self.dataset_name == 'SBU':
                annofile = json.load(f)
            else:
                annofile = f
            for i, l in enumerate(annofile):
                if self.use_node_distirbuted_sampler and ((i % self.cache_local_size) != self.cache_local_rank):
                    _temp_list.append(None)
                    continue
                l = l.strip()
                if (l == ''):
                    continue
                if self.dataset_name == 'SBU':
                    self.img_path_to_index[l] = i
                    _temp_list.append([l, annofile[l]])
                else:
                    _data = json.loads(l)
                    if not self.zip_mode:
                        _data['image'] = _data['image'].replace('.zip@', '')
                    self.img_path_to_index[_data['image']] = i
                    if self.random_caption:
                        _temp_list.append([_data['image'], _smart_join(_data['caption'], '\t'), _data['title'], _data['description']])
                    else:
                        _temp_list.append([_data['image'], _smart_join(_data['caption'], '\t')])

            f.close()


            pickle.dump({
                "path_to_indext": self.img_path_to_index,
                "temp_list": _temp_list,
                }, open(cache_file, "wb"), protocol=4)
        else:
            cachedata = pickle.load(open(cache_file, "rb"))
            self.img_path_to_index, _temp_list = cachedata['path_to_indext'], cachedata['temp_list']

        self.database = pa.array(_temp_list)
        if comm.is_main_process():
            import sys
            print(f"!!! Dataset {self.dataset_name} with task {self.task_type}:")
            print('!!! length of _temp_list: ', len(_temp_list))
            print('!!! size of _temp_list: ', sys.getsizeof(_temp_list))
            print('!!! size of pa database: ', sys.getsizeof(self.database))
        del _temp_list

    @property
    def data_names(self):
        return ['image', 'im_info', 'text', 'mlm_labels']

    def __getitem__(self, index):
        for i_try in range(100):
            try:
                image_path = None
                image_id = None
                idb = None
                if self.dataset_name in ['VG', 'MSCOCO', 'FLICKR']:
                    self.dataset_dict = self.database[index].as_py()
                    image_id = self.dataset_dict['image_id']
                    if self.dataset_name == 'VG':
                        imagepath = self.dataset_dict['path']
                        image_path = os.path.join(self.data_path, imagepath)
                    elif self.dataset_name == 'FLICKR':
                        image_path = os.path.join(self.data_path, self.id2path[image_id])
                    else:
                        image_split = self.idx2name[int(image_id)]['split']
                        image_name = self.idx2name[int(image_id)]['name']
                        image_path = os.path.join(self.data_path, image_split, image_name)
                else:
                    _idb = self.database[index]
                    idb = {'image': str(_idb[0]).strip('./'), 'caption': str(_idb[1]).split('\t')}
                    if self.random_caption:
                        idb['title'] = [_idb[2].as_py()]
                        idb['description'] = [_idb[3].as_py()]
                return self._data_transform(idb, index=index, as_numpy_as_possible=self.as_numpy_as_possible, image_path=image_path, image_id=image_id)
            except Exception as e:
                print(
                    "Failed to load image from idb {} with error {} ; trial {};".format(
                        self.database[index], e, i_try
                    )
                )
                index = (index + 1)%len(self.database)
                while (self.database[index].as_py() is None):
                    index = (index + 1)%len(self.database)
                continue

    def _data_transform(self, idb, index=None, as_numpy_as_possible=False, fail_image_fill=(0.0, 0.0, 0.0), image_path=None, image_id=None):

        if self.dataset_name in ['VG', 'MSCOCO', 'FLICKR']:
            image = self._load_image(image_path)
        else:
            if index is None:
                index = self.img_path_to_index[idb['image']]
            # image data

            image = self.get_image(idb, index=index)
            if isinstance(image, Image.Image):
                w0, h0 = image.size
            elif isinstance(image, np.ndarray):
                h0, w0, c_ = image.shape
                assert c_ == 3
            else:
                raise NotImplementedError

        if self.transform is not None:
            image = self.transform(image)

        if image_id is not None:
            img_sample_info = {
                'id': image_id,
                'path': image_path
            }
        else:
            img_sample_info = {
                'id': index
            }
        ret = {
                'input_sample': [{
                    'data'        : image,
                    'invalid_mask': None,
                    'modality'    : 'image',
                    'data_type'   : 'input',
                    'sample_info' : copy.deepcopy(img_sample_info)
                }]
        }

        self.target_set = self.cfg.DATASETS.TARGET_SET

        mlm_labels = None
        u_mask_type = None
        if self.task_type == 'image_caption' and self.stage != 'train':
            ret.update({
                'target_set': copy.deepcopy(self.target_set),
                'target_sample': [],
                'target_idx': [],
                'task_info': copy.deepcopy(self.task_info)
            })
            dict_as_tensor(ret)
            return ret

        if self.task_type =='image_retrieval' and self.stage != 'train':
            captions = [caption +  " <|endoftext|>" for caption in  self.dataset_dict['captions']]
            caption_tokens_raw = [ self.tokenizer.encode(caption) for caption in captions]
            if self.dataset_name in ['MSCOCO', 'FLICKR']:
                caption_tokens = [ caption_token[:(self.seq_len - 1)] + [caption_token[-1]]
                                if len(caption_token) > self.seq_len else caption_token
                                for caption_token in caption_tokens_raw ]
            return self.package_item(ret, caption_tokens, mlm_labels, u_mask_type)

        # Task #1: Masked Language Modeling
        if self.random_caption:
            if len(idb['title']) == 0:
                caption = idb['description']
                if len(self.tokenizer.encode(' '.join(caption))) == 0:
                    caption = ['image']
            else:
                if random.random() < 0.5:
                    caption = idb['title']
                    if len(self.tokenizer.encode(' '.join(caption))) == 0:
                        caption = idb['description']
                        if len(self.tokenizer.encode(' '.join(caption))) == 0:
                            caption = ['image']
                else:
                    caption = idb['description']
                    if len(self.tokenizer.encode(' '.join(caption))) == 0:
                        caption = idb['title']
                        if len(self.tokenizer.encode(' '.join(caption))) == 0:
                            caption = ['image']
        else:
            if self.dataset_name == 'VG':
                caption = random.sample(self.dataset_dict['captions'], self.seq_per_img)[0]
                while len(caption) < 1:
                    caption = random.sample(self.dataset_dict['captions'], self.seq_per_img)[0]
                if caption and caption.lower()[-1] in "qwertyuiopasdfghjklzxcvbnm1234567890":
                    caption = caption + "."
            elif self.dataset_name in ['MSCOCO', 'FLICKR']:
                caption = random.sample(self.dataset_dict['captions'], self.seq_per_img)[0]
            else:
                caption = idb['caption']
                if caption and caption[-1] and caption[-1].lower()[-1] in "1234567890qwertyuiopasdfghjklzxcvbnm":
                    caption.append(".")

                # <PERSON> in CC12m
                # print('Before:', caption)
                for i_, tok in enumerate(caption):
                    if '<PERSON>' in tok:
                        tok = tok.replace('<PERSON>', 'person')
                        caption[i_] = tok

        if self.task_type == 'mlm':
            u_mask_type = 1
        elif self.task_type == 'image_caption':
            u_mask_type = 0 # causal mask 
        
        if self.dataset_name in ['VG', 'MSCOCO', 'FLICKR']:
            caption = caption + " <|endoftext|>"
        else:
            caption = caption + ["<|endoftext|>"]

        if self.task_type=='mlm':
            if self.dataset_name in ['VG', 'MSCOCO', 'FLICKR']:
                caption_tokens = self.tokenizer.basic_tokenize(caption)
            else:
                if self.use_clip_tokenizer:
                    caption_tokens = self.tokenizer.basic_tokenize(' '.join(caption))
                else:
                    caption_tokens = self.tokenizer.basic_tokenizer.tokenize(' '.join(caption))
            caption_tokens, mlm_labels = self.random_word_wwm(caption_tokens)
        elif self.task_type == 'image_caption':
            if self.dataset_name in ['VG', 'MSCOCO', 'FLICKR']:
                caption_tokens = self.tokenizer.encode(caption)
                mlm_labels = self.tokenizer.encode("<|spe|>")*len(caption_tokens)
            else:
                # caption
                caption_tokens = self.tokenizer.encode(' '.join(caption))
                mlm_labels = self.tokenizer.encode("<|spe|>")*len(caption_tokens)
        else:
            if self.dataset_name in ['VG', 'MSCOCO', 'FLICKR']:
                caption_tokens = self.tokenizer.encode(caption)
            else:
                caption_tokens = self.tokenizer.encode(' '.join(caption))
                mlm_labels = [-1] * len(caption_tokens)

        text =  caption_tokens

        # truncate seq to max len
        if len(text) > self.seq_len:
            # mlm task
            text_len_keep = self.seq_len
            text = text[:(text_len_keep - 1)] + [text[-1]]
            if self.task_type=='image_caption' or self.task_type=='mlm':
                mlm_labels = mlm_labels[:(text_len_keep - 1)] + [mlm_labels[-1]]


        if as_numpy_as_possible:
            text = np.array(text)
            mlm_labels = np.array(mlm_labels)

        return self.package_item(ret, text, mlm_labels, u_mask_type)


        # return image, im_info, text, mlm_labels

    def package_item(self, ret, text, mlm_labels, u_mask_type):


        if self.task_type == 'image_retrieval':
            if self.stage == 'train':
                ret.update({
                    'target_sample': [{
                        'data'        : [np.array(text, dtype=np.int64)],
                        'modality'    : 'text',
                        'data_type'  : 'target',
                        'invalid_mask':  None,
                        'sample_info' : {}
                    }],
                    'target_idx'   : [],
                    'target_set'   : [],
                    'task_info'    : copy.deepcopy(self.task_info)
                })
            else:
                image_id = ret['input_sample'][0]['sample_info']['id']
                ret['input_sample'][0]['sample_info']['id']  = (image_id, [image_id] * len(text))
                ret.update({
                    'target_sample': [{
                        'data': [np.array(single_text, dtype=np.int64) for single_text in text],
                        'modality': 'text',
                        'invalid_mask': None,
                        'data_type': 'target',
                        'sample_info': {
                            'sample_alone': True,
                        }
                    }],
                    'target_idx': [],
                    'target_set': [],
                    'task_info':
                    copy.deepcopy(self.task_info)
                })

        elif self.task_type == 'mlm':

            raise NotImplementedError('no needed for masked language modeling when given image now.')

        elif self.task_type == 'image_caption':
            source = np.array(text, dtype=np.int64)
            source2 = np.array(mlm_labels, dtype=np.int64)
            ret['input_sample'].append({
                'data': [source, source2],
                'invalid_mask': None,
                'modality': 'text',
                'data_type': 'input',
                'sample_info': {
                    'text_spe_cat': True,
                }
            })
            ret.update({
                'target_sample': [],
                'target_idx'   : [np.array(text, dtype=np.int64)],
                'target_set'   : copy.deepcopy(self.target_set),
                'task_info'    : copy.deepcopy(self.task_info)
            })
        else:
            raise NotImplementedError

        dict_as_tensor(ret)

        return ret

    def random_word_wwm(self, tokens):
        output_tokens = []
        output_label = []

        for i, token in enumerate(tokens):
            if self.use_clip_tokenizer:
                sub_tokens = self.tokenizer.encode_basic_tokenized_token(token)
            else:
                sub_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(token)
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    for sub_token in sub_tokens:
                        if self.use_clip_tokenizer:
                            output_tokens.append(self.tokenizer.encoder["<|spe|>"])
                        else:
                            output_tokens.append("[MASK]")
                # 10% randomly change token to random token
                elif prob < 0.9:
                    for sub_token in sub_tokens:
                        if self.use_clip_tokenizer:
                            output_tokens.append(random.choice(list(range(len(self.tokenizer.encoder)))))
                        else:
                            output_tokens.append(random.choice(list(self.tokenizer.vocab.keys())))
                # -> rest 10% randomly keep current token
                else:
                    for sub_token in sub_tokens:
                        output_tokens.append(sub_token)

                # append current token to output (we will predict these later)
                for sub_token in sub_tokens:
                    if self.use_clip_tokenizer:
                        output_label.append(sub_token)
                    else:
                        try:
                            output_label.append(self.tokenizer.vocab[sub_token])
                        except KeyError:
                            # For unknown words (should not occur with BPE vocab)
                            output_label.append(self.tokenizer.vocab["[UNK]"])
                            logging.warning("Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(sub_token))
            else:
                for sub_token in sub_tokens:
                    # no masking token (will be ignored by loss function later)
                    output_tokens.append(sub_token)
                    output_label.append(-1)

        return output_tokens, output_label

    def cache_images(self, resize_to=(224, 224)):
        assert not self.zip_mode
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        barray = bytearray()
        cursor = []
        c_ = 0
        for i in trange(len(self.database)):
            if i % self.cache_local_size != self.cache_local_rank:
                cursor.append(c_)
                continue
            idb = self.database[i]
            if self.cache_origin_image:
                try:
                    with open(os.path.join(self.data_path, idb['image']), 'rb') as f:
                        im = f.read()
                except:
                    print("Failed to cache image {}, cache zero byte!".format(idb['image']))
                    im = bytes()
            else:
                im = cv2.imread(os.path.join(self.data_path, idb['image']), cv2.IMREAD_COLOR)
                if im is None:
                    print("Failed to cache image {}, cache zero image!".format(idb['image']))
                    w, h = resize_to
                    im = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    im = cv2.resize(im, resize_to)
                _, im = cv2.imencode('.jpg', im, encode_param)
                im = im.tobytes()
            barray += im
            cursor.append(c_)
            c_ += len(im)
        cursor.append(c_)

        return barray, cursor

    def get_image(self, idb, index=None):
        if index is None:
            index = self.img_path_to_index[idb['image']]
        if self.circular_cache_mode:
            im = idb['image_augmented']
        else:
            im = self._load_image(os.path.join(self.data_path, idb['image']))
        return im

    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())

    @staticmethod
    def group_aspect(database):
        print('grouping aspect...')
        t = time.time()

        # get shape of all images
        widths = torch.as_tensor([idb['width'] for idb in database])
        heights = torch.as_tensor([idb['height'] for idb in database])

        # group
        group_ids = torch.zeros(len(database))
        horz = widths >= heights
        vert = 1 - horz
        group_ids[horz] = 0
        group_ids[vert] = 1

        print('Done (t={:.2f}s)'.format(time.time() - t))

        return group_ids

    def __len__(self):
        length = len(self.database)
        if self.max_length > 0:
            length = min(self.max_length, length)
        return length
        # return 10000000

    def _load_image(self, path):
        if '.zip@' in path:
            return self.zipreader.imread(path).convert('RGB')
        else:
            if self.use_ceph:
                # print('USE TCS!!!!!')
                return self.tcs_loader(path).convert('RGB')
            elif not memorycache:
                with open(path, 'rb') as f:
                    return Image.open(f).convert('RGB')
            else:
                # memcached
                raise NotImplementedError

    def _load_json(self, path):
        if '.zip@' in path:
            f = self.zipreader.read(path)
            return json.loads(f.decode())
        else:
            with open(path, 'r') as f:
                return json.load(f)


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
