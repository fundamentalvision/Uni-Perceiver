import os
import copy
import pickle
from PIL import Image
from torchvision import transforms
import random
from torchvision.transforms.transforms import ToTensor
from tqdm import tqdm
import numpy as np
from uniperceiver.config import configurable
from uniperceiver.functional import read_np, dict_as_tensor, boxes_to_locfeats
from ..build import DATASETS_REGISTRY
import glob
from uniperceiver.tokenization import ClipTokenizer
import json
from collections import defaultdict
from uniperceiver.datasets.custom_transforms import clip_transforms
import pyarrow as pa
from uniperceiver.utils import comm

__all__ = ["ImageTextPairDataset"]

memorycache = False

@DATASETS_REGISTRY.register()
class ImageTextPairDataset:
    @configurable
    def __init__(
        self,
        cfg: str,
        stage: str,
        anno_file: str,
        seq_per_img: int,
        max_seq_len: int,
        feats_folder: str,
        relation_file: str,
        gv_feat_file: str,
        attribute_file: str,
        transform,
        tokenizer,
        data_percentage,
        tokenizer_name,
        use_ceph: bool,
        tcs_conf_path,
        task_type,
        preload_feats = None,
        random_mask=False,
        text_type_id=0,
    ):
        assert len(task_type)>0
        self.cfg = cfg
        self.stage = stage
        self.anno_file = anno_file
        self.seq_per_img = seq_per_img
        assert self.seq_per_img == 1
        self.use_ceph = use_ceph
        self.task_type = task_type
        if self.use_ceph:
            self.feats_folder = 's3://coco'
            print('debug info for coco pretrain: {} '.format(self.feats_folder))
            from uniperceiver.datasets import TCSLoader
            if 'SLURM_PROCID' in os.environ:
                self.tcs_loader = TCSLoader(tcs_conf_path)
            else:
                self.tcs_loader = TCSLoader('petreloss_local.config')
        else:
            # local image folder
            self.feats_folder = feats_folder
        self.max_seq_len = max_seq_len
        self.relation_file = relation_file
        self.gv_feat_file = gv_feat_file
        self.attribute_file = attribute_file

        self.data_percentage = data_percentage
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.use_clip_tokenizer = tokenizer_name == 'clip'

        self.initialized = False
        self.transform = transform

        self.loaded_feats = None
        if preload_feats:
            self.loaded_feats = self.pre_load_feats(preload_feats)

        # for index_maping
        self.idx2name = dict()
        self.name2idx = dict()
        # please
        if isinstance(self.anno_file, list):
            imageinfo = list()
            for anno_file in self.anno_file:
                imageinfo.extend(json.load(open(anno_file))["images"])
        else:
            imageinfo = json.load(open(self.anno_file))["images"]
        for info in imageinfo:
            self.idx2name[info['id']] = {
                "split": info['file_path'],
                "name": info['file_name']}
            self.name2idx[info['file_name']] = info['id']
        self.random_mask = random_mask

        self.text_type_id = text_type_id

        self.task_info = {
                'task_type'      : self.cfg.DATASETS.TASK_TYPE,
                'dataset_name'   : self.cfg.DATASETS.DATASET_NAME,
                'batch_size'     : self.cfg.DATALOADER.TRAIN_BATCH_SIZE
                                  if self.stage == 'train' else self.cfg.DATALOADER.TEST_BATCH_SIZE,
                'sampling_weight': self.cfg.DATALOADER.SAMPLING_WEIGHT
            }

        _temp_list =self.load_data(self.cfg)
        self.database = pa.array(_temp_list)
        if comm.is_main_process():
            import sys
            print("MSCOCO Pretrain Dataset:")
            print('!!! length of _temp_list: ', len(_temp_list))
            print('!!! size of _temp_list: ', sys.getsizeof(_temp_list))
            print('!!! size of pa database: ', sys.getsizeof(self.database))
        del _temp_list


    def pre_load_feats(self, preload_feat_folder):
        loaded_feats = {}
        file_list = glob.glob(os.path.join(preload_feat_folder, '*.pkl'))
        for fname in file_list:
            with open(fname, 'rb') as f:
                feats = pickle.load(f)
                loaded_feats.update(feats)
        return loaded_feats

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        if 'SLURM_PROCID' in os.environ:
            tcs_conf_path = cfg.DATALOADER.get("TCS_CONF_PATH", "petreloss.config")
        else:
            # dev machine
            tcs_conf_path = "slurm_tools/petreloss_local.config"
        ann_files = {
            "train": [os.path.join(cfg.DATALOADER.ANNO_FOLDER, "captions_train113k.json"), os.path.join(cfg.DATALOADER.ANNO_FOLDER, "captions_val5k.json")],
            # no validation
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "captions_test5k.json")
        }
        if getattr(cfg.DATALOADER, 'TRANSFORM', None) == 'clip_transforms':
            transform = clip_transforms(stage,
                                        img_size=cfg.MODEL.IMG_INPUT_SIZE)
        else:
            transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))]
            )
        ret = {
            "cfg"            : cfg,
            "stage"          : stage,
            "anno_file"      : ann_files[stage],
            "seq_per_img"    : 1,
            "feats_folder"   : cfg.DATALOADER.FEATS_FOLDER,
            "relation_file"  : cfg.DATALOADER.RELATION_FILE,
            "gv_feat_file"   : cfg.DATALOADER.GV_FEAT_FILE,
            "attribute_file" : cfg.DATALOADER.ATTRIBUTE_FILE,
            "max_seq_len"    : cfg.MODEL.MAX_SEQ_LEN,
            "use_ceph"       : getattr(cfg.DATALOADER, 'USE_CEPH', False),
            "tcs_conf_path"  : tcs_conf_path,
            "transform"      : transform,
            'task_type'      : cfg.DATASETS.TASK_TYPE,
            "random_mask"    : getattr(cfg.DATALOADER, 'RANDOM_MASK', False),
            "data_percentage": cfg.DATALOADER.DATA_PERCENTAGE,
            "text_type_id"   : getattr(cfg.DATALOADER, 'TYPE_EMBEDDING_ID', 0),
        }

        ret['tokenizer'] = ClipTokenizer()
        ret['tokenizer_name']  = "clip"

        return ret

    def _preprocess_datalist(self, datalist):
        return datalist

    def load_data(self, cfg):
        if self.stage == "test":
            cache_path = os.path.join(
                os.path.dirname(self.anno_file), "cache",
                "mscoco_caption_w_testcap_%s.pkl" % ( self.stage)
            )
            if not os.path.exists(os.path.dirname(cache_path)):
                os.makedirs(os.path.dirname(cache_path))
            if not os.path.exists(cache_path):
                datalist = self.load_raw_data(cfg, self.anno_file)
                pickle.dump(datalist, open(cache_path, "wb"))
            datalist = pickle.load(open(cache_path, "rb"))
        else:
            datalist = list()
            assert self.stage == "train", "no validation now"
            for i, stage in enumerate(["train", "val"]):
                cache_path = os.path.join(
                    os.path.dirname(self.anno_file[i]), "cache",
                    "mscoco_caption_w_testcap_%s.pkl" % ( stage)
                )
                if not os.path.exists(os.path.dirname(cache_path)):
                    os.makedirs(os.path.dirname(cache_path))
                if not os.path.exists(cache_path):
                    datalist_part = self.load_raw_data(cfg, self.anno_file[i])
                    pickle.dump(datalist_part, open(cache_path, "wb"))
                datalist_part = pickle.load(open(cache_path, "rb"))
                datalist.extend(datalist_part)

        def _load_pkl_file(filepath):
            return pickle.load(open(filepath, 'rb'), encoding='bytes') if len(filepath) > 0 else None

        ext_data = {
            "relation": _load_pkl_file(self.relation_file),
            "attribute": _load_pkl_file(self.attribute_file),
            "gv_feat": _load_pkl_file(self.gv_feat_file)
        }
        for i in range(len(datalist)):
            image_id = int(datalist[i]['image_id'])
            for data_type in ext_data:
                if ext_data[data_type] is not None:
                    if str(image_id) in ext_data[data_type]:
                        datalist[i][data_type] = ext_data[data_type][str(image_id)]
                    elif image_id in ext_data[data_type]:
                        datalist[i][data_type] = ext_data[data_type][image_id]

        if self.data_percentage < 1.0 and self.stage == 'train':
            datalist = random.sample(datalist, k = int(self.data_percentage* len(datalist) )  )

        return datalist


    def load_raw_data(self, cfg, anno_file):
        datalist = []
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

        return datalist

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
            else:
                for sub_token in sub_tokens:
                    # no masking token (will be ignored by loss function later)
                    output_tokens.append(sub_token)
                    output_label.append(-1)

        return output_tokens, output_label

    def __len__(self):
        return len(self.database)

    # def __call__(self, dataset_dict):
    def __getitem__(self, index):
        for i_try in range(100):
            try:
                dataset_dict = self.database[index].as_py()
                image_id = dataset_dict['image_id']
                image_split = self.idx2name[int(image_id)]['split']
                image_name = self.idx2name[int(image_id)]['name']

                # load image
                image_path = os.path.join(self.feats_folder, image_split, image_name)

                if self.use_ceph:
                    img = self.tcs_loader(image_path).convert('RGB')

                else:
                    img = Image.open(image_path).convert("RGB")
                
                break  
            except Exception as e:
                print("Failed to load image from idb {} with error {} ; trial {};".format(self.database[index], e, i_try))
                index = (index + 1) % len(self.database)
                while (self.database[index].as_py() is None):
                    index = (index + 1) % len(self.database)
                continue



        img = self.transform(img)

        ret = {
            'input_sample': [{
                    'data'        : img,
                    'invalid_mask': None,
                    'modality'    : 'image',
                    'data_type': 'input',
                    'sample_info' :{'id': image_id, 'path': image_path}
                }]
        }

        self.target_set = self.cfg.DATASETS.TARGET_SET

        if self.task_type == 'image_caption' and self.stage != 'train':
            ret.update({
                'target_set': copy.deepcopy(self.target_set),
                'target_sample': [],
                'target_idx': [],
                'task_info'    : copy.deepcopy(self.task_info)
            })
            dict_as_tensor(ret)
            return ret





        if self.task_type =='image_retrieval' and self.stage != 'train':
            captions = [caption +  " <|endoftext|>" for caption in  dataset_dict['captions']]
            caption_tokens_raw = [ self.tokenizer.encode(caption) for caption in captions]

            caption_tokens = [ caption_token[:(self.max_seq_len - 1)] + [caption_token[-1]]
                              if len(caption_token) > self.max_seq_len else caption_token
                              for caption_token in caption_tokens_raw ]


        else:
            caption = random.sample(dataset_dict['captions'], self.seq_per_img)[0]
            # caption = ['pilgrims', 'coffee', 'house', '-', 'outside', 'the', 'store']
            caption = caption + " <|endoftext|>"

            if self.task_type == 'mlm':
                u_mask_type = 1
            elif self.task_type == 'image_caption':
                u_mask_type = 0 # causal mask

            if self.task_type=='image_caption' or self.task_type =='mlm':
                if u_mask_type == 1: # mlm
                    caption_tokens = self.tokenizer.basic_tokenize(caption)
                    caption_tokens, mlm_labels = self.random_word_wwm(caption_tokens)
                else:
                    # caption
                    caption_tokens = self.tokenizer.encode(caption)
                    mlm_labels = self.tokenizer.encode("<|spe|>")*len(caption_tokens)

            else:
                caption_tokens = self.tokenizer.encode(caption)

            if len(caption_tokens) > self.max_seq_len:
                # mlm task
                text_len_keep = self.max_seq_len
                caption_tokens = caption_tokens[:(text_len_keep - 1)] + [caption_tokens[-1]]
                if self.task_type=='image_caption' or self.task_type == 'mlm':
                    mlm_labels = mlm_labels[:(text_len_keep - 1)] + [mlm_labels[-1]]

        # self.task_info = {
        #         'task_type'      : self.cfg.DATASETS.TASK_TYPE,
        #         'dataset_name'   : self.cfg.DATASETS.DATASET_NAME,
        #         'batch_size'     : self.cfg.DATASETS.TRAIN_BATCH_SIZE
        #                           if self.stage == 'train' else self.cfg.DATASETS.TEST_BATCH_SIZE,
        #         'sampling_weight': self.cfg.DATALOADER.SAMPLING_WEIGHT
        #     }

        if self.task_type == 'image_caption':
            source = np.array(caption_tokens, dtype=np.int64)
            source2 = np.array(mlm_labels, dtype=np.int64)
            ret['input_sample'].append({
                'data'            :[source, source2],
                'invalid_mask'    : None,
                'modality'        : 'text',
                'data_type'       : 'input',
                'sample_info'     :
                {
                    'text_spe_cat': True,
                }
            })
            ret.update({
                "target_sample": [],
                "target_idx"   : [np.array(caption_tokens, dtype=np.int64)],
                "target_set"   : copy.deepcopy(self.target_set),
                'task_info'    : copy.deepcopy(self.task_info)
            })

        elif self.task_type == 'mlm':
            ret['input_sample'].append({
                    'data'        : [np.array(caption_tokens, dtype=np.int64)],
                    'invalid_mask': None,
                    'modality'    : 'text',
                    'data_type'   : 'input',
                    'sample_info' : {"text_token_padding_length": self.max_seq_len}
                })
            ret.update({
                'target_sample': [],
                "target_idx"   : [np.array(mlm_labels, dtype=np.int64)],
                "target_set"   : copy.deepcopy(self.target_set),
                'task_info'    : copy.deepcopy(self.task_info)
            })
        elif self.task_type == 'image_retrieval':
            if self.stage == 'train':
                ret.update({
                    'target_sample':   [{
                        'data'        : [np.array(caption_tokens, dtype=np.int64)],
                        'modality'    : 'text',
                        'invalid_mask': None,
                        'data_type'   : 'target',
                        'sample_info' : {}
                    }],
                    'target_idx'      : [],
                    'target_set'      : [],
                    'task_info'       : copy.deepcopy(self.task_info)
                })
            else:
                ret.update(
                    {
                        'input_sample': [{
                                'data'        : img, 'invalid_mask': None, 'modality': 'image', 'data_type': 'input',
                                'sample_info' : {
                                    'id'      : (image_id, [image_id] * len(caption_tokens)),
                                    'path'    : image_path
                                }
                            }],
                        'target_sample': [{
                                'data'        : [np.array(caption_token, dtype=np.int64)
                                                for caption_token in caption_tokens],
                                'modality'    : 'text',
                                'invalid_mask': None,
                                'data_type'   : 'target',
                                'sample_info' : {
                                    'sample_alone': True,
                                }

                            }],
                        'target_idx'          : [],
                        'target_set'          : [],
                        'task_info'           : copy.deepcopy(self.task_info)
                    }
                )
        else:
            raise NotImplementedError

        dict_as_tensor(ret)

        return ret
