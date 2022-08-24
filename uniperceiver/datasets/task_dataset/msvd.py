import os
import copy
import pickle
import random
import numpy as np
import torch
from uniperceiver.config import configurable
from uniperceiver.functional import read_np, dict_as_tensor
from ..build import DATASETS_REGISTRY
from uniperceiver.tokenization import ClipTokenizer
from torchvision.transforms import Compose, RandomApply, ToTensor, Normalize, CenterCrop, Lambda, RandomHorizontalFlip, ColorJitter, Resize, RandomCrop
from .video_transform import random_short_side_scale_jitter, uniform_crop
import json
from io import BytesIO
import av
from .video_raw import VideoDataSet
import io
from collections import defaultdict

import pyarrow as pa
from uniperceiver.utils import comm
import copy

__all__ = ["MSVDDataset"]

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
    highest_idx = video_frames - int(new_sampling_rate * frames_per_clip)
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
class MSVDDataset(VideoDataSet):
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: str,
        seq_per_img: int,
        max_feat_num: int,
        max_seq_len: int,
        feats_folder: str,
        tokenizer,
        tokenizer_name,
        use_ceph: bool,
        tcs_conf_path,
        frames_per_clip, interval, num_clips, timesformer_aug,
        task_type,
        data_percentage,
        target_fps=30,
        random_mask=False,
        cfg=None,
    ):
        self.cfg = cfg
        self.stage = stage
        self.anno_file = anno_file
        self.seq_per_img = seq_per_img
        self.max_feat_num = max_feat_num
        self.feats_folder = feats_folder
        self.max_seq_len = max_seq_len
        self.task_type = task_type

        self.initialized = False

        # sample_list = list(self.fin.keys())
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.use_clip_tokenizer = self.tokenizer_name == 'clip'
        # for index_maping
        self.idx2name = dict()
        self.name2idx = dict()

        self.use_ceph = use_ceph
        if isinstance(self.anno_file, list):
            self.cache_dir = os.path.join(os.path.dirname(self.anno_file[0]), 'cache')
        else:
            self.cache_dir = os.path.join(os.path.dirname(self.anno_file), 'cache')
        self.frames_per_clip = frames_per_clip
        self.interval = interval
        
        # self.MULTI_VEIW = self.cfg.DATALOADER.get('MULTI_VEIW', 'v0')
        # self.MULTI_VEIW_NUM = self.cfg.DATALOADER.get('MULTI_VEIW_NUM', 1)
        self.random_stride = self.cfg.DATALOADER.get('RANDON_STRIDE', False)

        self.num_clips = num_clips
        self.is_train = stage == 'train'
        self.test_mode = stage != 'train'
        self.transform = self._timesformer_transform() if timesformer_aug else self._transform()
        self.target_fps = target_fps
        self.data_percentage = data_percentage

        if self.use_ceph:
            self.feats_folder = 's3://msvd/YouTubeClips/'
            if isinstance(self.anno_file, list):
                self.anno_file = [os.path.join('s3://msvd/annotations/', os.path.basename(anno_file)) for anno_file in self.anno_file]
            else:
                self.anno_file = os.path.join('s3://msvd/annotations/', os.path.basename(self.anno_file))
            print('debug info for msvd pretrain: {} '.format(self.feats_folder))
            from uniperceiver.datasets import TCSLoader
            if 'SLURM_PROCID' in os.environ:
                self.tcs_loader = TCSLoader(tcs_conf_path)
            else:
                self.tcs_loader = TCSLoader('slurm_tools/petreloss_local.config')
        else:
            # local image folder
            self.feats_folder = feats_folder

        if self.use_ceph:
            if isinstance(self.anno_file, list):
                videoinfo = list()
                for anno_file in self.anno_file:
                    videoinfo.extend(json.load(BytesIO(self.tcs_loader.client.get(anno_file)))["images"])
            else:
                videoinfo = json.load(BytesIO(self.tcs_loader.client.get(self.anno_file)))["images"]
        else:
            if isinstance(self.anno_file, list):
                videoinfo = list()
                for anno_file in self.anno_file:
                    videoinfo.extend(json.load(open(anno_file))["images"])
            else:
                videoinfo = json.load(open(self.anno_file))["images"]
        for vinfo in videoinfo:
            self.idx2name[vinfo['id']] = vinfo['file_name']
            self.name2idx[vinfo['file_name']] = vinfo['id']
        self.random_mask = random_mask
        pass

        _temp_list =self.load_data(self.cfg)
        self.video_list = pa.array(_temp_list)
        if comm.is_main_process():
            import sys
            print(f"!!! Dataset {self.cfg.DATASETS.DATASET_NAME} with task {self.cfg.DATASETS.TASK_TYPE}:")
            print('!!! length of _temp_list: ', len(_temp_list))
            print('!!! size of _temp_list: ', sys.getsizeof(_temp_list))
            print('!!! size of pa database: ', sys.getsizeof(self.video_list))
        del _temp_list

        self.task_info = {
            'task_type'      : self.cfg.DATASETS.TASK_TYPE,
            'dataset_name'   : self.cfg.DATASETS.DATASET_NAME,
            'batch_size'     : self.cfg.DATALOADER.TRAIN_BATCH_SIZE if self.stage == 'train' else self.cfg.DATALOADER.TEST_BATCH_SIZE,
            'sampling_weight': self.cfg.DATALOADER.SAMPLING_WEIGHT
        }

        self.target_set = self.cfg.DATASETS.TARGET_SET


    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        if stage == "train":
            ann_file = [os.path.join(cfg.DATALOADER.ANNO_FOLDER, "caption_msvd_train_cocostyle.json"),
                        os.path.join(cfg.DATALOADER.ANNO_FOLDER, "caption_msvd_val_cocostyle.json")]
        else:
            assert stage == "test"
            ann_file = os.path.join(cfg.DATALOADER.ANNO_FOLDER, "caption_msvd_{}_cocostyle.json".format(stage))
        feat_path = os.path.join(cfg.DATALOADER.FEATS_FOLDER, "MSVD_ResNet152_{}.hdf5".format(stage))

        if 'SLURM_PROCID' in os.environ:
            tcs_conf_path = cfg.DATALOADER.get("TCS_CONF_PATH", "slurm_tools/petreloss.config")
        else:
            # dev machine
            tcs_conf_path = "slurm_tools/petreloss_local.config"

        ret = {
            "stage": stage,
            "anno_file": ann_file,
            "seq_per_img": cfg.DATALOADER.SEQ_PER_SAMPLE,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "feats_folder": feat_path,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
            "use_ceph": getattr(cfg.DATALOADER, 'USE_CEPH', False),
            "tcs_conf_path": tcs_conf_path,
            'task_type': cfg.DATASETS.TASK_TYPE,
            "frames_per_clip": cfg.DATALOADER.FRAMES_PER_CLIP,
            "interval": cfg.DATALOADER.STRIDE,
            "num_clips": 1 if stage == 'train' else cfg.INFERENCE.NUM_VIEWS,
            "timesformer_aug": cfg.DATALOADER.TIMESFORMER_AUG,
            "data_percentage": cfg.DATALOADER.DATA_PERCENTAGE,
            "cfg": cfg,
        }
        if getattr(cfg.INFERENCE, "VOCAB", None) == 'CLIP':
            ret['tokenizer'] = ClipTokenizer()
            ret['tokenizer_name']  = "clip"
        else:
            raise NotImplementedError
        return ret

    def load_data(self, cfg):
        if self.stage == "train":
            total_datalist = list()
            for i, stage in enumerate(["train", "val"]):
                cache_path = os.path.join(
                    self.cache_dir,
                    "msvd_raw_caption_retrieval_%s_%s_%d.pkl" % (self.tokenizer_name, stage, self.max_seq_len)
                )
                if not os.path.exists(os.path.dirname(cache_path)):
                    os.makedirs(os.path.dirname(cache_path))
                if not os.path.exists(cache_path):
                    datalist = self.load_raw_data(cfg, self.anno_file[i])
                    pickle.dump(datalist, open(cache_path, "wb"))
                datalist = pickle.load(open(cache_path, "rb"))
                if isinstance(datalist[0]['caption'], list):
                    new_datalist = list()
                    for data in datalist:
                        if isinstance(data['caption'], str):
                            new_datalist.append(data)
                        else:
                            video_id = data['video_id']
                            for caption in data['caption']:
                                new_datalist.append({
                                    "video_id": video_id,
                                    "caption": caption,
                                })
                    datalist = new_datalist
                total_datalist.extend(datalist)

            if self.data_percentage < 1.0 and self.stage == 'train':
                datalist = random.sample(total_datalist, k = int(self.data_percentage* len(total_datalist) )  )
                total_datalist = datalist

        else:
            assert self.stage == "test"
            cache_path = os.path.join(
                self.cache_dir,
                "msvd_raw_caption_retrieval_%s_%s_%d.pkl" % (self.tokenizer_name, self.stage, self.max_seq_len)
            )
            if not os.path.exists(os.path.dirname(cache_path)):
                os.makedirs(os.path.dirname(cache_path))
            if not os.path.exists(cache_path):
                datalist = self.load_raw_data(cfg, self.anno_file)
                pickle.dump(datalist, open(cache_path, "wb"))
            datalist = pickle.load(open(cache_path, "rb"))
            total_datalist = datalist
        return total_datalist


    def load_raw_data(self, cfg, anno_file):
        datalist = []
        if self.stage == 'train':
            if self.use_ceph:
                annoinfo = json.load(BytesIO(self.tcs_loader.client.get(anno_file)))
            else:
                annoinfo = json.load(open(anno_file))
            captions_train = sorted( annoinfo['annotations'], key=lambda x: x['id'])
            for data in captions_train:
                datalist.append(
                    {
                        'video_id': data['image_id'],
                        'caption': data['caption']
                    }
                )

        else:
            if self.use_ceph:
                annoinfo = json.load(BytesIO(self.tcs_loader.client.get(self.anno_file)))
            else:
                annoinfo = json.load(open(self.anno_file))
            captions_train = sorted( annoinfo['annotations'], key=lambda x: x['id'])
            video2caps = defaultdict(list)
            for data in captions_train:
                video2caps[data['image_id']].append(data['caption'])

            for videoid, caps in video2caps.items():
                datalist.append(
                    {
                        'video_id': videoid,
                        'caption': caps
                    }
                )
        return datalist

    def _timesformer_transform(self):
        transforms = [
            Lambda(lambda frames: torch.stack([ToTensor()(frame.convert("RGB")) for frame in frames])),
        ]
        if self.test_mode:
            test_scale = self.cfg.MODEL.IMG_INPUT_SIZE
            transforms.extend([
                Lambda(lambda frames: random_short_side_scale_jitter(
                    frames, test_scale, test_scale)[0]),
                CenterCrop(test_scale),
                # Lambda(lambda images: torch.stack([uniform_crop(images, 224, i)[0] for i in range(3)], 0))
            ])
        else:
            min_scale = int((256 / 224)*self.cfg.MODEL.IMG_INPUT_SIZE)
            max_scale = int((320 / 224)*self.cfg.MODEL.IMG_INPUT_SIZE)

            transforms.extend([
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

    def _sample_frame(self, atten_feats):
        interval = atten_feats.shape[0] / self.max_feat_num
        selected_indexes = [int(i * interval) for i in range(self.max_feat_num)]
        selected_frames = atten_feats[selected_indexes, :]
        return selected_frames

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

        # if no word masked, random choose a word to mask
        # if all([l_ == -1 for l_ in output_label]):
        #    choosed = random.randrange(0, len(output_label))
        #    output_label[choosed] = self.tokenizer.vocab[tokens[choosed]]

        return output_tokens, output_label


    def __getitem__(self, idx):
        
        for i_try in range(100):
            try:
                record = self.video_list[idx].as_py()
                record = copy.deepcopy(record)
                video_id = record['video_id']
                # load video

                video_path = os.path.join(self.feats_folder, self.idx2name[video_id] + '.avi')
                if self.use_ceph:
                    container = av.open(io.BytesIO(self.tcs_loader.client.get(video_path)))
                else:
                    container = av.open(video_path)


                # container.streams.video[0].thread_type = "AUTO"
                stream = container.streams.video[0]
                total_frames = stream.frames
                fps = float(container.streams.video[0].average_rate)

                if total_frames == 0:
                    # it returns 0 if not know, but that doesn't mean the video is null
                    for frame in container.decode(stream):
                        total_frames += 1
                    container.close()
                    container = av.open(video_path)
                    stream = container.streams.video[0]
            except Exception as e:
                print(
                    "Failed to load video from {} with error {} ; trial {}".format(
                        video_path, e, i_try
                    )
                )

                # let's try another one
                index = random.randint(0, len(self.data_list) - 1)
                record = self.data_list[index]
                continue

            if self.stage=='train':
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

            if video_data.dim() == 4:
                video_data.unsqueeze_(0) # in case there is only one frame

            ret = {
                'input_sample': [{
                    'data': video_data, 'invalid_mask': None, 'modality': 'video', 'data_type': 'input',
                    'sample_info':{
                        'id': video_id,
                        'path': video_path,
                        'num_views':num_frames,
                        'cat_along_first_dim': True,
                        }
                }]
            }

            if self.stage == 'train' and record['caption'] is not None:
                caption = record['caption']
                caption = caption + " <|endoftext|>"

                if self.task_type == 'video_mlm':
                    u_mask_type = 1
                elif self.task_type == 'video_caption':
                    u_mask_type = 0 # causal mask

                if self.task_type=='video_caption' or self.task_type =='video_mlm':
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
                    if self.task_type == 'video_caption' or self.task_type == 'video_mlm':
                        mlm_labels = mlm_labels[:(text_len_keep - 1)] + [mlm_labels[-1]]

                if self.task_type == 'video_caption':

                    source = np.array(caption_tokens, dtype=np.int64)
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
                        'target_idx'   : [np.array(caption_tokens, dtype=np.int64)],
                        'target_set'   : copy.deepcopy(self.target_set),
                        'task_info'    : copy.deepcopy(self.task_info)
                    })

                elif self.task_type == 'video_mlm':

                    raise NotImplementedError('no needed for masked language modeling when given video now.')


                elif self.task_type == 'video_retrieval':
                    ret.update({
                    'target_sample': [{
                        'data'        : [np.array(caption_tokens, dtype=np.int64)],
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
                    raise NotImplementedError

            elif self.stage != 'train':
                if self.task_type == 'video_caption':
                    ret.update({
                        'target_set': copy.deepcopy(self.target_set),
                        'target_sample': [],
                        'target_idx': [],
                        'task_info'    : copy.deepcopy(self.task_info)
                    })
                elif self.task_type=='video_retrieval':
                    captions = [caption +  " <|endoftext|>" for caption in  record['caption']]
                    caption_tokens_raw = [ self.tokenizer.encode(caption) for caption in captions]

                    caption_tokens = [ caption_token[:(self.max_seq_len - 1)] + [caption_token[-1]]
                                    if len(caption_token) > self.max_seq_len else caption_token
                                    for caption_token in caption_tokens_raw ]
                    ret.update(
                        {
                            'input_sample': [{
                                    'data'        : video_data, 'invalid_mask': None, 'modality': 'video', 'data_type': 'input',
                                    'sample_info' : {
                                        'id'      : (video_id, [video_id] * len(caption_tokens)),
                                        'path'    : video_path,
                                        'num_views':num_frames,
                                        'cat_along_first_dim': True,
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
