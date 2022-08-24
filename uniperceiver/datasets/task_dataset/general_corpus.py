from uniperceiver.functional import read_np, dict_as_tensor, boxes_to_locfeats
import random
import numpy as np
import copy
from torch.utils.data import Dataset
from uniperceiver.tokenization import ClipTokenizer
import logging
import os
from ..build import DATASETS_REGISTRY
from uniperceiver.config import configurable
import pickle
from uniperceiver.utils import comm

__all__ = ["GeneralCorpusDataset"]


@DATASETS_REGISTRY.register()
class GeneralCorpusDataset(Dataset):
    @configurable
    def __init__(self, ann_file, stage,
                 tokenizer, tokenizer_name,
                 seq_len=64, min_seq_len=64,
                 encoding="utf-8",
                 cache_mode=True, cache_local_rank=0, cache_local_size=1,
                 append_eos=False,
                 one_stream=False,
                 random_mask=False,
                 task_type=None,
                 text_type_id=0,
                 mask_bpe_word='spe',
                 version='v1',
                 task_info=None,
                 target_set=None,
                 **kwargs):
        assert cache_mode, print("only support cache mode!")
        assert len(task_type) > 0
        self.version = version
        self.stage = stage
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.use_clip_tokenizer = tokenizer_name == 'clip'
        self.task_type = task_type
        self.append_eos = append_eos
        self.task_info = task_info
        self.target_set = target_set

        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.cache_mode = cache_mode
        self.cache_local_size = cache_local_size
        self.cache_local_rank = cache_local_rank

        self.ann_file = ann_file
        self.encoding = encoding
        self.test_mode = False
        self.random_mask = random_mask

        self.one_stream = one_stream

        self.text_type_id = text_type_id

        self.mask_bpe_word = "<|spe|>" if mask_bpe_word == 'spe' else '<|startoftext|>'

        # load samples into memory
        if cache_mode:
            print('dataset cache mode is ON: local size: {}; local rank: {}'.format(cache_local_size,
                                                                                    cache_local_rank))
            self.corpus, self.cursor = self.load_corpus()

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        version = getattr(cfg.DATASETS, 'VERSION', 'v1')
        if 'SLURM_PROCID' not in os.environ:
            version = 'v1'
        if version == 'v2':
            ann_files = {
            "train":
            os.path.join(cfg.DATALOADER.ANNO_FOLDER, "bookswiki_v2.txt")
            if comm.get_world_size() > 1 else os.path.join(
                cfg.DATALOADER.ANNO_FOLDER, "bookswiki_v2-1000.doc"),
            "val":
            os.path.join(cfg.DATALOADER.ANNO_FOLDER, "bookswiki_v2-1000.doc")
            }
        elif version == 'v3':
            ann_files = {
            "train":
            os.path.join(cfg.DATALOADER.ANNO_FOLDER, "bookswikiopen.txt")
            if comm.get_world_size() > 1 else os.path.join(
                cfg.DATALOADER.ANNO_FOLDER, "bookswiki_v2-1000.doc"),
            "val":
            os.path.join(cfg.DATALOADER.ANNO_FOLDER, "bookswiki_v2-1000.doc")
            }
        else:
            ann_files = {

                "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "bookswiki.doc") if comm.get_world_size() > 1 else
                os.path.join(cfg.DATALOADER.ANNO_FOLDER, "bookswiki-1000.doc"),
                "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "bookswiki-1000.doc")
            }

        task_info = {
            'task_type'     : cfg.DATASETS.TASK_TYPE,
            'dataset_name'  : cfg.DATASETS.DATASET_NAME,
            'batch_size'    : cfg.DATALOADER.TRAIN_BATCH_SIZE if stage == 'train' else cfg.DATALOADER.TEST_BATCH_SIZE,
            'sampling_weight': cfg.DATALOADER.SAMPLING_WEIGHT
        }

        ret = {
            "version"         : version,
            "stage"           : stage,
            "ann_file"        : ann_files[stage],
            "seq_len"         : cfg.MODEL.MAX_SEQ_LEN,
            "min_seq_len"     : cfg.MODEL.MAX_SEQ_LEN,
            "cache_mode"      : cfg.DATALOADER.CACHE_MODE,
            "append_eos"      : cfg.DATALOADER.APPEND_EOS,
            "cache_local_rank": comm.get_local_rank(),
            "cache_local_size": comm.get_local_size(),
            "one_stream"      : cfg.DATALOADER.ONE_STREAM,
            "task_type"       : cfg.DATASETS.TASK_TYPE,
            "random_mask"     : getattr(cfg.DATALOADER, 'RANDOM_MASK', False),
            "text_type_id"    : getattr(cfg.DATALOADER, 'TYPE_EMBEDDING_ID', 0),
            "mask_bpe_word"   : getattr(cfg.DATALOADER, 'MASK_BPE_WORD', 'spe'),
            "task_info"       : task_info,
            "target_set"      : cfg.DATASETS.TARGET_SET
        }


        ret['tokenizer'] = ClipTokenizer()
        ret['tokenizer_name']  = "clip"


        return ret

    @classmethod
    def add_config(cls, cfg):
        cfg.DATALOADER.SAMPLER = "NodeDistributed"
        cfg.DATALOADER.CACHE_MODE = True
        cfg.DATALOADER.SEQ_PER_SAMPLE = 256
        cfg.DATALOADER.MIN_SEQ_PER_SAMPLE = 256
        cfg.DATALOADER.APPEND_EOS = True


    def load_corpus(self):
        if 'SLURM_PROCID' in os.environ:
            self.cache_local_size = 8 # for convenice
        cache_path = os.path.dirname(self.ann_file)
        if self.version == 'v2':
            cache_filename = 'cache/cache_block' + os.path.basename(self.ann_file).replace('.', "_") + "_" + str(self.cache_local_rank) + "_" + str(self.cache_local_size) + '.pkl'
        elif self.version == 'v3':
            cache_filename = 'cache_v3/cache_block_books_wiki_openweb'  + "_" + str(self.cache_local_rank) + "_" + str(self.cache_local_size) + '.pkl'
        else:
            cache_filename = 'cache_block' + os.path.basename(self.ann_file).replace('.', "_") + "_" + str(self.cache_local_rank) + "_" + str(self.cache_local_size) + '.pkl'
        cache_file = os.path.join(cache_path, cache_filename)
        if not os.path.exists(cache_file):
            if self.version == 'v3':
                raise NotImplementedError
            # [HACK] we hard code the  corpus length
            if 'SLURM_PROCID' in os.environ:
                if self.version == 'v2':
                    self.file_len = 244208263
                    block_size  = (self.file_len + self.cache_local_size - 1)// self.cache_local_size
                    block_start = block_size * self.cache_local_rank
                    block_end = (block_size) * (
                        1 + self.cache_local_rank
                    ) if self.cache_local_rank + 1 < self.cache_local_size else self.file_len
                else:
                    block_start = self.cache_local_rank * 13000000
                    block_end = ( self.cache_local_rank  + 1 ) * 13000000
            else:
                block_start = 0
                block_end = 1000
            count = 0
            corpus = bytearray()
            cursor = []
            c_ = 0
            i_ = 0
            for ann_file in self.ann_file.split('+'):
                with open(ann_file, 'r', encoding=self.encoding) as f:
                    for l in f:
                        l = l.strip()
                        if l != '':
                            # if i_ % self.cache_local_size != self.cache_local_rank:
                            if i_< block_start or i_ >= block_end:
                                # cursor.append(c_)
                                i_ += 1
                                continue
                            l = l.encode()
                            corpus += l
                            cursor.append(c_)
                            c_ += len(l)
                            i_ += 1
                            count += 1
            cursor.append(len(corpus))
            cursor = np.array(cursor).astype(np.int, copy=False)
            pickle.dump({
                "corpus": corpus,
                "cursor": cursor,
                "count": count,
                }, open(cache_file, "wb"), protocol=4)

        else:
            cachedata = pickle.load(open(cache_file, "rb"))
            corpus, cursor, count = cachedata['corpus'], cachedata['cursor'],  cachedata['count']

        print("BooksWiki info: rank {} has {} sentences".format(self.cache_local_rank, count))

        return corpus, cursor


    def get_line(self, index):
        return self.corpus[self.cursor[index]:self.cursor[index+1]].decode()

    @property
    def data_names(self):
        return ['text', 'mlm_labels']

    def __len__(self):
        return len(self.cursor) - 1

    def __getitem__(self, item):
        # def __call__(self, item):
        raw = self.get_line(item)

        # tokenize
        if self.use_clip_tokenizer:
            tokens = self.tokenizer.basic_tokenize(raw)
            if len(tokens) > 0 and self.append_eos:
                tokens.append('<|endoftext|>')
        else:
            tokens = self.tokenizer.basic_tokenizer.tokenize(raw)

        # add more tokens if len(tokens) < min_len
        _cur = (item + 1) % (len(self.cursor) - 1)
        while len(tokens) < self.min_seq_len:
            if self.use_clip_tokenizer:
                _cur_tokens = self.tokenizer.basic_tokenize(self.get_line(_cur))
                if len(_cur_tokens) > 0 and self.append_eos:
                    _cur_tokens.append('<|endoftext|>')
            else:
                _cur_tokens = self.tokenizer.basic_tokenizer.tokenize(self.get_line(_cur))
            tokens.extend(_cur_tokens)
            _cur = (_cur + 1) % (len(self.cursor) - 1)

        if self.task_type == 'text_mlm':
            tokens, mlm_labels = self.random_word_wwm(tokens)

        elif self.task_type == 'caption':
            tokens_tmp = []
            for token in tokens:
                tokens_tmp.extend(self.tokenizer.encode_basic_tokenized_token(token))
            tokens = tokens_tmp
            mlm_labels = self.tokenizer.encode(
                self.mask_bpe_word) * len(tokens)

        if self.use_clip_tokenizer:
            ids = tokens
        else:
            # add [CLS], [SEP]
            tokens =   tokens + ['[SEP]']
            mlm_labels = mlm_labels + [-1]

            # convert token to its vocab id
            ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # truncate
        if len(ids) > self.seq_len:
            ids = ids[:(self.seq_len-1)] + [ids[-1]]
            mlm_labels = mlm_labels[:(self.seq_len-1)] + [mlm_labels[-1]]
        elif len(ids) < self.seq_len:
            ids = ids + [0 for _ in range(self.seq_len - len(ids))]
            mlm_labels = mlm_labels + [-1 for _ in range(self.seq_len - len(ids))]




        if self.task_type == 'text_mlm':
            ret = {
                'input_sample': [{
                        'data'        : [np.array(ids, dtype=np.int64)],
                        'invalid_mask': None,
                        'modality'    : 'text',
                        'data_type': 'input',
                        'sample_info' : {
                            'seq_length': len(ids)
                        }
                    }],
                'target_sample': [],
                'target_idx'   : [np.array(mlm_labels, dtype=np.int64)],
                'target_set'   : copy.deepcopy(self.target_set),
                'task_info'    : copy.deepcopy(self.task_info)
            }
        elif self.task_type == 'caption':
            source = np.array(ids, dtype=np.int64)
            source2 = np.array(mlm_labels, dtype=np.int64)

            ret = {
                'input_sample': [{
                        'data': [source, source2],
                        'invalid_mask': None,
                        'modality': 'text',
                        'data_type': 'input',
                        'sample_info': {}
                    }],
                'target_sample': [],
                'target_idx': [np.array(ids, dtype=np.int64)],
                'target_set'   : copy.deepcopy(self.target_set),
                'task_info'    : copy.deepcopy(self.task_info)
            }

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
                            output_tokens.append(
                                self.tokenizer.encoder[self.mask_bpe_word])
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