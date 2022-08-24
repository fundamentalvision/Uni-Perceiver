import os
import copy
import pickle
import random
import json
import glob
import numpy as np
from uniperceiver.config import configurable
from uniperceiver.functional import dict_as_tensor
from uniperceiver.tokenization import  ClipTokenizer
from ..build import DATASETS_REGISTRY
import pyarrow as pa

__all__ = ["GLUEDataset"]


@DATASETS_REGISTRY.register()
class GLUEDataset:
    @configurable
    def __init__(
        self,
        cfg: dict,
        stage: str,
        anno_file: str,
        max_seq_len: int,
        tokenizer,
        tokenizer_name,
        input_columns,
        label_column,
        input_count,
        task_name,
        data_percentage,
        data_k_sample,
    ):
        self.cfg = cfg
        self.stage = stage
        self.anno_file = anno_file
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len

        self.input_columns = input_columns
        self.label_column = label_column
        self.input_count = input_count

        self.task_name = task_name

        self.data_percentage = data_percentage
        self.data_k_sample = data_k_sample

        self.task_info = {
                'task_type'      : self.cfg.DATASETS.TASK_TYPE,
                'dataset_name'   : self.cfg.DATASETS.DATASET_NAME,
                'batch_size'     : self.cfg.DATALOADER.TRAIN_BATCH_SIZE if self.stage == 'train' else self.cfg.DATALOADER.TEST_BATCH_SIZE,
                'sampling_weight': self.cfg.DATALOADER.SAMPLING_WEIGHT,
            }
        self.target_set = cfg.DATASETS.TARGET_SET

        self.load_data(cfg)

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        task_name = cfg.DATASETS.DATASET_NAME
        namesmapping = {
            "train": "train",
            "val": "dev",
            "test": "test",
        }
        data_dir = cfg.DATALOADER.ANNO_FOLDER
        if task_name in ['MNLI', 'QNLI', 'QQP', 'RTE', 'SST-2', 'MRPC', 'CoLA', 'STS-B']:
            anno_file = os.path.join(data_dir, task_name, 'processed/{name}.tsv'.format(name=namesmapping[stage]))
        elif task_name == 'MNLI_Match':
            namesmapping = {
                "train": "train",
                "val": "dev_matched",
                "test": "test_matched",
            }
            anno_file = os.path.join(data_dir, 'MNLI', 'processed/{name}.tsv'.format(name=namesmapping[stage]))
        elif task_name == 'MNLI_Mismatch':
            namesmapping = {
                "train": "train",
                "val": "dev_mismatched",
                "test": "test_mismatched",
            }
            anno_file = os.path.join(data_dir, 'MNLI', 'processed/{name}.tsv'.format(name=namesmapping[stage]))

        input_count = 2
        if task_name == "QQP":
            input_columns = [4, 5]
            if stage == 'test':
                input_columns = [2, 3]
            label_column = 6
        elif task_name in ["MNLI_Match", "MNLI_Mismatch"]:  # "MNLI" :
            input_columns = [9, 10]
            if stage == 'test':
                input_columns = [9, 10]

            label_column = 12
            if stage == 'val':
                label_column = 16
        elif task_name == "QNLI":
            input_columns = [2, 3]
            if stage == 'test':
                input_columns = [2, 3]
            label_column = 4
        elif task_name == "MRPC":
            input_columns = [4, 5]
            if stage == 'test':
                input_columns = [4, 5]
            label_column = 1
        elif task_name == "RTE":
            input_columns = [2, 3]
            if stage == 'test':
                input_columns = [2, 3]
            label_column = 4
        elif task_name == "STS-B":
            input_columns = [8, 9]
            if stage == 'test':
                input_columns = [8, 9]
            label_column = 10
        # Following are single sentence tasks.
        elif task_name == "SST-2":
            input_columns = [1]
            if stage == 'test':
                input_columns = [2]
            label_column = 2
            input_count = 1
        elif task_name == "CoLA":
            input_columns = [4]
            if stage == 'test':
                input_columns = [2]
            label_column = 2
            input_count = 1
        else:
            raise NotImplementedError

        ret = {
            "cfg": cfg,
            "stage": stage,
            "anno_file": anno_file,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
            "input_columns": input_columns,
            "label_column": label_column,
            "input_count": input_count,
            "task_name": task_name,
            "data_percentage": getattr(cfg.DATALOADER, "DATA_PERCENTAGE", 1.0),
            "data_k_sample": getattr(cfg.DATALOADER, "DATA_K_SAMPLE", -1),
            "tokenizer": ClipTokenizer(),
            "tokenizer_name": "clip"
        }

        return  ret



    def load_data(self, cfg):
        cache_path = os.path.join(os.path.dirname(self.anno_file), "cache_GLUE_raw_%s_%s_%s.pkl" % (self.task_name, self.tokenizer_name, self.stage))
        if not os.path.exists(cache_path):
            datalist = self.load_raw_data(cfg)

            pickle.dump(datalist, open(cache_path, "wb"))

        datalist = pickle.load(open(cache_path, "rb"))

        # for few shot exp

        if self.data_percentage < 1.0 and self.stage == "train":
            print("will sample {} data for trianing-->".format(self.data_percentage))
            labels2l = dict()
            for data in datalist:

                label = data['label']
                if label not in labels2l:
                    labels2l[label] = list()
                labels2l[label].append(data)

            # samplers_label = len(datalist) * self.data_percentage // len(labels2l.keys())
            datalist = []

            for v in labels2l.values():
                datalist.extend(random.sample(v, k=int(self.data_percentage * len(v) + 1)))
                # datalist.extend(random.sample(v, k=int(samplers_label+1)))

        elif self.data_k_sample > 0 and self.stage == "train":
            print("will sample {} data for each class when training -->".format(self.data_k_sample))
            labels2l = dict()
            for data in datalist:

                label = data['label']
                if label not in labels2l:
                    labels2l[label] = list()
                labels2l[label].append(data)

            datalist = []

            for v in labels2l.values():
                datalist.extend(random.sample(v, k=int(self.data_k_sample)))

        while len(datalist) < 200:
            datalist = datalist + datalist

        self.datalist = datalist


    def load_raw_data(self, cfg):
        datalist = []
        if self.task_name.startswith("MNLI"):
            labelmapping = {
                "contradiction": 0,
                "neutral": 1,
                "entailment": 2,
            }
        fin = open(self.anno_file, 'r').readlines()
        for _, line in enumerate(fin):
            sensinfo = line.strip().split('\t')
            if self.task_name == "RTE":
                label = 1.0 if sensinfo[self.label_column - 1] == "entailment" else 0.0
            elif self.task_name.startswith("MNLI"):
                label = labelmapping[sensinfo[self.label_column - 1]]
            elif self.task_name == "QNLI":
                label = 1.0 if sensinfo[self.label_column - 1] == "entailment" else 0.0
            elif self.task_name == "STS-B":
                label = float(sensinfo[self.label_column - 1]) / 5.0
            else:
                label = float(sensinfo[self.label_column - 1])
            datalist.append({
                # start index from 1 to 0
                "sentences": [sensinfo[i - 1] for i in self.input_columns],
                "label": label
            })
        return datalist
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        dataset_dict = copy.deepcopy(self.datalist[index])

        sentences = dataset_dict['sentences']

        # input1: SEN1, this sentence is (spe)  input2: word choice: postive and negative

        if self.input_count == 1:

          
            if self.task_name == "SST-2":
                tokens = self.tokenizer.encode(sentences[0] + "  <|endoftext|> It is <|spe|>.   <|endoftext|>")
            elif self.task_name == "CoLA":
                tokens = self.tokenizer.encode(sentences[0] + " This is <|spe|>. <|endoftext|>")
            else:
                raise NotImplementedError

            index = len(tokens) - 3
            assert index < self.max_seq_len
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len - 4] + tokens[-4:]



        else:

            if self.task_name in ["RTE"]:
                tokens1 = self.tokenizer.encode(sentences[0])
                if tokens1[-1] == 269:
                    tokens1 = tokens1[:-1]
                tokens1 = tokens1 + self.tokenizer.encode(" ? <|endoftext|> it is  ")
                tokens2 = self.tokenizer.encode(sentences[1] + " <|endoftext|> ")

                tokens2 = self.tokenizer.encode("  <|spe|> , ") + tokens2
                if len(tokens2) > self.max_seq_len // 2:
                    tokens2 = tokens2[:self.max_seq_len // 2 - 1] + [tokens2[-1]]
                max_len = self.max_seq_len - len(tokens2)

            elif self.task_name in ["MRPC"]:
                tokens1 = self.tokenizer.encode(sentences[0])
                if tokens1[-1] == 269:
                    tokens1 = tokens1[:-1]
                tokens1 = tokens1 + self.tokenizer.encode(" . ")
                tokens2 = self.tokenizer.encode(sentences[1] + " <|endoftext|> ")

                tokens2 = self.tokenizer.encode("  <|spe|> , ") + tokens2
                if len(tokens2) > self.max_seq_len // 2:
                    tokens2 = tokens2[:self.max_seq_len // 2 - 1] + [tokens2[-1]]
                max_len = self.max_seq_len - len(tokens2)

            elif self.task_name in ["QQP"]:
                tokens1 = self.tokenizer.encode(sentences[0])
                if tokens1[-1] == 269:
                    tokens1 = tokens1[:-1]
                tokens1 = tokens1 + self.tokenizer.encode(" <|endoftext|> ")
                tokens2 = self.tokenizer.encode(sentences[1] + " <|endoftext|> ")

                tokens2 = self.tokenizer.encode("  <|spe|> , ") + tokens2
                if len(tokens2) > self.max_seq_len // 2:
                    tokens2 = tokens2[:self.max_seq_len // 2 - 1] + [tokens2[-1]]
                max_len = self.max_seq_len - len(tokens2)

            elif self.task_name in ["QNLI"]:
                tokens1 = self.tokenizer.encode(sentences[0])
                if tokens1[-1] == 269:
                    tokens1 = tokens1[:-1]
                tokens1 = tokens1 + self.tokenizer.encode("  <|endoftext|> it is ")
                tokens2 = self.tokenizer.encode(sentences[1] + " <|endoftext|> ")

                tokens2 = self.tokenizer.encode("  <|spe|> , ") + tokens2
                if len(tokens2) > self.max_seq_len - len(tokens1):
                    tokens2 = tokens2[:self.max_seq_len - len(tokens1) - 1] + [tokens2[-1]]
                max_len = self.max_seq_len - len(tokens2)

            elif self.task_name in ["MNLI", "MNLI_Match"]:
                # sentence0 = sentences[0].replace(")", "").replace("(", "")
                tokens1 = self.tokenizer.encode(sentences[0])
                # if tokens1[-1] == 269:
                #     tokens1 = tokens1[:-1]
                tokens1 = tokens1  # + self.tokenizer.encode(" ? ")
                tokens2 = self.tokenizer.encode(sentences[1] + " <|endoftext|> ")

                tokens2 = self.tokenizer.encode("  <|spe|> , ") + tokens2
                if len(tokens2) > self.max_seq_len // 2:
                    tokens2 = tokens2[:self.max_seq_len // 2 - 1] + [tokens2[-1]]
                max_len = self.max_seq_len - len(tokens2)

            elif self.task_name in ["RTE", "QNLI", "MNLI", "MNLI_Match"]:
                tokens1 = self.tokenizer.encode(sentences[0] + "? <|endoftext|>")
                tokens2 = self.tokenizer.encode(sentences[1] + " <|endoftext|> ")
       
                if tokens1[-1] == 269:
                    tokens1 = tokens1[:-1]
                tokens2 = self.tokenizer.encode("   <|spe|> , ") + tokens2
                if len(tokens2) > self.max_seq_len // 2:
                    tokens2 = tokens2[:self.max_seq_len // 2 - 1] + [tokens2[-1]]
                max_len = self.max_seq_len - len(tokens2)
            elif self.task_name in ["MRPC", "QQP"]:
                tokens1 = self.tokenizer.encode(sentences[0] + " <|endoftext|>")
                tokens2 = self.tokenizer.encode(sentences[1] + " <|endoftext|> ")
                tokens2 = self.tokenizer.encode(" <|spe|>,   ") + tokens2
                if len(tokens2) > self.max_seq_len // 2:
                    tokens2 = tokens2[:self.max_seq_len // 2 - 1] + [tokens2[-1]]
                max_len = self.max_seq_len - len(tokens2)
            else:
                NotImplementedError

            # tokens = self.tokenizer.add_special_tokens_sentences_pair(tokens1, tokens2, start_type='SPE')
            if len(tokens1) > max_len:
                tokens1 = tokens1[:max_len - 1] + [tokens1[-1]]

            tokens = tokens1 + tokens2

            index = len(tokens1)
            assert index < self.max_seq_len


        sentences = np.array(tokens, dtype=np.int64)


        if self.task_name in ["SST-2", "CoLA", "MRPC", "RTE", "QNLI", "MNLI", "QQP", "MNLI_Match"]:
            label = int(dataset_dict['label'])
        else:
            raise NotImplementedError()


        ret = {
            'input_sample': [{
                'data': [sentences],
                'modality': 'text',
                'data_type': 'input',
                'invalid_mask': None,
                'sample_info' : {
                    'spe_index': index 
                }
            }],
            'target_sample': [],
            'target_idx'   : [label],
            'target_set'   : copy.deepcopy(self.target_set),
            'task_info'    : copy.deepcopy(self.task_info)
        }

        dict_as_tensor(ret)
        return ret
