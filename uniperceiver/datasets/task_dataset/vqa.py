import os
import copy
import pickle
import random
import json
import glob
from uniperceiver.utils import comm
from numpy.random import choice
import pyarrow as pa
from PIL import Image
from torchvision import transforms
import numpy as np
from uniperceiver.config import configurable
from uniperceiver.functional import read_np, dict_as_tensor, boxes_to_locfeats
from uniperceiver.tokenization import ClipTokenizer
from ..build import DATASETS_REGISTRY
import torch
from uniperceiver.datasets.custom_transforms import clip_transforms

__all__ = ["VQADataset"]

memorycache = False
try:
    if "SLURM_JOB_ID" in os.environ:
        import mc
        import io
        memorycache = True
#         print("VQA using memory cache")
    else:
        # print("missing memory cache")
        pass
except:
    # print("missing memory cache")
    pass

@DATASETS_REGISTRY.register()
class VQADataset:
    @configurable
    def __init__(
        self,
        cfg,
        dataset_name,
        task_type,
        stage: str,
        anno_folder: str,
        ans2label_path: str,
        label2ans_path: str,
        feats_folder: str,
        max_feat_num: int,
        max_seq_len: int,
        use_global_v: bool,
        tokenizer,
        tokenizer_name,
        use_ceph,
        transform,
        as_gen,
        inf_input,
        single_class,
        small_val,
        block_vq,
        data_percentage,
        two_eot,
    ):
        self.stage = stage
        self.anno_folder = anno_folder
        self.ans2label = pickle.load(open(ans2label_path, "rb"))
        self.label2ans = pickle.load(open(label2ans_path, "rb"))
        self.feats_folder = feats_folder
        self.max_feat_num = max_feat_num
        self.max_seq_len = max_seq_len
        self.use_global_v = use_global_v
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.num_labels = len(self.ans2label)
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.task_type = task_type
        
        self.id2path = self.load_img_info(self.anno_folder)

        self.initialized = False
        self.transform = transform
        self.as_gen = as_gen
        self.inf_input = inf_input
        self.single_class = single_class
        self.small_val = small_val
        self.block_vq = block_vq
        self.data_percentage = data_percentage
        self.two_eot = two_eot
        # if as_retrieval:
        if self.tokenizer_name == "clip":
            self.mask_tokens = [tokenizer.encoder["<|spe|>"]]
        else:
            raise NotImplementedError
        # remove the first null answer, we are not using complementay dataset
        self.answer_tokens = self.tokenize_answer()
        self.answer_type_tokens = [np.zeros(len(x), dtype=np.int64) for x in self.answer_tokens]
        
        self.use_ceph = use_ceph
        if self.use_ceph:
            self.feats_folder = "s3://coco"
            print('debug info for vqa  {}'.format( self.feats_folder))
            from uniperceiver.datasets import TCSLoader
            if 'SLURM_PROCID' in os.environ: 
                tcs_conf_path = cfg.DATALOADER.get("TCS_CONF_PATH", "slurm_tools/petreloss.config")
            else: 
                # dev machine 
                tcs_conf_path = "slurm_tools/petreloss_local.config"
            self.tcs_loader = TCSLoader(tcs_conf_path)

        self.load_data(self.cfg)
        
        self.task_info = {
                'task_type'      : self.cfg.DATASETS.TASK_TYPE,
                'dataset_name'   : self.cfg.DATASETS.DATASET_NAME,
                'batch_size'     : self.cfg.DATALOADER.TRAIN_BATCH_SIZE if self.stage == 'train' else self.cfg.DATALOADER.TEST_BATCH_SIZE,
                'sampling_weight': self.cfg.DATALOADER.SAMPLING_WEIGHT,
                'single_class'   : self.cfg.DATALOADER.SINGLE_CLASS  
            }
            
    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/cache/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/cache/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True


    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ans2label_path = os.path.join(cfg.DATALOADER.ANNO_FOLDER, "trainval_ans2label.pkl")
        label2ans_path = os.path.join(cfg.DATALOADER.ANNO_FOLDER, "trainval_label2ans.pkl")
        
        feats_folder = cfg.DATALOADER.FEATS_FOLDER
        # if stage == "test":
        #     feats_folder = feats_folder + "/test2015"

        if getattr(cfg.DATALOADER, 'TRANSFORM', None) == 'clip_transforms':
            transform = clip_transforms(stage, flip_prob=0.0)
        else:
            transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), 
                                (0.229, 0.224, 0.225))]
            )
        
        ret = {
            'cfg': cfg,
            'dataset_name': cfg.DATASETS.DATASET_NAME,
            'task_type': cfg.DATASETS.TASK_TYPE,
            "stage": stage,
            "anno_folder": cfg.DATALOADER.ANNO_FOLDER,
            "ans2label_path": ans2label_path,
            "label2ans_path": label2ans_path,
            "feats_folder": feats_folder,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
            "use_global_v": cfg.DATALOADER.USE_GLOBAL_V,
            "use_ceph": getattr(cfg.DATALOADER, 'USE_CEPH', False),
            "transform": transform,
            "as_gen": cfg.DATALOADER.DO_AS_GEN,
            "inf_input": cfg.DATALOADER.VQA_INPUT,
            "single_class": cfg.DATALOADER.SINGLE_CLASS,
            "small_val": cfg.DATALOADER.SMALL_VAL,
            "block_vq": cfg.DATALOADER.BLOCK_VQ,
            "data_percentage": cfg.DATALOADER.DATA_PERCENTAGE,
            "two_eot": cfg.DATALOADER.TWO_EOT,
        }

        ret['tokenizer'] = ClipTokenizer()
        ret['tokenizer_name']  = "clip"

        return ret
    
    def load_img_info(self, anno_folder):
        id2path = {}
                
        coco_map = json.load(open(os.path.join(anno_folder, "coco_map.json")))
        for k, v in coco_map.items():
            id2path[int(k)] = v
                
        return id2path

    def load_data(self, cfg):
        
        cache_path = os.path.join(
            self.anno_folder, "cache", 
            "VQA_sep_%s_%s_%d%s.pkl" % (self.tokenizer_name, self.stage, self.max_seq_len, "_full_val" if self.stage == "val" and not self.small_val else "")
        )
        if not os.path.exists(os.path.dirname(cache_path)):
            os.makedirs(os.path.dirname(cache_path))
        if not os.path.exists(cache_path):
            datalist = self.load_raw_data(cfg)    
            self.tokenize(datalist)
            pickle.dump(datalist, open(cache_path, "wb"))
        datalist = pickle.load(open(cache_path, "rb"))
        if self.data_percentage < 1.0 and self.stage == "train":
            labels2l = dict()
            for data in datalist:
                if not data['answer']['labels']:
                    continue
                ans = data['answer']['labels'][0]
                if ans not in labels2l:
                    labels2l[ans] = list()
                labels2l[ans].append(data)
            datalist = []
            for v in labels2l.values():
                datalist.extend(random.sample(v, k=int(self.data_percentage * len(v)+1)))
        
        self.database = pa.array(datalist)
        self.datalist = datalist

        if comm.is_main_process():
            import sys
            print(f"!!! Dataset {self.dataset_name} with task {self.task_type}:")
            print('!!! length of _temp_list: ', len(datalist))
            print('!!! size of _temp_list: ', sys.getsizeof(datalist))
            print('!!! size of pa database: ', sys.getsizeof(self.database))
        del datalist
        

    def tokenize(self, datalist):
        for entry in datalist:
            tokens = self.tokenizer.encode(entry["question"])
            tokens = tokens[: self.max_seq_len - 2]
            # tokens = self.tokenizer.add_special_tokens_single_sentence(tokens)
            entry["question"] = tokens
            
    def tokenize_answer(self):
        output = list()
        for answer in self.label2ans:
            answer_tokens = self.tokenizer.encode(answer + " <|endoftext|>")
            # answer_tokens = self.tokenizer.add_special_tokens_single_sentence(answer_tokens)
            output.append(answer_tokens)
        return output

    def load_raw_data(self, cfg):
        if self.stage == 'train': # trainval mode
            question_path_train = os.path.join(self.anno_folder, "v2_OpenEnded_mscoco_train2014_questions.json")
            questions_train = sorted(
                json.load(open(question_path_train))["questions"],
                key=lambda x: x["question_id"],
            )
            answer_path_train = os.path.join(self.anno_folder, "train_target.pkl")
            answers_train = pickle.load(open(answer_path_train, "rb"))
            answers_train = sorted(answers_train, key=lambda x: x["question_id"])

            question_path_val = os.path.join(self.anno_folder, "v2_OpenEnded_mscoco_val2014_questions.json")
            questions_val = sorted(
                json.load(open(question_path_val))["questions"],
                key=lambda x: x["question_id"],
            )   
            answer_path_val = os.path.join(self.anno_folder, "val_target.pkl")
            answers_val = pickle.load(open(answer_path_val, "rb"))
            answers_val = sorted(answers_val, key=lambda x: x["question_id"])

            # VG
            vg_question_path_train = os.path.join(self.anno_folder, "VG_questions2.json")
            vg_questions_train = sorted(
                json.load(open(vg_question_path_train))["questions"],
                key=lambda x: x["question_id"],
            )
            vg_answer_path_train = os.path.join(self.anno_folder, "vg_target.pkl")
            vg_answers_train = pickle.load(open(vg_answer_path_train, "rb"))
            vg_answers_train = sorted(vg_answers_train, key=lambda x: x["question_id"])

            questions = questions_train + questions_val[:-3000] + vg_questions_train
            answers = answers_train + answers_val[:-3000] + vg_answers_train
        elif self.stage == "val": # minval
            question_path_val = os.path.join(self.anno_folder, "v2_OpenEnded_mscoco_val2014_questions.json")
            questions_val = sorted(
                json.load(open(question_path_val))["questions"],
                key=lambda x: x["question_id"],
            )
            answer_path_val = os.path.join(self.anno_folder, "val_target.pkl")
            answers_val = pickle.load(open(answer_path_val, "rb"))
            answers_val = sorted(answers_val, key=lambda x: x["question_id"])
            if self.small_val:
                questions = questions_val[-3000:]
                answers = answers_val[-3000:]
            else:
                questions = questions_val
                answers = answers_val
        else:
            question_path_test = os.path.join(self.anno_folder, "v2_OpenEnded_mscoco_test2015_questions.json")
            # question_path_test = os.path.join(self.anno_folder, "v2_OpenEnded_mscoco_test-dev2015_questions.json")
            questions_test = sorted(
                json.load(open(question_path_test))["questions"],
                key=lambda x: x["question_id"],
            )
            questions = questions_test

        datalist = []
        if self.stage == "test":
            for question in questions:
                datalist.append({
                    "question_id": str(question["question_id"]),
                    "image_id": str(question["image_id"]),
                    "question": question["question"],
                })
        else:
            assert len(questions) == len(answers)
            for question, answer in zip(questions, answers):
                assert question["question_id"] == answer["question_id"]
                assert question["image_id"] == answer["image_id"]
                
                answer.pop("image_id")
                answer.pop("question_id")
                datalist.append({
                    "question_id": str(question["question_id"]),
                    "image_id": str(question["image_id"]),
                    "question": question["question"],
                    "answer": answer,
                })
        return datalist

    def __len__(self):
        return len(self.database)

    def __getitem__(self, index):
        
        for i_try in range(100):
            try:
                dataset_dict = self.database[index].as_py()
                image_id = dataset_dict['image_id']
                question_id = dataset_dict["question_id"]
                
                global memorycache
                
                image_path = os.path.join(self.feats_folder, self.id2path[int(image_id)])
                ### LOAD IMAGE ###
                
                if self.use_ceph:
                    img = self.tcs_loader(image_path).convert('RGB')
                
                elif not memorycache:
                    img = Image.open(image_path).convert("RGB")
                else:
                    # memcached
                    self._init_memcached()
                    value = mc.pyvector()
                    self.mclient.Get(image_path, value)
                    value_str = mc.ConvertBuffer(value)
                    buff = io.BytesIO(value_str)
                    img = Image.open(buff).convert("RGB")
            except Exception as e:
                print(
                    "Failed to load video from {} with error {} ; trial {}".format(
                        image_path, e, i_try
                    )
                )
                
                # let's try another one
                index = random.randint(0, len(self.datalist) - 1)
                dataset_dict = self.datalist[index]
                continue
        
            
            img = self.transform(img)
            
            prob = random.random()
            if prob > 0.5 and self.stage == 'train':
                # img = img[:, :, ::-1]
                img = torch.flip(img, [2])

            question = dataset_dict["question"]
            if self.as_gen:
                if self.two_eot:
                    question = question + self.tokenizer.encode("<|endoftext|>")
                question = question + self.tokenizer.encode("<|spe|> <|endoftext|>")
                index = len(question) - 2

            question = np.array(question, dtype=np.int64)

            #######################################################
            if prob > 0.5 and self.stage == 'train':
                for i in range(1, len(question)):
                    if self.tokenizer_name == "clip":
                        left = self.tokenizer.encoder["left"]
                        right = self.tokenizer.encoder["right"]
                        if question[i] == left:
                            question[i] = right
                        elif question[i] == right:
                            question[i] = left
                    else:
                        raise NotImplementedError
                        
            if 'image' in self.inf_input:
                ret = {
                'input_sample': [{
                    'data'        : img,
                    'invalid_mask': None,
                    'modality'    : 'image',
                    'data_type'   : 'input',
                    'sample_info' : {
                        'id':   image_id,
                        'path': image_path
                    }
                }]
            }

            self.target_set = self.cfg.DATASETS.TARGET_SET
            
            target = 0
            
            if "answer" in dataset_dict:
                answer = dataset_dict["answer"]
                labels = answer["labels"]
                scores = answer["scores"]

                #######################################################
                if prob > 0.5 and self.stage == 'train':
                    for i in range(len(labels)):
                        if labels[i] == self.ans2label['left']:
                            labels[i] = self.ans2label['right']
                        elif labels[i] == self.ans2label['right']:
                            labels[i] = self.ans2label['left']
                #######################################################


                if self.single_class:
                    if len(labels) < 1:
                        target = 0
                    else:
                        s = sum(scores)
                        # probabilty
                        p = [t / s for t in scores]
                        # sample
                        target = choice(labels, 1, p=p).item()
                else:
                    target = np.zeros(self.num_labels)
                    if len(labels) > 0:
                        for label, score in zip(labels, scores):
                            target[label] = score
                    target = np.array(target, dtype=np.float32)

            
            if self.as_gen:
                # caption like
                ret['input_sample'].append({
                    'data': [question],
                    'invalid_mask': None,
                    'modality': 'text',
                    'data_type': 'input',
                    'sample_info': {
                        'spe_index': index,
                        'question_id': question_id
                    }
                })
                ret.update({
                    'target_sample': [],
                    'target_idx'   : [target],
                    'target_set'   : copy.deepcopy(self.target_set),
                    'task_info'    : copy.deepcopy(self.task_info)
            })
                    
                
            dict_as_tensor(ret)
            return ret
