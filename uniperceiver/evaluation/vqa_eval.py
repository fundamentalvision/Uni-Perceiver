# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os
import sys
import pickle
import json
from json import encoder
from .build import EVALUATION_REGISTRY

@EVALUATION_REGISTRY.register()
class VQAEvaler(object):
    def __init__(self, cfg, annfile, output_dir):
        super(VQAEvaler, self).__init__()
        label2ans_path = os.path.join(cfg.DATALOADER.ANNO_FOLDER, "trainval_label2ans.pkl")
        ori_annotation = json.load(open(os.path.join(cfg.DATALOADER.ANNO_FOLDER, "v2_mscoco_val2014_annotations.json")))
        self.id2type = {t['question_id']: t['question_type'] for t in ori_annotation['annotations']}
        self.id2type_answer = {t['question_id']: t['answer_type'] for t in ori_annotation['annotations']}
        self.label2ans = pickle.load(open(label2ans_path, "rb"))

        self.id2label = {}
        if len(annfile) > 0:
            answers_val = pickle.load(open(annfile, "rb"))
            for datum in answers_val:
                quesid = datum['question_id']
                self.id2label[quesid] = {}
                for i, label in enumerate(datum['labels']):
                    label_str = self.label2ans[label]
                    self.id2label[quesid][label_str] = datum['scores'][i]

        if output_dir is not None:
            self.output_dir = os.path.join(output_dir, 'results')
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
        else:
            self.output_dir = None

    def eval(self, results, epoch):
        for res in results:
            res['answer'] = self.label2ans[res['answer']]
        if self.output_dir is not None:
            json.dump(results, open(os.path.join(self.output_dir, str(epoch) + '.json'), "w", encoding="utf-8"))

        accuracy = 0.
        acc_by_type = dict()
        acc_by_type_answer = dict()
        for result in results:
            quesid = result['question_id']
            ans = result['answer']
            if quesid not in self.id2type:
                print("Test Stage has no target")
                return { "accuracy": 0.0 }
            type = self.id2type[quesid]
            ans_type = self.id2type_answer[quesid]
            if type not in acc_by_type:
                acc_by_type[type] = [0,0]
            if ans_type not in acc_by_type_answer:
                acc_by_type_answer[ans_type] = [0,0]
            if quesid not in self.id2label:
                return { "accuracy": 0.0 }

            datum = self.id2label[quesid]
            acc_by_type[type][1] += 1
            acc_by_type_answer[ans_type][1] += 1
            if ans in datum:
                accuracy += datum[ans]
                acc_by_type[type][0] += datum[ans]
                acc_by_type_answer[ans_type][0] += datum[ans]

        accuracy = accuracy / len(results)
        print('vqa acc: {}'.format(accuracy*100))
        for k in acc_by_type.keys():
            acc_by_type[k] = acc_by_type[k][0] / acc_by_type[k][1] * 100
        for k in acc_by_type_answer.keys():
            acc_by_type_answer[k] = acc_by_type_answer[k][0] / acc_by_type_answer[k][1] * 100
        print('vqa acc by question type: ', acc_by_type)
        print("vqa acc by answer type: ", acc_by_type_answer)
        return { "accuracy": accuracy }
