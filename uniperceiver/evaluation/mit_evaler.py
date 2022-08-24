import os
import tempfile
import json
from uniperceiver.config import configurable
from .build import EVALUATION_REGISTRY
from uniperceiver.utils import comm
import numpy as np

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

@EVALUATION_REGISTRY.register()
class MiTEvaler(object):
    def __init__(self, cfg, annfile, output_dir):
        super(MiTEvaler, self).__init__()
        self.video_dict = dict()

        self.cls2idx = dict()
        with open(os.path.join(os.path.dirname(annfile), "category_mapping.txt"), 'r') as f:
            for line in f.readlines():
                class_name, idx = line.strip().split('\t')
                class_name = class_name.replace(" ", "_")
                self.cls2idx[class_name] = int(idx)

        with open(annfile) as f:
            data_file = json.load(f)
            for name, info in data_file['database'].items():
                #                 if info['subset'] == "validation" or True: # debug
                if info['subset'] == "validation":
                    self.video_dict[name] = self.cls2idx[info['annotations']['label']]

        if not os.path.exists(comm.TEMP_DIR):
            os.mkdir(comm.TEMP_DIR)

        if output_dir is not None:
            self.output_dir = os.path.join(output_dir, 'results')
            if not os.path.exists(self.output_dir) and comm.is_main_process():
                os.mkdir(self.output_dir)
        else:
            self.output_dir = None

    def eval(self, results, epoch):
        preds = []
        labels = []
        for result in results:
            labels.append(self.video_dict[result["video_name"]])
            preds.append(result["label"].item())
        preds = np.array(preds)
        labels = np.array(labels)
        acc = simple_accuracy(preds, labels)
        # if self.output_dir is not None:
        #     json.dump(results, open(os.path.join(self.output_dir, str(epoch) + '.json'), "w"))


        return {"accuracy": acc}
