import os
import sys
import tempfile
import json
from json import encoder
from uniperceiver.config import configurable
from .build import EVALUATION_REGISTRY
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from uniperceiver.utils import comm

@EVALUATION_REGISTRY.register()
class COCOEvaler(object):
    def __init__(self, cfg, annfile, output_dir):
        super(COCOEvaler, self).__init__()

        self.coco = COCO(annfile) if annfile != '' else None
        if not os.path.exists("./data/temp") and comm.is_main_process():
            os.makedirs("./data/temp")

        if output_dir is not None:
            self.output_dir = os.path.join(output_dir, 'results')
            if not os.path.exists(self.output_dir) and comm.is_main_process():
                os.makedirs(self.output_dir)
        else:
            self.output_dir = None

    def eval(self, results_input, epoch):
        image_ids = []
        results = []
        for result in results_input:
            if result['image_id'] not in image_ids:
                results.append(result)
                image_ids.append(result['image_id'])

        if self.output_dir is not None:
            json.dump(results, open(os.path.join(self.output_dir, str(epoch) + '.json'), "w"))

        in_file = tempfile.NamedTemporaryFile(mode='w',
                                              delete=False,
                                              dir="./data/temp")
        json.dump(results, in_file)
        in_file.close()

        if self.coco is None:
            return {}

        cocoRes = self.coco.loadRes(in_file.name)
        cocoEval = COCOEvalCap(self.coco, cocoRes)
        cocoEval.evaluate()
        os.remove(in_file.name)
        return cocoEval.eval