import json
from collections import defaultdict 
import jsonlines

subsets = ['train', 'val', 'test']
savepath = "flickr30k/annotations"

set2jsonline = {
    'train': 'flickr30k/all_data_final_train_2014.jsonline', 
    'val': 'flickr30k/all_data_final_val_set0_2014.jsonline', 
    'test': 'flickr30k/all_data_final_test_set0_2014.jsonline',
}

import os 
if not os.path.exists(savepath):
    os.makedirs(savepath)


savename = {
    'train': "flickr30k/captions_train.json",
    'val': "flickr30k/captions_val.json",
    'test': "flickr30k/captions_test.json",
}

# imagefields = defaultdict(list)
# annotationsfields = defaultdict(list)

for subset in subsets: 
    imagefield = []
    annotaionfiled = []
    sen_id = 0
    with jsonlines.open(set2jsonline[subset]) as reader:
        for annotation in reader:
            sentences = annotation["sentences"]
            image_id = annotation["img_path"]
            imagefield.append({
                "filename": annotation["img_path"],
                "id": annotation['id'],
            })
            for sentence in sentences:
                annotaionfiled.append({
                    "image_id": annotation['id'],
                    "id": sen_id,
                    "caption": sentence,
                })
                sen_id += 1
                
    data = {
        "images": imagefield,
        "annotations": annotaionfiled,
    }
    json.dump( data, open(savename[subset], "w"))
    