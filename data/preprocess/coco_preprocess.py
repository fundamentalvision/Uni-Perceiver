import json
from collections import defaultdict 

original_json = json.load(open("mscoco_dataset/new_annotations/dataset_coco.json"))

subsets = ['train', 'val', 'test']
savepath = "mscoco_dataset/new_annotations"

import os 
if not os.path.exists(savepath):
    os.makedirs(savepath)

savename = {
    'train': "captions_train113k.json",
    'val': "captions_val5k.json",
    'test': "captions_test5k.json",
}

imagefields = defaultdict(list)
annotationsfields = defaultdict(list)

for imagecaps in original_json['images']:
    filepath = imagecaps['filepath']
    filename = imagecaps['filename']
    image_id = int(filename.split(".")[0].split('_')[-1])
    split = imagecaps['split']
    if split == 'restval':
        split = 'train'
    imagefields[split].append({
        "file_name": filename,
        "file_path": filepath,
        "id": image_id
    })
    for sen in imagecaps['sentences']:
        annotationsfields[split].append({
            "image_id": image_id,
            "id": sen["sentid"],
            "caption": sen["raw"],
        })
    
for subset in subsets:
    data = {
        "images":  imagefields[subset],
        "annotations": annotationsfields[subset]
    }
    json.dump(data, open(os.path.join(savepath, savename[subset]), "w"))
pass



