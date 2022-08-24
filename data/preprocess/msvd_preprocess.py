import json
import os

subsets = ["train", "val", "test"]
save_path = 'msvd_dataset/new_annotations'

subset = subsets[1]

videoindex = open("msvd_dataset/txt_labels/youtube_mapping.txt", 'r').readlines()
sentence_count = 1
for subset in subsets:
    name2idx = dict()
    idx2name = dict()
    
    for v in videoindex:   
        name2idx[v.split()[0]] = v.split()[1] 
        idx2name[v.split()[1]] = v.split()[0] 
        
    images_field = []
    annotations_field = []
    visited_imames = set()
    txtfile = "msvd_dataset/txt_labels/sents_{}_lc_nopunc.txt".format(subset)
    capinfos = open(txtfile, 'r').readlines()
    for caption in capinfos:
        vidindex = caption.split('\t')[0]
        if vidindex not in visited_imames:
            visited_imames.add(vidindex)
            images_field.append(
                {
                    "id":  int(vidindex.replace('vid', '')),
                    "file_name": idx2name[vidindex]
                }
            )
        annotations_field.append(
            {
                "image_id":int(caption.split()[0].replace('vid', '')),
                "id": sentence_count,
                "caption": caption.split('\t')[1].strip()
                
            }
        )
        sentence_count += 1
    
    data = {
        "images": images_field,
        "annotations": annotations_field
    }
    json.dump(data, open(os.path.join(save_path, "caption_msvd_{}_cocostyle.json".format(subset)), "w"))