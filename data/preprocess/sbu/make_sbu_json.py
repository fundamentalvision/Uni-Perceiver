import json
import os
from glob import glob
imagefile = open('dataset/SBU_captioned_photo_dataset_urls.txt', 'r').readlines()
captionfile = open('dataset/SBU_captioned_photo_dataset_captions.txt', 'r').readlines()

valid_list = list(glob("images/*"))
valid_list = [ i.split('/')[-1] for i in valid_list]
                  

name2cap = {}
for imageurl, caption in zip(imagefile, captionfile):
    filename = imageurl.strip().split('/')[-1]
    name2cap[filename] = caption.strip()

data_list = {}
for valid_img in valid_list:
    data_list[valid_img]=name2cap[valid_img]

fp = open('annotations/subcaption.json', 'w')
json.dump(data_list, fp)

print(len(data_list))
