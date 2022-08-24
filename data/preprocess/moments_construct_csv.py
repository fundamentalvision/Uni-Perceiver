import os
import scandir 
import json
import random


data = dict()
data['database'] = dict()

categories = list()

print("{} validation instances".format(len(data['database'])))
c = 0
with open("moments_train.csv", 'w') as f:
    for root, dirs, files in scandir.walk("./training"):
        print(c)
        c += 1
        label = root.strip().split('/')[-1]
        if files:
            categories.append(label)
        for fi in files:
            f.write("{},{}\n".format(os.path.join(root, fi), label))
            data['database'][fi] = {'subset': 'training', 'annotations': {'label': label}}

with open("categories.txt", 'w') as f:
    categories = sorted(categories)
    for i, label in enumerate(categories):
        f.write("{},{}\n".format(label, i))

with open("annotation.json", 'w') as f:
    json.dump(data, f)


