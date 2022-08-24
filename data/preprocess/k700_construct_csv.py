import os 
import json
import random


data = dict()
data['database'] = dict()

categories = list()


with open("K700_val.csv", 'w') as f:
    for root, dirs, files in os.walk("./validation"):
        label = root.strip().split('/')[-1]
        if files:
            categories.append(label)
        for fi in files:
#             f.write("{},{}\n".format(os.path.join(root, fi), label))
            data['database'][fi] = {'subset': 'validation', 'annotations': {'label': label}}

print("{} validation instances".format(len(data['database'])))

with open("K700_train.csv", 'w') as f:
    for root, dirs, files in os.walk("./training"):
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


