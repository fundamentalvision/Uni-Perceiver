import os 
import json


data = dict()
data['database'] = dict()

categories = list()

with open("K400_val.csv", 'w') as f:
    for root, dirs, files in os.walk("./validation"):
        label = root.strip().split('/')[-1]
        if files and label not in categories:
            categories.append(label)
        for fi in files:
            f.write("{},{}\n".format(os.path.join(root, fi), label))
            data['database'][fi] = {'subset': 'validation', 'annotations': {'label': label}}

with open("K400_train.csv", 'w') as f:
    for root, dirs, files in os.walk("./training"):
        label = root.strip().split('/')[-1]
        if files and label not in categories:
            categories.append(label)
        for fi in files:
            f.write("{},{}\n".format(os.path.join(root, fi), label))
            data['database'][fi] = {'subset': 'training', 'annotations': {'label': label}}

with open("categories.txt", 'w') as f:
    for i, label in enumerate(categories):
        f.write("{},{}\n".format(label, i))

with open("annotation.json", 'w') as f:
    json.dump(data, f)


