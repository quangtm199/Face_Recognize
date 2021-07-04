import cv2
from matplotlib import pyplot as plt
import glob
import os
import random

from tqdm import tqdm

images = []
labels = []
names = []
count = 0

for path, dirs, files in os.walk('/home/quang/Downloads/arcface-pytorch/data/Datasets/lfw'):
    for d in tqdm(dirs):
        class_image = []
        for ext in ('jpg', 'jpeg', 'png'):
            for f in glob.glob(os.path.join(path, d, '*.' + ext)):
                try:
                    print(f)
                    img = read_image(f)
                except:
                    continue
                if img is None:
                    continue
                class_image.append(img)

        if len(class_image) >= 1:
            names.append(d)
            for img in class_image:
                images.append(img)
                labels.append(count)
            count += 1
import pickle

with open('X_train_triplet.pkl', 'wb') as f:
    pickle.dump(images, f)
with open('y_train_triplet.pkl', 'wb') as f:
    pickle.dump(labels, f)
with open('name_map.pkl', 'wb') as f:
    pickle.dump(names, f)

# import numpy as np
# import cv2
# import pickle
#
# with open('/home/quang/Downloads/arcface-pytorch/models/X_train_triplet.pkl', 'rb') as f:
#     X = pickle.load(f)
#     X = np.array(X)
# #     X = np.expand_dims(X, axis=3)
# with open('/home/quang/Downloads/arcface-pytorch/models/y_train_triplet.pkl', 'rb') as f:
#     y = pickle.load(f)
#     y = np.array(y)
#
# with open('/home/quang/Downloads/arcface-pytorch/models/name_map.pkl', 'rb') as f:
#     name_map = pickle.load(f)
# print(np.shape(X), np.shape(y))