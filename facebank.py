import cv2
import torch.cuda
from matplotlib import pyplot as plt
import glob
import os
import numpy as np
import random
from tqdm import tqdm
from models.resnet import *
images = []
labels = []
names = []
count = 0
def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output
if __name__ == "__main__":
    for path, dirs, files in os.walk('/home/quang/Downloads/arcface-pytorch/facebank_Image'):
        for d in (dirs):
            class_image = []
            for ext in ('jpg', 'jpeg', 'png'):
                for f in glob.glob(os.path.join(path, d, '*.' + ext)):
                    try:

                        img = cv2.imread(f)
                    except:
                        continue
                    if img is None:
                        continue
                    class_image.append(f)

            if len(class_image) > 1:
                names.append(d)
                for img in class_image:
                    images.append(img)
                    labels.append(count)
                count += 1
    import pickle

    with open('/home/quang/Downloads/arcface-pytorch/bank/X_train_triplet.pkl', 'wb') as f:
        pickle.dump(images, f)
    with open('/home/quang/Downloads/arcface-pytorch/bank/y_train_triplet.pkl', 'wb') as f:
        pickle.dump(labels, f)
    with open('/home/quang/Downloads/arcface-pytorch/bank/name_map.pkl', 'wb') as f:
        pickle.dump(names, f)
    images= np.array(images)
    with open('/home/quang/Downloads/arcface-pytorch/bank/X_train_triplet.pkl', 'rb') as f:
        X = pickle.load(f)
        X = np.array(X)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet_face18(use_se=False)
        # model = DataParallel(model)
    model.load_state_dict(torch.load("./pretrainModel_ARC/train4.pth"))

    model.eval()

    embs128 = []
    images = None
    model.to(torch.device("cpu"))
    features = None
    for i, x in tqdm(enumerate(X)):
        with torch.no_grad():
            image = cv2.imread(x, 0)
            image = np.dstack((image, np.fliplr(image)))
            image = image.transpose((2, 0, 1))
            image = image[:, np.newaxis, :, :]
            image = image.astype(np.float32, copy=False)
            image -= 127.5
            image /= 127.5
            data = torch.from_numpy(image)
            data = data.to(torch.device("cpu"))

            try:
                output = model(data)
                output = output.data.cpu().numpy()
                fe_1 = output[::2]
                fe_2 = output[1::2]
                out = fe_1 + fe_2
                out = torch.from_numpy(out)
                out=l2_norm(out)
                out = out.cpu().numpy()
                out=out[0][:]
                print(out.shape)
                embs128.append(out)
            except:
                print(x)
                print("except")
                continue
            images = None
    with open('/home/quang/Downloads/arcface-pytorch/bank/facebank.pkl', 'wb') as f:
        pickle.dump(embs128, f)