from models.resnet import *
from test import *
from network import *
from pylab import plt
import mxnet as mx
from util.utility import pad_bbox, square_bbox, py_nms
from MTCNN import *
opt = Config()
embs128 = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
model = resnet_face18(use_se=opt.use_se)
# model = DataParallel(model)
model.load_state_dict(torch.load("/home/quang/Downloads/arcface-pytorch/save_model/model_ARC.pth"))
# model.load_state_dict(torch.load("/home/quang/Downloads/arcface-pytorch/set/trainnew3.pth"))
model.to(device)
model.eval()
embs128 = []
images = None
import numpy as np
import cv2
import pickle

with open('/home/quang/Downloads/arcface-pytorch/bank/X_train.pkl', 'rb') as f:
    X = pickle.load(f)
    X = np.array(X)
#     X = np.expand_dims(X, axis=3)
with open('/home/quang/Downloads/arcface-pytorch/bank/y_train.pkl', 'rb') as f:
    y = pickle.load(f)
    y = np.array(y)

with open('/home/quang/Downloads/arcface-pytorch/bank/name_map.pkl', 'rb') as f:
    name_map = pickle.load(f)
print(X.shape)
with open('/home/quang/Downloads/arcface-pytorch/bank/facebank.pkl', 'rb') as f:
    embs128 = pickle.load(f)
print(X.shape)
print(y.shape)
print(len(name_map))
# with open('/home/quang/Downloads/arcface-pytorch/pretrainModel_ARC/facebank.pkl', 'rb') as f:
#     embs128 = pickle.load(f)

image = cv2.imread("/home/quang/Downloads/arcface-pytorch/download.jpeg", 0)
# data = data.to(torch.device("cuda"))
data=processImageInput(image)
output = model(data)
output = output.data.cpu().numpy()
minimum = 99999
person = -1
for k, e in enumerate(embs128):
    dist = np.linalg.norm(output[0] - e)
    if dist < minimum:
        minimum = dist
        person = k
        sim = cosin_metric(output[0], e)
        print('sim',sim)
        print(dist)
        print(person)
        print(name_map[y[person]])
