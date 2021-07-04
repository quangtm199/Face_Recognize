
from models.resnet import *
import argparse
from test import *
import imutils
from PIL import Image
from MTCNN import *
import numpy as np
import cv2
import pickle
opt = Config()
def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    embs128 = []
    images = None
    # read file PKL
    with open('/home/quang/Downloads/arcface-pytorch/bank/y_train.pkl', 'rb') as f:
        y = pickle.load(f)
        y = np.array(y)
    with open('/home/quang/Downloads/arcface-pytorch/bank/name_map.pkl', 'rb') as f:
        name_map = pickle.load(f)
    with open('/home/quang/Downloads/arcface-pytorch/bank/facebank.pkl', 'rb') as f:
        embs128 = pickle.load(f)

    model = resnet_face18(use_se=opt.use_se)
    # model = DataParallel(model)
    model.load_state_dict(torch.load("/home/quang/Downloads/arcface-pytorch/save_model/model_ARC.pth"))
    model.to(device)
    model.eval()
    pnet = PNet1(test=True)
    rnet = RNet1(test=True)
    onet = ONet1(test=True)
    ctx = mx.cpu()
    pnet.load_parameters('./save_model/pnet1_150000', ctx = ctx )
    pnet.hybridize()

    rnet.load_parameters('./save_model/rnet1_300000', ctx=ctx)
    rnet.hybridize()

    onet.load_parameters('./save_model/onet_80000',ctx=ctx)
    onet.hybridize()
    mtcnn = MTCNN(detectors=[pnet, rnet, onet], min_face_size = 24, scalor = 0.709,threshold=[0.6, 0.7, 0.7], ctx = ctx )
   #  mtcnn=MTCNN()
   #  facebank_path="/home/quang/Downloads/arcface-pytorch/data/"+args.file_name
   #  cap = cv2.VideoCapture('/home/quang/Downloads/arcface-pytorch/IMG_4284.mp4')
    cap = cv2.VideoCapture(0)
    # img_width, img_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    phase = 'val'

    image = cv2.imread("/home/quang/Downloads/arcface-pytorch/1.jpeg")
    image = np.asarray(image)
    bboxes = mtcnn.detect(image)
    print(bboxes)
    e = time.time()
    if bboxes is not None:
        plt.figure()
        tmp = image.copy()
        count = 0
        for i in bboxes:
            x0 = int(i[0])
            y0 = int(i[1])
            x1 = x0 + int(i[2])
            y1 = y0 + int(i[3])
            img1 = image.copy()
            img1 = cv2.resize(img1[y0:y1, x0:x1], (128, 128))
                            # cv2.imwrite("/home/quang/Downloads/arcface-pytorch/data/12.jpg", img1)
            data=processImageInput(img1)
            t0 = time.time()

            output = model(data)
            output = output.data.cpu().numpy()
            minimum = 99999
            person = -1
            diff = np.subtract(embs128, output[0])
            dist = np.sum(np.square(diff), 1)
            idx = np.argmin(dist)
            t1 = time.time()
            print('finish', t1 - t0)
            threshold = 100
            if dist[idx] < threshold:
                name = name_map[y[idx]]
            else:
                name = name_map[y[idx]]

            for k, e in enumerate(embs128):
                dist = np.linalg.norm(output[0] - e)
                sim = cosin_metric(output[0], e)
                if dist < minimum:
                    minimum = dist
                    person = k
                    print('sim',sim)
            print(dist)
            print(name_map[y[person]])
            print(name)
            try:
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
                cv2.putText(image, name, (x0, y0), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (255, 255, 255), thickness=1, lineType=2)
            except:
                print("excpet")
    cv2.imshow("img1", image)
    cv2.imwrite("1_infer.jpg",image)
        # cv2.imwrite("/home/quang/Downloads/arcface-pytorch/img2.jpg",image)
    cv2.waitKey(0)

