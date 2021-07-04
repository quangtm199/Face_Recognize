
from models.resnet import *
from test import *
import imutils
from PIL import Image
from MTCNN import *
import numpy as np
import cv2
import pickle
from facebank import l2_norm
opt = Config()
def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    embs128 = []
    images = None
    # read file PKL
    with open('./bank/X_train.pkl', 'rb') as f:
        X = pickle.load(f)
        X = np.array(X)
    #     X = np.expand_dims(X, axis=3)
    with open('./bank/y_train.pkl', 'rb') as f:
        y = pickle.load(f)
        y = np.array(y)
    with open('./bank/name_map.pkl', 'rb') as f:
        name_map = pickle.load(f)
    print(X.shape)
    with open('./bank/facebank.pkl', 'rb') as f:
        embs128 = pickle.load(f)
    print(len(embs128))
    model = resnet_face18(use_se=opt.use_se)
    # model = DataParallel(model)
    model.load_state_dict(torch.load("./save_model/model_ARC.pth"))
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
    cap = cv2.VideoCapture('./video/real.mp4')
   #  cap = cv2.VideoCapture(0)
    # img_width, img_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    phase = 'val'
    while (True):
        isSuccess, frame = cap.read()
        frame = imutils.resize(frame, width=400)
        # image=cv2.resize(frame,(300,300),interpolation = cv2.INTER_AREA)
        # print(image.shape)
        image = Image.fromarray(frame)
        image = np.asarray(image)
        origin = image.copy()
        if isSuccess:
            bboxes = mtcnn.detect(image)
            print(bboxes)
            e = time.time()
            if True:
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
                        img1 = cv2.resize(frame[y0:y1, x0:x1], (128, 128))

                        cv2.imwrite("/home/quang/Downloads/arcface-pytorch/1.jpg", img1)
                        data=processImageInput(img1)
                        print(data.shape)
                        t0 = time.time()
                        output = model(data)
                        output = output.data.cpu().numpy()
                        fe_1 = output[::2]
                        fe_2 = output[1::2]
                        out = fe_1 + fe_2
                        out = torch.from_numpy(out)
                        out = l2_norm(out)
                        out = out.cpu().numpy()
                        out = out[0][:]
                        embs128= np.array(embs128)
                        minimum = -9999
                        person = -1
                        diff = np.subtract(embs128, out)
                        dist = np.sum(np.square(diff), 1)

                        idx = np.argmin(dist)
                        sim = cosin_metric(out, embs128[idx])
                        print("idx",idx)
                        print(sim)
                        t1 = time.time()
                        # print('finish',t1-t0)
                        threshold=0.3
                        if sim > threshold:
                            name=name_map[y[idx]]
                        else:
                            name = "unknow"
                        # for k, e in enumerate(embs128):
                        #     dist = np.linalg.norm(out - e)
                        #     sim = cosin_metric(out, e)
                        #     if sim > minimum:
                        #         minimum = sim
                        #         person = k
                        #         print(sim)
                        # name=(name_map[y[person]])
                        try:
                            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
                            cv2.putText(frame, name, (x0, y0), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                        except:
                            print("excpet")
        cv2.imshow("img1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()