# USAGE
import argparse
from model.lib import *
import os
from model.MobileNet import MobileNetV2
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"
from models.resnet import *
from test import *
from PIL import Image
from MTCNN import *
import numpy as np
import cv2
import pickle
from facebank import l2_norm
opt = Config()
def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

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
FaceModel = resnet_face18(use_se=opt.use_se)
# model = DataParallel(model)
FaceModel.load_state_dict(torch.load("./save_model/model_ARC.pth"))
FaceModel.to(device)
FaceModel.eval()

# Cai dat cac tham so dau vao
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", type=str, default='face_detector',
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
class_index = ["fake", "real"]

def predict_max(output, class_index):  # [0.9, 0.1]
    max_id = np.argmax(output.detach().numpy())
    predicted_label = class_index[max_id]
    return predicted_label


# Load model nhan dien khuon mat
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load model nhan dien fake/real
print("[INFO] loading liveness detector...")
model = MobileNetV2()
model.load_state_dict(torch.load("/home/quang/Documents/face_anti_spoofing/weight_1.pth"))
model.to(torch.device("cpu"))
model.eval()
resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
tranforms = transforms.Compose([
    transforms.Resize(resize),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
#  Doc video tu webcam
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()

path_video="/home/quang/Downloads/arcface-pytorch/video/outpy1.avi"


vs = cv2.VideoCapture(path_video)

time.sleep(2.0)

while True:
    # Doc anh tu webcam
    cap, frame = vs.read()
    # # Chuyen thanh blob
    # frame = cv2.imread("/home/quang/Downloads/arcface-pytorch/9.jpg")
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # Phat hien khuon mat
    net.setInput(blob)
    detections = net.forward()
    # print(detections.shape)
    # Loop qua cac khuon mat
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        # Neu conf lon hon threshold
        if confidence > args["confidence"]:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # Lay vung khuon mat
            face = frame[startY:endY, startX:endX]
            small_face=cv2.resize(face,(128,128))
            data = processImageInput(small_face)
            # print(data.shape)
            t0 = time.time()
            output = FaceModel(data)

            #process output
            output = output.data.cpu().numpy()
            fe_1 = output[::2]
            fe_2 = output[1::2]
            out = fe_1 + fe_2
            out = torch.from_numpy(out)
            out = l2_norm(out)
            out = out.cpu().numpy()
            out = out[0][:]



            embs128 = np.array(embs128)
            minimum = -9999
            person = -1

            #calculate dist
            diff = np.subtract(embs128, out)
            dist = np.sum(np.square(diff), 1)

            idx = np.argmin(dist)
            sim = cosin_metric(out, embs128[idx])
            # print("idx", idx)
            # print(sim)
            t1 = time.time()
            # print('finish',t1-t0)
            threshold = 0.5
            if sim > threshold:
                name = name_map[y[idx]]
            else:
                name = "unknow"
            sim = np.array2string(sim, formatter={'float_kind': lambda x: "%.2f" % x})
            small_frame = cv2.resize(face, (224, 224))
            pil_im = Image.fromarray(small_frame)
            img = tranforms(pil_im)
            img = np.array(img)
            img = np.expand_dims(img, 0)
            img = torch.from_numpy(img)
            img = img.to(torch.device("cpu"))
            output_1 = model(img)

            soft_output = torch.softmax(output_1, dim=-1)
            preds = soft_output.to('cpu').detach().numpy()
            _, predicted = torch.max(soft_output.data, 1)
            predicted_2 = predicted.to('cpu').detach().numpy()
            # print('preds', preds[0][predicted_2])
            # print(predicted_2)

            # Dua vao model de nhan dien fake/real
            # preds = model.predict(face)[0]

            # j = np.argmax(preds)
            label = class_index[int(predicted_2)]
            j = int(predicted_2)
            # Ve hinh chu nhat quanh mat
            label = "{}: {:.4f}".format(label, preds[0][j])
            if (j == 0):
                # Neu la fake thi ve mau
                cv2.putText(frame, str(sim), (startX, startY + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, name, (startX, startY + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
            else:
                # Neu real thi ve mau xanh
                cv2.putText(frame, sim, (startX, startY + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, name, (startX, startY + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # Bam 'q' de thoat
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()
