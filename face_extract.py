# CACH DUNG LENH
# python face_extract.py --input videos/real.mp4 --output dataset/real
# python face_extract.py --input videos/fake.mp4 --output dataset/fake
from models.resnet import *
import argparse
from test import *
import imutils
from PIL import Image
from MTCNN import *
import numpy as np
import cv2
import pickle
import numpy as np
import argparse
import cv2
import os
# Cac tham so dau vao
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="/home/quang/Documents/video.mp4",
	help="path to input video")
ap.add_argument("-o", "--output", type=str, default="/home/quang/Documents/face_anti_spoofing/face_detector",
	help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, default='face_detector',
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=1,
	help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())
import time
model = resnet_face18(use_se=False)
# model = DataParallel(model)
model.load_state_dict(torch.load("/home/quang/Downloads/arcface-pytorch/pretrainModel_ARC/train4.pth"))
# model.load_state_dict(torch.load("/home/quang/Downloads/arcface-pytorch/set/trainnew3.pth"))
model.to(device)
model.eval()
with open('./pretrainModel_ARC/X_train_triplet.pkl', 'rb') as f:
	X = pickle.load(f)
	X = np.array(X)
#     X = np.expand_dims(X, axis=3)
with open('./pretrainModel_ARC/y_train_triplet.pkl', 'rb') as f:
	y = pickle.load(f)
	y = np.array(y)
with open('./pretrainModel_ARC/name_map.pkl', 'rb') as f:
	name_map = pickle.load(f)
# print(X.shape)
with open('./pretrainModel_ARC/facebank.pkl', 'rb') as f:
	embs128 = pickle.load(f)
# Load model ssd nhan dien mat
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Doc file video input
vs = cv2.VideoCapture('/home/quang/Documents/mtcnn/file.mp4')
read = 0
saved = 0

# Lap qua cac frame cua video
while True:
	(grabbed, frame) = vs.read()
	# Neu khong doc duoc frame thi thoat
	if not grabbed:
		break

	read += 1
	if read % args["skip"] != 0:
		continue

	# Chuyen tu frame thanh blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
	# Phat hien cac khuon mat trong frame
	net.setInput(blob)
	detections = net.forward()
	# Neu tim thay it nhat 1 khuon mat
	if len(detections) > 0:
		# Tim khuon  mat to nhat trong anh
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]
		frame1=frame.copy()
		# Neu muc do nhan dien > threshold
		if confidence > args["confidence"]:
			#Tach khuon mat va ghi ra file
			time1=time.time()
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			frame1=cv2.rectangle(frame1, (startX, startY), (endX, endY), (0, 0, 255), 2)
			face =cv2.resize(frame[startY:endY, startX:endX], (128, 128))
			time2=time.time()
			data = processImageInput(face)
			t0 = time.time()
			# print('start')
			output = model(data)
			output = output.data.cpu().numpy()
			minimum = 99999
			person = -1
			diff = np.subtract(embs128, output[0])
			dist = np.sum(np.square(diff), 1)
			idx = np.argmin(dist)
			t1 = time.time()
			# print('finish',t1-t0)
			threshold = 100
			if dist[idx] < threshold:
				name = name_map[y[idx]]
			else:
				name = name_map[y[idx]]
			for k, e in enumerate(embs128):
				dist = np.linalg.norm(output[0] - e)
				if dist < minimum:
					minimum = dist
					person = k
			# write the frame to disk
			p = os.path.sep.join([args["output"],
				args["input"].split('/')[1] + "{}.png".format(saved)])
		cv2.imshow("img",frame1)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
			# cv2.imwrite(p, face)
			# saved += 1
			# print("[INFO] saved {} to disk".format(p))


vs.release()
cv2.destroyAllWindows()
