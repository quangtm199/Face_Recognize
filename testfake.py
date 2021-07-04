# USAGE
# python liveness_demo.py

# import the necessary packages
from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import torch
from model.lib import *
import cv2
import os
from model.MobileNet import load_weight,MobileNetV2,moilenetv2
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"

# Cai dat cac tham so dau vao
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default='liveness.model',
	help="path to trained model")
ap.add_argument("-l", "--le", type=str, default='le.pickle',
	help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, default='face_detector',
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
class_index = ["fake", "real"]

def predict_max(output,class_index):  # [0.9, 0.1]
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
# model = load_model(args["model"])
le = pickle.loads(open("/home/quang/Documents/face_anti_spoofing/le.pkl", "rb").read())
model=MobileNetV2()
# model_path="/home/quang/Documents/face_anti_spoofing/model.pth"
# checkpoint=torch.load(model_path)
state_dict=model.state_dict()
model_dict={}
# for (k,v) in checkpoint.items():
#     if k[7:] in state_dict:
#         model_dict[k[7:]]=v
state_dict.update(model_dict)
model.load_state_dict(torch.load("/home/quang/Documents/face_anti_spoofing/weight_1.pth"))
model.to(torch.device("cpu"))
model.eval()

resize=224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
tranforms=transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])
# image=cv2.imread("/home/quang/Documents/face_anti_spoofing/dataset/val/real/home0.png")
# small_frame = cv2.resize(image, (224, 224))
# pil_im = Image.fromarray(small_frame)
# img=tranforms(pil_im)
# img = np.array(img)
# img = np.expand_dims(img, 0)
# img = torch.from_numpy(img)
# img=img.to(torch.device("cpu"))
# output_1 = model(img)
# # predict=predict_max(output_1,class_index)
# # print(predict)
# soft_output = torch.softmax(output_1, dim=-1)
# preds = soft_output.to('cpu').detach().numpy()
# _, predicted = torch.max(soft_output.data, 1)
# predicted_2 = predicted.to('cpu').detach().numpy()
# print('preds',preds[0][predicted_2])
# print(predicted_2)

#  Doc video tu webcam
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
vs = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
	# Doc anh tu webcam
	cap,frame = vs.read()
	# Chuyen thanh blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# Phat hien khuon mat
	net.setInput(blob)
	detections = net.forward()
	print(detections.shape)
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
			# face = cv2.resize(face, (224, 224))
			# face = face.astype("float") / 255.0
			# face = img_to_array(face)
			# face = np.expand_dims(face, axis=0)

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
			print('preds', preds[0][predicted_2])
			print(predicted_2)


			# Dua vao model de nhan dien fake/real
			# preds = model.predict(face)[0]

			# j = np.argmax(preds)
			label = class_index[int(predicted_2)]
			j=int(predicted_2)
			# Ve hinh chu nhat quanh mat
			label = "{}: {:.4f}".format(label, preds[0][j])
			if (j==0):
				# Neu la fake thi ve mau do
				cv2.putText(frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
			else:
				# Neu real thi ve mau xanh
				cv2.putText(frame, label, (startX, startY - 10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
							  (0,  255,0), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# Bam 'q' de thoat
	if key == ord("q"):
		break
cv2.destroyAllWindows()
vs.stop()