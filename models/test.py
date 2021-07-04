from torchvision import models
import torch
resnet = models.resnet18(pretrained=True)

from torchvision import transforms
transform = transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.ToTensor(),
 transforms.Normalize(
 mean=[0.485, 0.456, 0.406],
 std=[0.229, 0.224, 0.225]
 )])
from PIL import Image
img = Image.open("/home/quang/Downloads/arcface-pytorch/data/1.jpg")
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)
resnet.eval()
out = resnet(batch_t)
out.argmax() # 281
print(out.shape)