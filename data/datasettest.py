import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.utils.weight_norm as weight_norm

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
def conv3x3(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False)
class BasicBLock(nn.Module):
    expansion =1
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BasicBLock, self).__init__()
        self.conv1=conv3x3(inplanes,planes,stride)
        self.bn1=nn.BatchNorm2d(planes)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=conv3x3(planes,planes)
        self.bn2=nn.BatchNorm2d(planes)
        self.downsample=downsample
        self.stride=stride
    def forward(self,x):
        residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)

        if self.downsample is not None:
            residual=self.downsample(x)
        out+=residual
        out=self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion=4

    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1=nn.Conv2d(inplanes,planes,kernel_size=1,bias=False)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)

        self.bn2=nn.BatchNorm2d(planes)
        self.conv3=nn.Conv2d(planes,planes*self.expansion,kernel_size=1,bias=False)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)

        if self.downsample is not None:
            residual=self.downsample(x)
        out+=residual
        out=self.relu(out)

        return  out
class SEBlock(nn.Module):
    def __init__(self,channel,reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel,channel//reduction),
            nn.PReLU(),
            nn.Linear(channel//reduction,channel),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_=x.size()
        y=self.avg_pool(x).view(b,c)
        y=self.fc(y).view(b,c,1,1)
        return  x * y

class ResNetFace(nn.Module):
    def __init__(self,block,layers,use_se=True):
        self.inplanes=64
        self.use_se=use_se
        super(ResNetFace, self).__init__()
        self.conv1=nn.Conv2d(1,64,kernel_size=3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.prelu=nn.PReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.layer1=self._make_layer(block,64,layers[0])
        self.layer2=self._make_layer(block,128,layers[1],stride=2)
        self.layer3=self._make_layer(block,256,layers[2],stride=2)
        self.layer4=self._make_layer(block,512,layers[3],stride=2)
        self.bn4=nn.BatchNorm2d(512)
        self.dropout=nn.Dropout()
        self.fc5=nn.Linear(512*8*8,512)
        self.bn5=nn.BatchNorm1d(512)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,0)
        def _make_layer(self,block,planes,blocks,stride=1)
            downsample=None
            if stride!=1 or self.inplanes != planes * block.expansion:
                downsample=nn.Sequential(
                  nn.Conv2d(self.inplanes,planes* block.expansion,
                            kernel_size=1,stride=stride,bias=False),
                    nn.BatchNorm2d(planes * block.expansion)

                )
                layers=[]
                layers.append(block(self.inplanes,planes,stride,downsample,use_se=self.use_se))
                self.inplanes=planes
                for i in range(1,blocks):
                    layers.append(block(self.inplanes,planes,use_se=self.use_se))
                return nn.Sequential(layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            x = self.fc5(x)
            x = self.bn5(x)

            return x



