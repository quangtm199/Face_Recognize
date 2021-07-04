import torch

from lib import *
from model.dataset import MyDataSet,ImageTransform,make_datapath_list
from model.MobileNet import load_weight,MobileNetV2,moilenetv2
from model.loss import FocalLoss
device="cpu"
def train_model(net,dataloader_dict,criterior,optimizer,num_epochs,save_path):
    device="cpu"
    print("device",device)
    for epoch in range(num_epochs):
        print("Epoch  {}/{}".format(epoch,num_epochs))
        net.to(device)
        torch.backends.cudnn.benchmark=True
        for phase in ["train","val"]:
            if phase =="train":
                net.train()
            else:
                net.eval()
            epoch_loss=0.0
            epoch_corrects=0
            if(epoch ==0)and (phase=="train"):
                continue
            for inputs,labels in tqdm(dataloader_dict[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=="train"):
                    outputs=net(inputs)
                    loss=criterior(outputs,labels)
                    _ , preds = torch.max(outputs,1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    epoch_loss+=loss.item()*inputs.size(0)
                    epoch_corrects+=torch.sum(preds==labels.data)
            epoch_loss=epoch_loss/len(dataloader_dict[phase].dataset)
            epoch_accuracy=epoch_corrects.double()/len(dataloader_dict[phase].dataset)
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))
    torch.save(net.state_dict(), save_path)
if __name__=="__main__":
    train_list=make_datapath_list("train")
    val_list=make_datapath_list("val")
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    batch_size=32
    num_epochs=20
    train_data=MyDataSet(train_list,transforms=ImageTransform(resize,mean,std),phase="train")
    val_data=MyDataSet(val_list,transforms=ImageTransform(resize,mean,std),phase="val")
    train_dataloader=torch.utils.data.DataLoader(train_data,batch_size,shuffle=True)
    val_dataloader=torch.utils.data.DataLoader(val_data,batch_size,shuffle=False)
    dataloader_dict={"train":train_dataloader,"val":val_dataloader}


    # model=MobileNetV2()
    # model_path="/home/quang/Downloads/FeatherNets_Face-Anti-spoofing-Attack-Detection-Challenge-CVPR2019/_4_best.pth.tar"
    # checkpoint=torch.load(model_path)
    # state_dict=model.state_dict()
    # model_dict={}
    # for (k,v) in checkpoint['state_dict'].items():
    #     if k[7:] in state_dict:
    #         model_dict[k[7:]]=v
    # state_dict.update(model_dict)
    # model.load_state_dict(state_dict,strict=True)

    model=moilenetv2()


    criterion = FocalLoss(device, 2, gamma=2)
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9,
                                weight_decay=0.0001)
    save_path = '/home/quang/Documents/face_anti_spoofing/model/weight_fine_tuning.pth'


    train_model(model,dataloader_dict,criterion,optimizer,num_epochs,save_path)
