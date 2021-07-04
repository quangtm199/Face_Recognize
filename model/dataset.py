from lib import *
def make_datapath_list(phase="train"):
    rootpath="/home/quang/Documents/face_anti_spoofing/dataset/"
    target_path=osp.join(rootpath+phase+"/**/*.png")

    path_list=[]
    for path in glob.glob(target_path):
        path_list.append(path)
    return path_list
class ImageTransform():
    def __init__(self,resize,mean,std):
        self.data_transform={
            'train':transforms.Compose([
                transforms.RandomResizedCrop(resize,scale=(0.5,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ]),
            'val':transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean,std),
            ]),
            'test':transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])
        }
    def __call__(self,img,phase='train'):
        return self.data_transform[phase](img)
class MyDataSet(data.Dataset):
    def __init__(self,file_list,transforms=None,phase="train"):
        self.file_list=file_list
        self.transforms=transforms
        self.phase=phase
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path=self.file_list[idx]
        img=Image.open(img_path)
        img_transformed=self.transforms(img,self.phase)
        label=img_path.strip().split("/")[-2]
        if label=="fake":
            label=0
        elif label=="real":
            label=1
        return  img_transformed,label


if __name__=="__main__":
    path_list=make_datapath_list()
    print(path_list)
    print(path_list[1].strip().split("/")[-2])