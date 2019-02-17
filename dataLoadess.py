from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transform


class Imgdataset(Dataset):

    def __init__(self, path):
        super(Imgdataset, self).__init__()
        self.data = []
        if os.path.exists(path):
            dir_list = os.listdir(path)
            LR_dir = dir_list[0]
            HR_dir = dir_list[1]
            LR_path = path + '/' + LR_dir
            HR_path = path + '/' + HR_dir
            if os.path.exists(LR_path) and os.path.exists(HR_path):
                LR_data = os.listdir(LR_path)
                HR_data = os.listdir(HR_path)
                self.data = [{'LR':LR_path +'/'+ LR_data[i], 'HR':HR_path +'/'+ HR_data[i]} for i in range(len(LR_data))]
            else:
                raise FileNotFoundError('path doesnt exist!')

    def __getitem__(self, index):

        LR_path, HR_path = self.data[index]["LR"], self.data[index]["HR"]
        tran = transform.ToTensor()

        LR_img = Image.open(LR_path)
        HR_img = Image.open(HR_path)
        # print(tran(img).shape)

        return tran(LR_img), tran(HR_img)

    def __len__(self):

        return len(self.data)

