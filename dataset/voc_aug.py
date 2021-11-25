import os
import numpy as np
import cv2
import torch
import math
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .mytransforms import *
from PIL import Image
from torch.utils.data import DataLoader

class VOCAugDataSet(Dataset):
    def __init__(self, dataset_path='./CULane/list', data_list='train_gt', transform=None, mode='train'):

        with open(os.path.join(dataset_path, data_list + '.txt')) as f:
            self.mode = mode
            self.img_list = []
            self.img = []
            self.label_list = []
            self.exist_list = []
            for line in f:
                self.img.append(line.strip().split(" ")[0])
                self.img_list.append(dataset_path.replace('/list', '') + line.strip().split(" ")[0])
                self.label_list.append(dataset_path.replace('/list', '') + line.strip().split(" ")[1])
                if self.mode == 'train':
                    self.exist_list.append(np.array([int(line.strip().split(" ")[2]), int(line.strip().split(" ")[3]), int(line.strip().split(" ")[4]), int(line.strip().split(" ")[5])]))

        self.img_path = dataset_path
        self.gt_path = dataset_path
        self.transform = transform

        self.segment_transform = transforms.Compose([
            FreeScaleMask((288, 800)),
            MaskToTensor(),
        ])
        
        self.img_transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.origi_img_transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
        ])

        self.simu_transform = Compose2([
            RandomRotate(6),
            RandomUDoffsetLABEL(100),
            RandomLROffsetLABEL(200)
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = Image.open(self.img_list[idx])
        label = Image.open(self.label_list[idx])

        if self.mode == 'train':
            exist = self.exist_list[idx]

        if self.transform:
            if self.mode == 'train':
                image, label = self.simu_transform(image, label)
            
            original_img = image.copy()
            original_img = self.origi_img_transform(original_img)

            image = self.img_transform(image)
            label = self.segment_transform(label)

        if self.mode == 'test':
            return original_img, image, label, self.img[idx]
        elif self.mode == 'train':
            return image, label, exist

class MY_VOCAugDataSet(Dataset):
    def __init__(self, dataset_path='./CULane/list', data_list='train_gt', transform=None, mode='train'):

        with open(os.path.join(dataset_path, data_list + '.txt')) as f:
            self.mode = mode
            self.img_list = []
            self.img = []
            self.label_list = []
            self.exist_list = []
            for line in f:
                self.img.append(line.strip().split(" ")[0])
                self.img_list.append(dataset_path.replace('/list', '') + line.strip().split(" ")[0])
                self.label_list.append(dataset_path.replace('/list', '') + line.strip().split(" ")[1])
                if self.mode == 'train':
                    self.exist_list.append(np.array([int(line.strip().split(" ")[2]), int(line.strip().split(" ")[3]), int(line.strip().split(" ")[4]), int(line.strip().split(" ")[5])]))

        self.img_path = dataset_path
        self.gt_path = dataset_path
        self.transform = transform

        self.segment_transform = transforms.Compose([
            FreeScaleMask((288, 800)),
            MaskToTensor(),
        ])
        
        self.img_transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.origi_img_transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.simu_transform = Compose2([
            RandomRotate(6),
            RandomUDoffsetLABEL(100),
            RandomLROffsetLABEL(200)
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = Image.open(self.img_list[idx])
        label = Image.open(self.label_list[idx])

        if self.mode == 'train':
            exist = self.exist_list[idx]

        if self.transform:
            if self.mode == 'train':
                image, label = self.simu_transform(image, label)
            
            original_img = image.copy()
            original_img = self.origi_img_transform(original_img)

            image = self.img_transform(image)
            label_2D = self.segment_transform(label)

            np_label = np.array(label_2D)

            # background = np_label.copy()
            # background[background != 0] = 5
            # background[background == 0] = 1
            # background[background == 5] = 0
            # background = np.expand_dims(background, axis=2)

            lane_1 = np_label.copy()
            lane_1[lane_1 != 1] = 0
            lane_1 = np.expand_dims(lane_1, axis=2)

            lane_2 = np_label.copy()
            lane_2[lane_2 != 2] = 0
            lane_2[lane_2 == 2] = 1
            lane_2 = np.expand_dims(lane_2, axis=2)

            lane_3 = np_label.copy()
            lane_3[lane_3 != 3] = 0
            lane_3[lane_3 == 3] = 1
            lane_3 = np.expand_dims(lane_3, axis=2)

            lane_4 = np_label.copy()
            lane_4[lane_4 != 4] = 0
            lane_4[lane_4 == 4] = 1
            lane_4 = np.expand_dims(lane_4, axis=2)
            
            # out = np.concatenate((background, lane_1, lane_2, lane_3, lane_4), axis=2)
            out = np.concatenate((lane_1, lane_2, lane_3, lane_4), axis=2)
            label_3D = self.test_transform(out).float()


        if self.mode == 'test':
            return original_img, image, label_2D, label_3D, self.img[idx]
        elif self.mode == 'train':
            return image, label_2D, label_3D, exist


# if __name__ == "__main__":
#     train_dataset = VOCAugDataSet(dataset_path='D:/dataset/CUlane/list', data_list='train_gt', transform=True, mode='train')
#     val_dataset = VOCAugDataSet(dataset_path='D:/dataset/CUlane/list', data_list='val_gt', transform=True, mode='train')

#     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=0)

#     for i, (ori, input, target, cls_label, target_exist) in enumerate(train_loader):
#         print(cls_label.size())
#         cls_label = (cls_label.numpy())[0]

#         ori = (ori.numpy())[0]*255
#         ori = np.transpose(ori, (1, 2, 0)).astype('uint8')
#         ori = cv2.cvtColor(ori, cv2.COLOR_RGB2BGR)

#         input = (input.numpy())[0]
#         input = np.transpose(input, (1, 2, 0))
#         target = (target.numpy())[0]

#         for row in range(cls_label.shape[0]):
#             line = cls_label[row]
#             buf = 0
#             for x in range(len(line)-2):
#                 if line[x] != 0:
#                     loc = line[x]
#                     buf = loc + buf
#                     ori = cv2.circle(ori, (int(buf*800), row*8), 3, (255, 255, 0), -1)

#         plt.figure(1)
#         plt.subplot(211)
#         plt.imshow(ori)

#         plt.subplot(212)
#         plt.imshow(target)
#         plt.show()

#         break