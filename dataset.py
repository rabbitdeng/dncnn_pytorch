import torch.utils.data as data
import os
import PIL.Image as Image
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

transform = A.Compose([
    A.RandomCrop(height=128, width=128),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Resize(64, 64),

    ToTensorV2(),
])


class ImgDenoiseDataset(data.Dataset):

    def __init__(self, root, sigma=15):
        file_list = os.listdir(root)
        self.gtimgs_list = []
        self.sigma = sigma
        for item in file_list:
            for i in range(0, 100):
                self.gtimgs_list.append(os.path.join(root, item))

    def __getitem__(self, index):
        gt_img = Image.open(self.gtimgs_list[index])
        gt_img = np.array(gt_img)
        img_y = transform(image=gt_img)['image'].to(
            dtype=torch.float32) / 255.0  # target img_y is the same as ground_truth image.
        noise = torch.randn(img_y.size()).mul_(self.sigma / 255.0)
        img_x = img_y + noise

        return img_x, img_y

    def __len__(self):
        return len(self.gtimgs_list)


class TestDataset(data.Dataset):

    def __init__(self, root, sigma=15):
        file_list = os.listdir(root)
        self.gtimgs_list = []
        self.sigma = sigma
        for item in file_list:
            self.gtimgs_list.append(os.path.join(root, item))

    def __getitem__(self, index):
        gt_img = Image.open(self.gtimgs_list[index])
        gt_img = np.array(gt_img)
        img_y = transform(image=gt_img)['image'].to(
            dtype=torch.float32) / 255.0  # target img_y is the same as ground_truth image.
        noise = torch.randn(img_y.size()).mul_(self.sigma / 255.0)
        img_x = img_y + noise

        return img_x, img_y

    def __len__(self):
        return len(self.gtimgs_list)
