import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
from PIL import Image

class BDD_Train_DataSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(321, 321)):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "bdd100k_images/100k/train/%s.jpg" % name)
            label_file = osp.join(self.root, "bdd100k_drivable_maps/drivable_maps/labels/train/%s_drivable_id.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        
        size = image.shape
        name = datafiles["name"]
        #image = np.asarray(image, np.float32)
        
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        #image = np.asarray(cv2.resize(img_pad, dsize = (self.crop_h,self.crop_w), interpolation = cv2.INTER_NEAREST), np.float32)
        #label = np.asarray(cv2.resize(label_pad, dsize = (self.crop_h,self.crop_w), interpolation = cv2.INTER_NEAREST), np.float32)

        image = img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w]
        label = label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w]
        

        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        
        return image.copy(), label.copy(), np.array(size), name


class BDD_Valid_DataSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(321, 321)):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.crop_h, self.crop_w = crop_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "bdd100k_images/100k/val/%s.jpg" % name)
            label_file = osp.join(self.root, "bdd100k_drivable_maps/drivable_maps/labels/val/%s_drivable_id.png" % name)
            
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        #image = np.asarray(image, np.float32)
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        #image = np.asarray(cv2.resize(img_pad, dsize = (self.crop_h,self.crop_w), interpolation = cv2.INTER_NEAREST), np.float32)
        #label = np.asarray(cv2.resize(label_pad, dsize = (self.crop_h,self.crop_w), interpolation = cv2.INTER_NEAREST), np.float32)

        image = img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w]
        label = label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w]

        
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        
        return image.copy(), label.copy(), np.array(size), name

class BDD_Test_DataSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(224, 224)):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "bdd100k_images/100k/test/%s.jpg" % name)
            self.files.append({
                "img": img_file
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        size = image.shape
        name = osp.splitext(osp.basename(datafiles["img"]))[0]
        #image = np.asarray(image, np.float32)
        
        img_h, img_w, _ = image.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
        image = image.transpose((2, 0, 1))
        return image, name, size


if __name__ == '__main__':
    dst = BDD_Train_DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
