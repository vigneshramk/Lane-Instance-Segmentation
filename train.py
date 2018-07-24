import argparse
import cv2
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
import pickle
from PIL import Image
from dataset.bdd_dataset import BDD_Train_DataSet, BDD_Valid_DataSet, BDD_Test_DataSet
import matplotlib.pyplot as plt
import random
import timeit

start = timeit.default_timer()

BATCH_SIZE = 5
DATA_DIRECTORY = './dataset/bdd100k'
DATA_LIST_PATH = './dataset/list/train_list.txt'
INPUT_SIZE = '720,720'
NUM_CLASSES = 21

def get_arguments():
    parser = argparse.ArgumentParser(description="Lane Instance Segmentation - BDD Dataset")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    return parser.parse_args()

args = get_arguments()

def main():

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    train_dataset = BDD_Train_DataSet(args.data_dir, args.data_list, crop_size=input_size)

    train_dataset_size = len(train_dataset)

    trainloader = data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)

    for img_data in trainloader:
        images, labels, size, name = img_data
        print('Unique Label IDs:{}'.format(np.unique(labels[0,:,:].cpu().data.numpy())))
        img = images[0,:,:,:].cpu().data.numpy()
        img = img.transpose(1,2,0)
        img = img[:, :, ::-1]
        cv2.imshow('Loaded image',img)
        cv2.waitKey()
    
    end = timeit.default_timer()
    print (end-start,'seconds')

if __name__ == '__main__':
    main()
