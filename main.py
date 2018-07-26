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
from src.train import TrainNetwork
from models.enet_model import ENetModel


BATCH_SIZE = 10
DATA_DIRECTORY = './dataset/bdd100k'
DATA_LIST_PATH = './dataset/list/train_list.txt'
INPUT_SIZE = '720,720'

#Select the GPU to run the network on
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_select

def main():

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    crop_size = (int(h),int(w))

    directory = 'saved_models/' + args.run_name + '/'
    if not os.path.exists(directory):
            os.makedirs(directory)

    train_dataset = BDD_Train_DataSet(args.data_dir, args.data_list, crop_size=crop_size)
    train_dataset_size = len(train_dataset)

    num_classes =3
    
    data_loader = data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    model = ENetModel(num_classes)

    train_net = TrainNetwork(model,data_loader,num_classes)

    train_net.train_model(interactive=False,run_name=args.run_name)


if __name__ == '__main__':
    main()
