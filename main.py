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
from src.test import TestNetwork
from models.enet_model import ENetModel
from src.arguments import get_args
from src.utils import enet_weighing

args = get_args()

#Select the GPU to run the network on
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_select

def train_main():

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    crop_size = (int(h),int(w))

    # Create the directory to store the checkpoints and verbose text files
    directory = 'saved_models/' + args.run_name + '/'
    if not os.path.exists(directory):
            os.makedirs(directory)

    # Initialize the Train Dataset        
    train_dataset = BDD_Train_DataSet(args.data_dir, args.data_list, crop_size=crop_size)
    train_dataset_size = len(train_dataset)

    num_classes =3
    
    #Pass the dataset to the dataloader
    data_loader = data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    model = ENetModel(num_classes)
    
    # Calculate the class weights
    print('Calculating class weights')    
    class_weights = enet_weighing(data_loader, num_classes)

    
    # Initialize the train network class with the dataloader
    train_net = TrainNetwork(model,data_loader,num_classes,class_weights)

    # Train the model for the given number of epochs
    print("Training the model")
    train_net.train_model(interactive=False,run_name=args.run_name)

def test_main():

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    # Initialize the Test Dataset        
    test_dataset = BDD_Test_DataSet(args.data_dir, args.data_list, crop_size=input_size)
    test_dataset_size = len(test_dataset)

    num_classes =3
    
    #Pass the dataset to the dataloader
    data_loader = data.DataLoader(test_dataset,batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    model = ENetModel(num_classes)

    # Calculate the class weights
    class_weights = None
    # print('Calculating class weights')    
    # class_weights = enet_weighing(data_loader, num_classes)

    # Initialize the train network class with the dataloader
    test_net = TestNetwork(model,data_loader,num_classes,class_weights)

    # Test the model
    print("Testing the model")
    test_net.test_model(args.load)



def main():

    if 'train' in args.mode:
        train_main()
    elif 'test' in args.mode:
        test_main()
    else:
        print(args.mode)
        print("Invalid mode, select train (default) or test")

if __name__ == '__main__':
    main()
