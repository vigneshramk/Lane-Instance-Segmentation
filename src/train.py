import os, sys
sys.path.append("..")
from models.enet_model import ENetModel
import torch.nn as nn
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
from torch.autograd import Variable
from .arguments import get_args
from .metrics.iou import IoU
import numpy as np
from .utils import save_checkpoint,CrossEntropyLoss2D,enet_weighing

args = get_args()

class TrainNetwork():

    def __init__(self,model,data_loader,num_classes,class_weights):

        #class_weights,class_encoding
        #num_classes = len(class_encoding)
        
        self.run_cuda = args.cuda and torch.cuda.is_available()
        print('Cuda is %r' %(self.run_cuda))
        self.model = model
        self.data_loader = data_loader
        self.data_size = len(self.data_loader)

        # The ENet paper uses the Adam optimizer
        self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay)

        # Cross Entropy loss between the prediction and the label
        # Use the class-weighting so as to account for the class imbalance
        self.criterion = CrossEntropyLoss2D(class_weights=class_weights) #weight=class_weights
        
        if self.run_cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
        
        # Intersection of Union as the metric to see the progress in training
        self.metric = IoU(num_classes)

        # Keeping this variable so as to resume training from a checkpoint later
        self.start_epoch = 0

    def save_model(self,run_name,epoch):
        filename = 'saved_models/' + run_name + '/' +'checkpoint_' + str(epoch) + '.h5'
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
        },filename=filename)

    def train_epoch(self,interactive=False):

        total_loss = 0.0
        self.metric.reset()

        for i,img_data in enumerate(self.data_loader):
            images, labels, size, name = img_data
            images,labels = images.type(torch.FloatTensor),labels.type(torch.LongTensor)
            if self.run_cuda:
                images = images.cuda()
                labels = labels.cuda()

            # labels = labels.unsqueeze_(1)    
            # labels = make_one_hot(labels,3)

            # Do the forward prop and compute the cross-entropy loss    
            output = self.model(images)
            loss = self.criterion(output,labels)

            #Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.data[0]
            self.metric.add(output.data,labels.data)

            if interactive:
                print("Mini-Batch-Number: %d, Loss : %.5f" %(i,loss.data[0]))

        return total_loss / self.data_size, self.metric.value()

    def train_model(self,interactive=True,save_freq=1,run_name="run_default"):

        for epoch in range(self.start_epoch,args.epochs):

            epoch_loss, (iou, miou) = self.train_epoch(interactive)

            # Save the checkpoint at the given frequency
            if epoch%save_freq == 0:
                self.save_model(run_name,epoch)

            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
              format(epoch, epoch_loss, miou))

            # Write out a verbose file with the output loss and metric data for every training epoch
            filename = 'saved_models/' + run_name + '/' + "verbose.txt"
            with open(filename, "w") as text_file:
                text_file.write(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
              format(epoch, epoch_loss, miou))


    # def make_one_hot(labels, num_classes):
    #         '''
    #         Converts an integer label torch.autograd.Variable to a one-hot Variable.

    #         Parameters
    #         ----------
    #         labels : torch.autograd.Variable of torch.cuda.LongTensor
    #             N x 1 x H x W, where N is batch size.
    #             Each value is an integer representing correct classification.
    #         Returns
    #         -------
    #         target : torch.autograd.Variable of torch.cuda.FloatTensor
    #             N x C x H x W, where C is class number. One-hot encoded.
    #         '''
    #         one_hot = torch.cuda.LongTensor(labels.size(0), num_classes, labels.size(2), labels.size(3)).zero_()
    #         target = one_hot.scatter_(1, labels.data, 1) 
    #         return target





