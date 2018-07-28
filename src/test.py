from .arguments import get_args
from models.enet_model import ENetModel
import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
from .arguments import get_args
from .metrics.iou import IoU
import numpy as np
from .utils import save_checkpoint,CrossEntropyLoss2D,enet_weighing

args = get_args()

class TestNetwork():

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
        self.criterion = CrossEntropyLoss2D(weight=class_weights) #weight=class_weights
        
        if self.run_cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
        
        # Intersection of Union as the metric 
        self.metric = IoU(num_classes)

    def load_model(self,checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def test_model(self,checkpoint_file):

        self.load_model(checkpoint_file)

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

            total_loss += loss.data[0]
            self.metric.add(output.data,labels.data)

            if interactive:
                print("Mini-Batch-Number: %d, Loss : %.5f" %(i,loss.data[0]))

        return total_loss / self.data_size, self.metric.value()
