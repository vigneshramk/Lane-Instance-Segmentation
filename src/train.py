import os, sys
sys.path.append("..")
from models.enet_model import ENetModel
import torch.nn as nn
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
from torch.autograd import Variable
from arguments import get_args

args = get_args()

class TrainNetwork():

	def __init__(self,model,data_loader,class_weights,class_encoding):

		num_classes = len(class_encoding)
		self.run_cuda = args.cuda and torch.cuda.is_available()
		self.model = model
		self.data_loader = data_loader
		self.data_size = len(self.data_loader)

		# The ENet paper uses the Adam optimizer
		self.optimizer = optim.Adam(
			        self.model.parameters(),
			        lr=args.learning_rate,
			        weight_decay=args.weight_decay)

		self.criterion = nn.CrossEntropyLoss(weight=class_weights)
		if self.run_cuda:
	        model = model.cuda()
	        criterion = criterion.cuda()
		self.metric = IoU(num_classes)


	def run_epoch(self, interactive=False):
		total_loss = 0.0
		self.metric.reset()
		# Change according to the loader
		for i,data in enumerate(self.data_loader):
			inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            if self.run_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Do the forward prop and compute the cross-entropy loss    
            output = self.model(inputs)
            loss = self.criterion(output,labels)

            #Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.data[0]
            self.metric.add(outputs.data,labels.data)

            if interactive:
            	print("Mini-Batch-Number: %d, Loss : %.2f" %(i,loss.data[0]))

    	return epoch_loss / self.data_size, self.metric.value()



