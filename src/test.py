from .arguments import get_args
from models.enet_model import ENetModel
import torch.nn as nn
import torch,torchvision
import torch.optim as optim
from torch.autograd import Variable
from .arguments import get_args
from .metrics.iou import IoU
import numpy as np
from .utils import save_checkpoint,CrossEntropyLoss2D,enet_weighing
import cv2
import matplotlib.pyplot as plt
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

    def imshow_batch(self,images, labels):
        """Displays two grids of images. The top grid displays ``images``
        and the bottom grid ``labels``
        Keyword arguments:
        - images (``Tensor``): a 4D mini-batch tensor of shape
        (B, C, H, W)
        - labels (``Tensor``): a 4D mini-batch tensor of shape
        (B, C, H, W)
        """

        # Make a grid with the images and labels and convert it to numpy
        images = torchvision.utils.make_grid(images).numpy()
        labels = torchvision.utils.make_grid(labels).numpy()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
        ax1.imshow(np.transpose(images, (1, 2, 0)))
        ax2.imshow(np.transpose(labels, (1, 2, 0)))

        plt.show()

    def test_model(self,checkpoint_file):

        self.load_model(checkpoint_file)

        total_loss = 0.0
        self.metric.reset()

        for i,img_data in enumerate(self.data_loader):

            if i > 1:
                break

            images, size, name = img_data
            images_copy = images
            # img = images[0,:,:,:].cpu().data.numpy()
            # img = img.transpose(1,2,0)
            # img = img[:, :, ::-1]
            images = images.type(torch.FloatTensor)
            if self.run_cuda:
                images = images.cuda()
            # labels = labels.unsqueeze_(1)    
            # labels = make_one_hot(labels,3)

            # Do the forward prop and compute the cross-entropy loss    
            output = self.model(images)

            labels = output.cpu().detach()
            images_copy = images_copy.cpu()

            self.imshow_batch(images_copy,labels)


          
            # pred_label = output.cpu().data.numpy()
            # pred_label = np.argmax(pred_label[0],axis=0)
            # pl = np.expand_dims(pred_label,axis=2)
            # pl = np.repeat(pl,3,axis=2)
            # pl = pl*128

            # pl = cv2.cvtColor(pred_label,cv2.COLOR_GRAY2RGB)

            # cv2.imshow('Loaded image',pl)
            # cv2.waitKey(100)

            plt.imshow(np.transpose(labels, (1, 2, 0)))
            plt.show()
            # # plt.colorbar()
            # im = Image.fromarray(np.uint8(cm.gist_earth(pred_label[0,:,:])*255))

            # im.save('output.jpg')
            # loss = self.criterion(output,labels)

            # total_loss += loss.data[0]
            # self.metric.add(output.data,labels.data)

            print('Testing')
                # print("Mini-Batch-Number: %d, Loss : %.5f" %(i,loss.data[0]))

        cv2.destroyAllWindows()        
        return
