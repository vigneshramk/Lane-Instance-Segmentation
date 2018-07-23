import torch.nn as nn
import torch
from torch.autograd import Variable

class ENetInitialBlock(nn.Module):

	def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=0,
                 bias=False,
                 nonlinearity='ReLU'):
        super().__init__()

        if nonlinearity is 'PReLU':
        	activation = nn.PReLU()
    	else:
    		activation = nn.ReLU()

		# Main branch which does MaxPooling with stride 2 on the 3 input channels
        self.main_branch = nn.MaxPool2d(kernel_size, stride=2, padding=padding)

		# Extension Conv Branch of the which outputs out_channels - 3 channels
		# since the 3 remaining channels come from the Main Branch above 
		self.conv_branch = nn.Conv2d(
            in_channels,
            out_channels - 3,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=bias)

        # Initialize batch normalization to be used after concatenation
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # PReLU layer to apply after concatenating the branches
        self.apply_activation = activation

    def forward(self, x):
        main_out = self.main_branch(x)
        conv_out = self.conv_branch(x)

        # Concatenate the Main and Extension branches
        out = torch.cat((conv_out,main_out), 1)
        # Apply batch norm 
        out = self.batch_norm(out)

        return self.apply_activation(out)


class ENetNormalBottleneck(nn.Module):

	def __init__(self,
                 in_channels,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric_conv=False,
                 dropout_prob=0,
                 bias=False,
                 reduce_ratio=4,
                 nonlinearity='ReLU'):
        super().__init__()

        if nonlinearity is 'PReLU':
        	activation = nn.PReLU()
    	else:
    		activation = nn.ReLU()

        # For the 1x1 convolution, check if the filter reduce ratio is above 1 and below the number of input channels
        if reduce_ratio <= 1 or reduce_ratio > channels:
            raise RuntimeError("The filter reduce ratio should be in the"
                               "interval [1, {0}], got reduce_ratio={1}."
                               .format(channels, reduce_ratio))

        reduced_channels = in_channels // reduce_ratio

        # Main Branch: For the Normal Bottleneck module where no downsampling occurs
        # the main branch output is just the input to the branch

        # Extension Branch
        # Step 1: Do a 1x1 projection convolution to do dimensionality reduction
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                reduced_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(reduced_channels), activation)

        # Step 2: Do either a symmetric 3x3 convolution or 
        # an assymetric convolution defined as two convolutions: 5x1 followed by a 1x5
        if assymetric_conv:
        	self.conv2 = nn.Sequential(nn.Conv2d(
                    reduced_channels,
                    reduced_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(reduced_channels), activation,
            	nn.Conv2d(
                    reduced_channels,
                    reduced_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(reduced_channels), activation)
        else:
        	self.conv2 = nn.Sequential(nn.Conv2d(
                    reduced_channels,
                    reduced_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(reduced_channels), activation)

    	# Step 3: Do the 1x1 expansion convolution to expand the number of filters
    	# to the input channels to the module
    	self.conv3 = nn.Sequential(nn.Conv2d(
                reduced_channels,
                channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(channels), activation)

        # Dropout as a weak form of regularization
        self.regularizer = nn.Dropout2d(p=dropout_prob)

        # Nonlinearity to apply the two branches are added
        self.apply_activation = activation

    def forward(self, x):
        
        # Main branch shorted with the input
        main = x

        # Extension branch
        x_conv = self.conv1(x)
        x_conv = self.conv2(x_conv)
        x_conv = self.conv3(x_conv)
        x_conv = self.regularizer(x_conv)

        # Add main and extension conv out branches
        out = main + x_conv

        return self.apply_activation(out)

class EnetDownsamplingBottleneck(nn.Module):


	def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=0,
                 dropout_prob=0,
                 bias=False,
                 reduce_ratio=4,
                 nonlinearity='ReLU',
                 return_indices=False):
        super().__init__()

        if nonlinearity is 'PReLU':
        	activation = nn.PReLU()
    	else:
    		activation = nn.ReLU()

        # For the 1x1 convolution, check if the filter reduce ratio is 
        # above 1 and below the number of input channels
        if reduce_ratio <= 1 or reduce_ratio > channels:
            raise RuntimeError("The filter reduce ratio should be in the"
                               "interval [1, {0}], got reduce_ratio={1}."
                               .format(channels, reduce_ratio))

        reduced_channels = in_channels // reduce_ratio

        # Main branch - max pooling followed by feature map (channels) padding
        self.maxpool = nn.MaxPool2d(
            kernel_size,
            stride=2,
            padding=padding,
            return_indices=return_indices)

        # Extension Conv branch
        # Do a 2x2 projection convolution with stride 2 to reduce the number of channels 
        self.conv1 = nn.Sequential(nn.Conv2d(
                in_channels,
                reduced_channels,
                kernel_size=2,
                stride=2,
                bias=bias), nn.BatchNorm2d(reduced_channels), activation)

        # Symmetric Convolution 
        self.conv2 = nn.Sequential(nn.Conv2d(
                reduced_channels,
                reduced_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias), nn.BatchNorm2d(reduced_channels), activation)

        # 1x1 expansion convolution to convert to the
        # given number of output channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                reduced_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels), activation)

        # Dropout as a weak form of regularization
        self.regularizer = nn.Dropout2d(p=dropout_prob)

        # Nonlinearity to apply the two branches are added
        self.apply_activation = activation

    def forward(self, x):
        
        # Main branch maxpooling followed by padding
        if self.return_indices:
            main, max_indices = self.maxpool(x)
        else:
            main = self.maxpool(x)

        # Extension Conv branch
        x_conv = self.conv1(x)
        x_conv = self.conv2(x_conv)
        x_conv = self.conv3(x_conv)
        x_conv = self.regularizer(x_conv)

        # Main branch channel padding
        n, ch_ext, h, w = x_conv.size()
        ch_main = main.size()[1]
        padding = Variable(torch.zeros(n, ch_ext - ch_main, h, w))
         
        # convert padding to GPU if necessary
        if main.is_cuda:
            padding = padding.cuda()

        # Zero Pad the main branch to match the number of feature maps
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + x_conv

        return self.apply_activation(out), max_indices








        






