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





