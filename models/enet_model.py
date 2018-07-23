from .enet_blocks import ENetInitialBlock,ENetNormalBottleneck
from .enet_blocks import ENetDownsamplingBottleneck,ENetUpsamplingBottleneck
import torch.nn as nn
import torch
from torch.autograd import Variable

class ENetModel(nn.Module):

    def __init__(self, num_classes, encoder_nonlinearity='PReLU', decoder_nonlinearity='ReLU'):
        super().__init__()

        # The initial block with 3 input channels and 16 output channels
        self.initial_block = ENetInitialBlock(3,16,padding=1,nonlinearity=encoder_nonlinearity)

        # Stage 1 (Encoder) - One downsampling and 4 normal bottleneck modules
        self.bottleneck1_0 = ENetDownsamplingBottleneck(
            16,
            64,
            padding=1,
            return_indices=True,
            dropout_prob=0.01,
            nonlinearity=encoder_nonlinearity)
        self.bottleneck1_1 = ENetNormalBottleneck(
            64, padding=1, dropout_prob=0.01, nonlinearity=encoder_nonlinearity)
        self.bottleneck1_2 = ENetNormalBottleneck(
            64, padding=1, dropout_prob=0.01, nonlinearity=encoder_nonlinearity)
        self.bottleneck1_3 = ENetNormalBottleneck(
            64, padding=1, dropout_prob=0.01, nonlinearity=encoder_nonlinearity)
        self.bottleneck1_4 = ENetNormalBottleneck(
            64, padding=1, dropout_prob=0.01, nonlinearity=encoder_nonlinearity)

        # Stage 2 (Encoder) - downsampling followed by dilated and assymetric conv layers
        self.bottleneck2_0 = ENetDownsamplingBottleneck(
            64,
            128,
            padding=1,
            return_indices=True,
            dropout_prob=0.1,
            nonlinearity=encoder_nonlinearity)
        # Normal bottleneck
        self.bottleneck2_1 = ENetNormalBottleneck(
            128, padding=1, dropout_prob=0.1, nonlinearity=encoder_nonlinearity)
        # Dilated Conv layer
        self.bottleneck2_2 = ENetNormalBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.1, nonlinearity=encoder_nonlinearity)
        # Assymetric Conv layer
        self.bottleneck2_3 = ENetNormalBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            nonlinearity=encoder_nonlinearity)
        # Dilated Conv layer
        self.bottleneck2_4 = ENetNormalBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.1, nonlinearity=encoder_nonlinearity)
        # Normal bottleneck
        self.bottleneck2_5 = ENetNormalBottleneck(
            128, padding=1, dropout_prob=0.1, nonlinearity=encoder_nonlinearity)
        # Dilated Conv layer
        self.bottleneck2_6 = ENetNormalBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.1, nonlinearity=encoder_nonlinearity)
        # Assymetric Conv layer
        self.bottleneck2_7 = ENetNormalBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            nonlinearity=encoder_nonlinearity)
        # Dilated Conv layer
        self.bottleneck2_8 = ENetNormalBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.1, nonlinearity=encoder_nonlinearity)


        # Stage 3 (Encoder) - Repeat Stage 2 without the downsampling module

        # Normal bottleneck
        self.bottleneck3_0 = ENetNormalBottleneck(
            128, padding=1, dropout_prob=0.1, nonlinearity=encoder_nonlinearity)
        # Dilated Conv layer
        self.bottleneck3_1 = ENetNormalBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.1, nonlinearity=encoder_nonlinearity)
        # Assymetric Conv layer
        self.bottleneck3_2 = ENetNormalBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            nonlinearity=encoder_nonlinearity)
        # Dilated Conv layer
        self.bottleneck3_3 = ENetNormalBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.1, nonlinearity=encoder_nonlinearity)
        # Normal bottleneck
        self.bottleneck3_4 = ENetNormalBottleneck(
            128, padding=1, dropout_prob=0.1, nonlinearity=encoder_nonlinearity)
        # Dilated Conv layer
        self.bottleneck3_5 = ENetNormalBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.1, nonlinearity=encoder_nonlinearity)
        # Assymetric Conv layer
        self.bottleneck3_6 = ENetNormalBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            nonlinearity=encoder_nonlinearity)
        # Dilated Conv layer
        self.bottleneck3_7 = ENetNormalBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.1, nonlinearity=encoder_nonlinearity)

        # Stage 4 (Decoder) - Upsampling followed by normal bottleneck
        # Upsampling module
        self.bottleneck4_0 = ENetUpsamplingBottleneck(
            128, 64, padding=1, dropout_prob=0.1, nonlinearity=encoder_nonlinearity)
        # 2 normal bottleneck layers
        self.bottleneck4_1 = ENetNormalBottleneck(
            64, padding=1, dropout_prob=0.1, nonlinearity=encoder_nonlinearity)
        self.bottleneck4_2 = ENetNormalBottleneck(
            64, padding=1, dropout_prob=0.1, nonlinearity=encoder_nonlinearity)

        # Stage 5 (Decoder) - Upsampling followed by normal bottleneck
        self.bottleneck5_0 = ENetUpsamplingBottleneck(
            64, 16, padding=1, dropout_prob=0.1, nonlinearity=encoder_nonlinearity)
        self.bottleneck5_1 = ENetNormalBottleneck(
            16, padding=1, dropout_prob=0.1, nonlinearity=encoder_nonlinearity)
        
        # A bare full convolution at the end of the network
        self.full_conv = nn.ConvTranspose2d(
            16,
            num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False)

    def forward(self,x):
        # Initial block
        x = self.initial_block(x)

        # Stage 1 - Encoder
        x, return_indices1_0 = self.bottleneck1_0(x)
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)

        # Stage 2 - Encoder
        x, return_indices2_0 = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x)
        x = self.bottleneck2_8(x)

        # Stage 3 - Encoder
        x = self.bottleneck3_0(x)
        x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x)
        x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x)
        x = self.bottleneck3_5(x)
        x = self.bottleneck3_6(x)
        x = self.bottleneck3_7(x)

        # Stage 4 - Decoder
        x = self.bottleneck4_0(x, return_indices2_0)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)

        # Stage 5 - Decoder
        x = self.bottleneck5_0(x, return_indices1_0)
        x = self.bottleneck5_1(x)

        # Final bare full convolution
        x = self.full_conv(x)

        return x





