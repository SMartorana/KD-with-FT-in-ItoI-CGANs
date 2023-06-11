import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.in1 = nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)

        # Down-sampling layers.

        self.conv2 = nn.ModuleList()
        self.in2 = nn.ModuleList()
        self.relu2 = nn.ModuleList()

        curr_dim = conv_dim
        for i in range(2):
            self.conv2.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            self.in2.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            self.relu2.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.

        self.res1 = nn.ModuleList()
        for i in range(repeat_num):
            self.res1.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.

        self.convtr1 = nn.ModuleList()
        self.in3 = nn.ModuleList()
        self.relu3 = nn.ModuleList()

        for i in range(2):
            self.convtr1.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            self.in3.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            self.relu3.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.conv3 = nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        x_conv1 = self.conv1(x)
        x_in1 = self.in1(x_conv1)
        x_relu1 = self.relu1(x_in1)

        # Down-sampling

        x_conv2_i = []
        x_in2_i = []
        x_relu2_i = []

        x_conv2_i.append(self.conv2[0](x_relu1))
        x_in2_i.append(self.in2[0](x_conv2_i[0]))
        x_relu2_i.append(self.relu2[0](x_in2_i[0]))

        x_conv2_i.append(self.conv2[1](x_relu2_i[0]))
        x_in2_i.append(self.in2[1](x_conv2_i[1]))
        x_relu2_i.append(self.relu2[1](x_in2_i[1]))

        # Bottleneck

        x_res_i = []

        x_res_i.append(self.res1[0](x_relu2_i[-1]))

        for i in range(1, len(self.res1)):
           # print(i)
            self.res1[i].cuda()
            x_res_i.append(self.res1[i](x_res_i[i-1]))
            
        # Up-sampling

        x_convtr1_i = []
        x_in3_i = []
        x_relu3_i = []

        x_convtr1_i.append(self.convtr1[0](x_res_i[-1]))
        x_in3_i.append(self.in3[0](x_convtr1_i[0]))
        x_relu3_i.append(self.relu3[0](x_in3_i[0]))

        x_convtr1_i.append(self.convtr1[1](x_relu3_i[0]))
        x_in3_i.append(self.in3[1](x_convtr1_i[1]))
        x_relu3_i.append(self.relu3[1](x_in3_i[1]))

        x_conv3 = self.conv3(x_relu3_i[-1])
        x_tanh = self.tanh(x_conv3)

        return x_tanh

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
