import torch
from torch.functional import Tensor
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

#reshape usato per il flusso teacher -> student
class Conv_reshape(nn.Module):
    def __init__(self, in_dim, factor):
        super(Conv_reshape, self).__init__()
        
        self.conv_reshape = nn.Conv2d(in_dim, in_dim // factor, kernel_size=1)

    def forward(self, x):
        return self.conv_reshape(x)

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, tipo_flusso = None, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        self.tipo_flusso = tipo_flusso       #conv2d, res[5], res[1,3,5], out, res[5]+out

        if (self.tipo_flusso == 'conv2d' or self.tipo_flusso == 'res[5]'):
            self.hook_layer = Tensor(0)
        elif (self.tipo_flusso == 'res[1,3,5]'):
            self.hook_layer = []

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

        if (self.tipo_flusso == 'conv2d' or self.tipo_flusso == 'res[5]' or self.tipo_flusso == 'out' or self.tipo_flusso == 'res[5]+out'):
            self.hook_layer = Tensor(0)
        elif (self.tipo_flusso == 'res[1,3,5]'):
            self.hook_layer = []

        self.x_conv1 = self.conv1(x)
        self.x_in1 = self.in1(self.x_conv1)
        self.x_relu1 = self.relu1(self.x_in1)

        # Down-sampling

        self.x_conv2_i = []
        self.x_in2_i = []
        self.x_relu2_i = []

        self.x_conv2_i.append(self.conv2[0](self.x_relu1))
        self.x_in2_i.append(self.in2[0](self.x_conv2_i[0]))
        self.x_relu2_i.append(self.relu2[0](self.x_in2_i[0]))

        self.x_conv2_i.append(self.conv2[1](self.x_relu2_i[0]))
        self.x_in2_i.append(self.in2[1](self.x_conv2_i[1]))
        self.x_relu2_i.append(self.relu2[1](self.x_in2_i[1]))
        
        if (self.tipo_flusso == 'conv2d'):
            self.hook_layer = self.conv2[1](self.x_relu2_i[0])         #flusso di prova con conv2d[1]

        # Bottleneck

        self.x_res_i = []

        self.x_res_i.append(self.res1[0](self.x_relu2_i[-1]))           
        if (self.tipo_flusso == 'res[1,3,5]' and len(self.res1) == 3):          # il primo blocco residual di 'student'
            self.hook_layer.append(self.res1[0](self.x_relu2_i[-1]))
            # print('hook: ', len(self.res1), 0)
        elif (len(self.res1) == 1 and (self.tipo_flusso == 'res[5]' or self.tipo_flusso == 'res[5]+out')):          # se abbiamo lo student con 1 blocco residual, prendiamo il primo ed unico blocco
            self.hook_layer = self.res1[0](self.x_relu2_i[-1])
        
        for i in range(1, len(self.res1)):
            self.res1[i].cuda()
            self.x_res_i.append(self.res1[i](self.x_res_i[i-1]))
            if (i == len(self.res1) - 5 and len(self.res1) == 6):
                if (self.tipo_flusso == 'res[1,3,5]'):                          # il primo blocco residual di 'teacher'
                    self.hook_layer.append(self.res1[i](self.x_res_i[i-1]))
                    # print('hook: ', len(self.res1), i)
            if (i == len(self.res1) - 2 and len(self.res1) == 3 or i == len(self.res1) - 3 and len(self.res1) == 6):
                if (self.tipo_flusso == 'res[1,3,5]'):                          # il secondo blocco residual
                    self.hook_layer.append(self.res1[i](self.x_res_i[i-1]))
                    # print('hook: ', len(self.res1), i)
            if (i == len(self.res1) - 1):                                       # il terzo blocco residual
                if (self.tipo_flusso == 'res[5]' or self.tipo_flusso == 'res[5]+out'):
                    self.hook_layer = self.res1[i](self.x_res_i[i-1])       #flusso con l'ultimo blocco residual
                elif (self.tipo_flusso == 'res[1,3,5]'):
                    self.hook_layer.append(self.res1[i](self.x_res_i[i-1]))
                    # print('hook: ', len(self.res1), i)
        # print()
            
        # Up-sampling

        self.x_convtr1_i = []
        self.x_in3_i = []
        self.x_relu3_i = []

        self.x_convtr1_i.append(self.convtr1[0](self.x_res_i[-1]))
        self.x_in3_i.append(self.in3[0](self.x_convtr1_i[0]))
        self.x_relu3_i.append(self.relu3[0](self.x_in3_i[0]))

        self.x_convtr1_i.append(self.convtr1[1](self.x_relu3_i[0]))
        self.x_in3_i.append(self.in3[1](self.x_convtr1_i[1]))
        self.x_relu3_i.append(self.relu3[1](self.x_in3_i[1]))

        self.x_conv3 = self.conv3(self.x_relu3_i[-1])
        self.x_tanh = self.tanh(self.x_conv3)

        #y = x_tanh

        return self.x_tanh    #, self.conv_reshape1(x_conv1)

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, tipo_flusso=None, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        
        self.tipo_flusso = tipo_flusso

        # if (self.tipo_flusso == 'conv2d' or self.tipo_flusso == 'res[5]' or self.tipo_flusso == 'out' or self.tipo_flusso == 'res[5]+out'):
        if(self.tipo_flusso):
            self.hook_layer = Tensor(0)
        # elif (self.tipo_flusso == 'res[1,3,5]'):
        #     self.hook_layer = []
        
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

        if(self.tipo_flusso):
            self.hook_layer = self.main(x)

        """ if (self.tipo_flusso == 'conv2d' or self.tipo_flusso == 'res[5]' or self.tipo_flusso == 'out' or self.tipo_flusso == 'res[5]+out'):
            self.hook_layer = Tensor(0)
        elif (self.tipo_flusso == 'res[1,3,5]'):
            self.hook_layer = [] """

        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
