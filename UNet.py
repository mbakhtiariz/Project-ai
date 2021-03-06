# Author: Alexander Hustinx
# Date: 11-06-2018
#
# U-Net implementation
# 	inspired by (U-Net: Convolutional Networks for Biomedical Image Segmentation)
# Version: v0.1

from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn.functional as F

# These are only required when actually creating an object and running
from torch.utils.data import Dataset, DataLoader
import argparse
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import math

# For reproducable results
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


# Expected input size: 572x572
class UNet(torch.nn.Module):
    def __init__(self, upsample_mode='bilinear'):
        super(UNet, self).__init__()

        # Down sampling layer, depth: 0	(1)
        self.conv1_1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0)
        self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Down sampling layer, depth: 1	(2)
        self.conv2_1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv2_2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Down sampling layer, depth: 2	(3)
        self.conv3_1 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv3_2 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.max_pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Down sampling layer, depth: 3	(4)
        self.conv4_1 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.conv4_2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.max_pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Bottom layer 			(5)
        self.conv5_1 = torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0)
        self.conv5_2 = torch.nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0)

        # Up sampling layer, depth: 3	(6)
        if (upsample_mode == 'linear'
                or upsample_mode == 'bilinear'
                or upsample_mode == 'trilinear'
                or upsample_mode == 'nearest'):

            self.up6 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=True)
        elif upsample_mode == 'transpose':
            self.up6 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        else:
            # NOT implemented yet.., just future-proofing
            self.up6 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # According to the papers this should be 2x2, and no padding...
        # that would reduce the size by 1x1 and that would be incorrect...

        if upsample_mode == 'transpose':
            self.conv6_1 = None
        else:
            self.conv6_1 = torch.nn.Conv2d(1024, 512, kernel_size=1, stride=1)
        self.conv6_2 = torch.nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0)
        self.conv6_3 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        # Up sampling layer, depth: 2	(7)
        if (upsample_mode == 'linear'
                or upsample_mode == 'bilinear'
                or upsample_mode == 'trilinear'
                or upsample_mode == 'nearest'):

            self.up7 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=True)
        elif upsample_mode == 'transpose':
            self.up7 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        else:
            # NOT implemented yet.., just future-proofing
            self.up7 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # According to the papers this should be 2x2, and no padding...
        # that would reduce the size by 1x1 and that would be incorrect...

        if upsample_mode == 'transpose':
            self.conv7_1 = None
        else:
            self.conv7_1 = torch.nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.conv7_2 = torch.nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.conv7_3 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)

        # Up sampling layer, depth: 1	(8)
        if (upsample_mode == 'linear'
                or upsample_mode == 'bilinear'
                or upsample_mode == 'trilinear'
                or upsample_mode == 'nearest'):

            self.up8 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=True)
        elif upsample_mode == 'transpose':
            self.up8 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        else:
            # NOT implemented yet.., just future-proofing
            self.up8 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # According to the papers this should be 2x2, and no padding...
        # that would reduce the size by 1x1 and that would be incorrect...

        if upsample_mode == 'transpose':
            self.conv8_1 = None
        else:
            self.conv8_1 = torch.nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.conv8_2 = torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.conv8_3 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)

        # Up sampling layer, depth: 0	(9)
        if (upsample_mode == 'linear'
                or upsample_mode == 'bilinear'
                or upsample_mode == 'trilinear'
                or upsample_mode == 'nearest'):

            self.up9 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=True)
        elif upsample_mode == 'transpose':
            self.up9 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        else:
            # NOT implemented yet.., just future-proofing
            self.up9 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # According to the papers this should be 2x2, and no padding...
        # that would reduce the size by 1x1 and that would be incorrect...

        if upsample_mode == 'transpose':
            self.conv9_1 = None
        else:
            self.conv9_1 = torch.nn.Conv2d(128, 64, kernel_size=1, stride=1)
        self.conv9_2 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.conv9_3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        # Output layer, depth: 0			(10)
        self.conv10_1 = torch.nn.Conv2d(64, 1, kernel_size=1, stride=1)

        # Softmax.. why not...
        # self.softmax = torch.nn.Softmax2d()
        # self.tanh = torch.nn.Tanh()

        # Init weights
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        # Note: I assume the crop in the paper is a center crop

        x = F.leaky_relu(self.conv1_1(x), 1e-2)
        out1 = F.leaky_relu(self.conv1_2(x), 1e-2)

        x = self.max_pool1(out1)
        x = F.leaky_relu(self.conv2_1(x), 1e-2)
        out2 = F.leaky_relu(self.conv2_2(x), 1e-2)

        x = self.max_pool2(out2)
        x = F.leaky_relu(self.conv3_1(x), 1e-2)
        out3 = F.leaky_relu(self.conv3_2(x), 1e-2)

        x = self.max_pool3(out3)
        x = F.leaky_relu(self.conv4_1(x), 1e-2)
        out4 = F.leaky_relu(self.conv4_2(x), 1e-2)

        x = self.max_pool4(out4)
        x = F.leaky_relu(self.conv5_1(x), 1e-2)
        x = F.leaky_relu(self.conv5_2(x), 1e-2)

        x = self.up6(x)
        if self.conv6_1:
            x = self.conv6_1(x)
        # Crop & concat: Start
        # currently hard-coded for testing purpose
        # from 64x64 to 56x56, zero-indexed
        cropped_out4 = out4[:, :, 3:59, 3:59]  # [1,512,56,56]
        x = torch.cat([x, cropped_out4], dim=1)
        # Crop & concat: End
        x = F.leaky_relu(self.conv6_2(x), 1e-2)
        x = F.leaky_relu(self.conv6_3(x), 1e-2)

        x = self.up7(x)
        if self.conv7_1:
            x = self.conv7_1(x)
        # Crop & concat: Start
        # currently hard-coded for testing purpose
        # from 136x136 to 104x104, zero-indexed
        cropped_out3 = out3[:, :, 15:119, 15:119]
        x = torch.cat([x, cropped_out3], dim=1)
        # Crop & concat: End
        x = F.leaky_relu(self.conv7_2(x), 1e-2)
        x = F.leaky_relu(self.conv7_3(x), 1e-2)

        x = self.up8(x)
        if self.conv8_1:
            x = self.conv8_1(x)
        # Crop & concat: Start
        # currently hard-coded for testing purpose
        # from 280x280 to 200x200, zero-indexed
        cropped_out2 = out2[:, :, 39:239, 39:239]
        x = torch.cat([x, cropped_out2], dim=1)
        # Crop & concat: End
        x = F.leaky_relu(self.conv8_2(x), 1e-2)
        x = F.leaky_relu(self.conv8_3(x), 1e-2)

        x = self.up9(x)
        if self.conv9_1:
            x = self.conv9_1(x)
        # Crop & concat: Start
        # currently hard-coded for testing purpose
        # from 568x568 to 392x392, zero-indexed
        cropped_out1 = out1[:, :, 87:479, 87:479]
        x = torch.cat([x, cropped_out1], dim=1)
        # Crop & concat: End
        x = F.leaky_relu(self.conv9_2(x), 1e-2)
        x = F.leaky_relu(self.conv9_3(x), 1e-2)

        # Output
        x = self.conv10_1(x)

        return x  # self.tanh(x)


if __name__ == '__main__':
    model = UNet()
    model.forward(torch.randn(1, 1, 572, 572))
