# U-net: just seg
# Project AI


import torch
print(torch.__version__)

import torch
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import scipy.ndimage

from hyperparam import load_hyperparams


class UNet(nn.Module):
    def __init__(self, hyper_params, channels = 3):
        super(UNet, self).__init__()
        
        self.hyper_params = hyper_params
        
        # later will change these to loop over all options for every epoch
        self.network_depth = int(hyper_params['depth'][0])
        filter_num = int(hyper_params['filterNumStart'][0])
        in_channels = int(channels)
        doBatchNorm = int(hyper_params['doBatchNorm'][0])
        
        print("----------- Building Encoder -------------")
        self.down_blocks = []
        self.up_blocks = []
        print(in_channels, filter_num)
        for d in range(self.network_depth):
            block_d = {}

            if doBatchNorm == 1:
                block_d['conv1'] = nn.Sequential(nn.Conv2d(in_channels, filter_num, kernel_size = 3, stride=1, padding=0), nn.BatchNorm2d(filter_num), nn.ReLU())
            else:
                block_d['conv1'] = nn.Sequential(nn.Conv2d(in_channels, filter_num, kernel_size=3, stride=1, padding=0), nn.ReLU())
            print(in_channels, filter_num)
            in_channels = filter_num

            if doBatchNorm == 1:
                block_d['conv2'] = nn.Sequential(nn.Conv2d(in_channels, filter_num, kernel_size = 3, stride=1, padding=0), nn.BatchNorm2d(filter_num), nn.ReLU())
            else:
                block_d['conv2'] = nn.Sequential(nn.Conv2d(in_channels, filter_num, kernel_size=3, stride=1, padding=0),nn.ReLU())

            print(in_channels, filter_num)
            if(d != self.network_depth-1):
                block_d['maxpool'] = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                filter_num *= 2
            self.down_blocks.append(block_d)

        #self.fc = nn.Linear(1*1*filter_num,3)

           
        print("----------- Building Decoder -------------")    
        in_channels = filter_num
        for d in range(int(self.network_depth - 1)):
            block_u = {}
            filter_num = int(in_channels / 2)
            print(in_channels, filter_num)

            if doBatchNorm == 1:
                block_u['upconv'] = nn.Sequential(nn.ConvTranspose2d(in_channels, filter_num, kernel_size=2, stride=2), nn.BatchNorm2d(filter_num), nn.ReLU())
            else:
                block_u['upconv'] = nn.Sequential(nn.ConvTranspose2d(in_channels, filter_num, kernel_size=2, stride=2),nn.ReLU())
            # torch.nn.functional.leaky_relu_(input, negative_slope=0.01)
            
            print(in_channels, filter_num)

            if doBatchNorm == 1:
                block_u['conv1'] = nn.Sequential(nn.Conv2d(in_channels, filter_num, kernel_size = 3, stride=1, padding=0), nn.BatchNorm2d(filter_num), nn.ReLU())
            else:
                block_u['conv1'] = nn.Sequential(nn.Conv2d(in_channels, filter_num, kernel_size=3, stride=1, padding=0), nn.ReLU())
            
            in_channels = int(in_channels / 2)
            print(in_channels, filter_num)

            if doBatchNorm == 1:
                block_u['conv2'] = nn.Sequential(nn.Conv2d(in_channels, filter_num, kernel_size = 3, stride=1, padding=0), nn.BatchNorm2d(filter_num), nn.ReLU())
            else:
                block_u['conv2'] = nn.Sequential(nn.Conv2d(in_channels, filter_num, kernel_size=3, stride=1, padding=0), nn.ReLU())

            self.up_blocks.append(block_u)

        self.seg_head = nn.Conv2d(filter_num, 2, kernel_size = 1, stride=1, padding=0)

    
    @staticmethod
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n)) 
                    
    def forward(self,x):
        
        # now create the down-sampling path
        
        concat_features = []

        for d in range(self.network_depth):
            x = self.down_blocks[d]['conv1'](x)
            x = self.down_blocks[d]['conv2'](x)
            concat_features.append(x)
            if d != self.network_depth - 1:
                x = self.down_blocks[d]['maxpool'](x)

        #y = F.adaptive_avg_pool2d(x, (1, 1))
        #cls_head = self.fc(y)
        # now create the up-sampling path
        for d in range(self.network_depth-1):
            x = self.up_blocks[d]['upconv'](x)
            
            x = self.crop_and_concat(x, concat_features[self.network_depth-2-d], crop=True)

            x = self.up_blocks[d]['conv1'](x)
            x = self.up_blocks[d]['conv2'](x)
        
        return self.seg_head(x)#, cls_head
    
    # from https://discuss.pytorch.org/t/cropping-images-in-a-batch-on-the-gpu/7485/2
    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            diff1 = bypass.size()[2] - upsampled.size()[2]
            diff2 = bypass.size()[3] - upsampled.size()[3]
            x_c = (diff1) // 2  # assumes equal width/height
            y_c = (diff2) // 2  # assumes equal width/height

            bypass = F.pad(bypass, (- y_c, - (diff2 - y_c), -x_c, -(diff1 - x_c)))

        return torch.cat((upsampled, bypass), 1)
        
            
            
if __name__ == '__main__':

    # Just for testing Unet:

    hyper_params = load_hyperparams("hyper_params")
    
    print(len(hyper_params['filterNumStart']))

    net = UNet(hyper_params, channels = 3)
    x = Variable(torch.FloatTensor(np.random.random((1,3, 572, 572))))
    seg_out = net(x)


