
from __future__ import print_function, division


import torch
print(torch.__version__)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms, utils, datasets

import numpy as np
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import scipy.ndimage

from GlaS_dataset import GlaSDataset
from masUNet import UNet

#.........................................
def pad_image(img, new_x, new_y,img_type):
    new_img = []
    if img_type == 'anno':
        x, y = img.shape
        x_pad = int((new_x - x) / 2)
        y_pad = int((new_y - y) / 2)
        new_img = np.lib.pad(img[:, :], ((x_pad, new_x - x_pad - x), (y_pad, new_y - y_pad - y)), 'reflect')
    else:
            x, y, z = img.shape
            x_pad = int((new_x - x) / 2)
            y_pad = int((new_y - y) / 2)
            padded_chanel = {}
            for ch in range(z):
                padded_chanel[ch] = np.lib.pad(img[:, :, ch],((x_pad, new_x - x_pad - x), (y_pad, new_y - y_pad - y)), 'reflect')
            new_img = np.stack((padded_chanel[0], padded_chanel[1], padded_chanel[2]))


    return new_img

def update_cell(img, new_dataset, i,img_type):

    if img_type == 'anno':
        # later make this dynamic:
        new_x = 532
        new_y = 788
        new_img = pad_image(img, new_x, new_y, img_type)
        new_dataset[i]['image_anno'] = new_img
    else:
        # later make this dynamic:
        new_x = 712
        new_y = 968
        new_img = pad_image(img, new_x, new_y, img_type)
        new_dataset[i]['image'] = new_img

    #print(new_dataset[5]['image'].shape)

def make_all_sizes_unique(dataset):
    new_dataset = copy.copy(dataset)
    for i in range(len(new_dataset)):
        sample = new_dataset[i]
        img = sample['image']
        img_anno = sample['image_anno']
        update_cell(img,new_dataset, i,"original")
        update_cell(img_anno, new_dataset, i,"anno")
    #print(new_dataset[5]['image'].shape)
    return new_dataset

#.........................................
def make_container(dataset):
    '''
    GlaSDataset reads every time from file for every sample. In this function we read all samples and save all results.
    dataset:param: GlasDataset
    dataset_listdict:return: list of dictionary containing all samples
    '''
    dataset_listdict = []
    for i in range(len(dataset)):
        sample = dataset[i]
        dataset_listdict.append(sample)
    return dataset_listdict
#.........................................
def histog(dataset):
    hist = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        x, y, z = sample['image'].shape
        #x, y = sample['image_anno'].shape
        if (x, y) not in hist:
            hist[(x, y)] = 1
        else:
            hist[(x, y)] += 1
    return hist


def sample_plot(i, original, mirrorPad):
    fig = plt.figure()

    ax = plt.subplot(2, 2, 1)
    ax.set_title('1')
    plt.imshow(mirrorPad[i]['image'])

    ax = plt.subplot(2, 2, 2)
    ax.set_title('2')
    plt.imshow(original[i]['image'])

    ax = plt.subplot(2, 2, 3)
    ax.set_title('3')
    plt.imshow(mirrorPad[i]['image_anno'])

    ax = plt.subplot(2, 2, 4)
    ax.set_title('4')
    plt.imshow(test_dataset[i]['image_anno'])
    plt.show()

#------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':


    # Load and preprocess data-set:
    test = GlaSDataset(desired_dataset='test')
    train = GlaSDataset(desired_dataset='train')


    # Get statistics:
    print(histog(train))
    print(histog(test))

'''
    # Not useful any more
    # Save all data in container to avoid loading from files each time:
    test_container = make_container(test)
    train_container = make_container(train)

    # Resize all images to the same size:
    test_dataset = make_all_sizes_unique(test_container)
    train_dataset = make_all_sizes_unique(train_container)

    # Plot resizing output:
    #sample_plot(20, test, test_dataset)
'''