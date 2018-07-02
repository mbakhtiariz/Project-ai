import torch
import torch.nn.functional as F
from torchvision import utils, transforms
from torch.utils.data import DataLoader
from UNet import UNet
from data_augmentation.binarize import Binarize, Binarize_Output
from data_augmentation.center_crop import CenterCrop
from data_augmentation.grayscale import Grayscale
from data_augmentation.normalise import Normalise
from data_augmentation.pil_image import ToPILImage
from data_augmentation.normalise_rgb import NormaliseRGB
from data_augmentation.tensor import ToTensor
from  GlaS_dataset import GlaSDataset
from UNet_test import jaccard_loss
import pickle
import os
import sys
import numpy as np
from collections import Counter

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    '''
    For saving weight maps of test files.
    '''
    print("start")
    result_path = "final_results"
    test_output_path = result_path + '/test_info'
    if not os.path.exists(test_output_path):
        os.makedirs(test_output_path)

    batch_size = 1 

    # List of data augmentations to be applied on the data
    test_transformations = transforms.Compose([
        ToPILImage(),
        NormaliseRGB(),
        CenterCrop(image_crop=(572, 572), mask_crop=(388, 388)),
        Grayscale(),
        Binarize(threshold=0.000001),   
        ToTensor(),
        Normalise(),
        # TransposeAndSqueeze()
    ])

    # load test dataset (unused)
    GlaS_test_dataset = GlaSDataset(transform=test_transformations,
                                    desired_dataset='test',
                                    data_expansion_factor=1)

    # create test data_loader (unused)
    test_loader = DataLoader(GlaS_test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=1)

    for batch_i, sample in enumerate(test_loader):
        # Loading every image and its annotation:
        data, mask, loss_weight,img_name = sample['image'], sample['image_anno'], sample['loss_weight'], sample['name'][0]

        loss_weight = loss_weight / 1000.0

        utils.save_image(data, "{}/{}_test_input.png".format(test_output_path, img_name))
        utils.save_image(mask, "{}/{}_test_target.png".format(test_output_path, img_name))
        utils.save_image(loss_weight, "{}/{}_test_weights.png".format(test_output_path, img_name), normalize=True)

