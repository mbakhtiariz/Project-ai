# For any kind of augmentation please change the prob values in hyper_params file
# 0 stand for no transformation
# 1 stand for absolute transformation

# TODO: seperate file of hyper params and augmentation params

from __future__ import print_function, division

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from new_GlaS_dataset import GlaSDataset
from other.hyperparam import load_hyperparams

matplotlib.use('GTKAgg')

# Example of a transformation that can be executed on the sample image
class Binarize(object):
    """
        Binarize the image given a certain threshold

    Args:
        threshold (float or int): Desired threshold to binarize the image with

    """

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        img = img > self.threshold
        return img.float()


def compute_padding(w, h, target_w, target_h):
    x_pad = (target_w - w) // 2
    y_pad = (target_h - h) // 2
    # print ("+++++++++", x_pad, y_pad)
    return (x_pad, y_pad, x_pad + (target_w - w) % 2, y_pad + (target_h - h) % 2)

def find_image_mask_new_size(hyper_params):
    img_w = hyper_params["img_w"]
    img_h = hyper_params["img_h"]

    # For x > 4 (x represents the size of feature map at the end of the down sampling layer)
    # Based on x, for d-layer network input size should be 2^(d-1)x + 4(2^(d)-1)
    # The margin between input & output 12*2^(d-1) - 8
    d = hyper_params["depth"]

    a = 2**(d - 1)
    b = 12 * (2**(d - 1)) - 4*(2**d) + 4

    # Find new mask size in a way to be slightly larger than original image

    mask_new_w = a * math.ceil(float(img_w + b) / a) - b
    mask_new_h = a * math.ceil(float(img_h + b) / a) - b

    diff = 12 * (2**(d - 1)) - 8
    img_new_w =  mask_new_w + diff
    img_new_h = mask_new_h + diff

    print(img_new_w, img_new_h, mask_new_w, mask_new_h)
    return img_new_w, img_new_h, mask_new_w, mask_new_h

if __name__ == '__main__':

    # Load cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # For reproducable results
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)


    # Load PARAMS:
    hyper_params = load_hyperparams("hyper_params")

    epochs = hyper_params["epochs"]
    batch_size = hyper_params["batchSize"]
    channels = hyper_params["channels"]
    learning_rate = hyper_params["lr"]
    lambda2 = hyper_params["lambda2"]


    #---------- Multi loss param ---------------
    cls_alpha = hyper_params["cls_alpha"]
    #---------- Validation split params----------
    valid_size = hyper_params["valid_size"]
    shuffle = hyper_params["shuffle"]
    #If using CUDA, num_workers should be set to 1 and pin_memory to True.
    pin_memory = hyper_params["pin_memory"] #False: cpu, True:cuda
    num_workers = hyper_params["num_workers"] # 0 if cpu 1 or more if cuda
    #---------- Early Stopping Param------------
    tolerance = hyper_params["tolerance"]
    #-------------------------------------------


    #------------- prob of each inner transformation ----------------
    flip_prob = hyper_params["flip_prob"]
    rotate_prob = hyper_params["rotate_prob"]
    elastic_deform_prob = hyper_params["elastic_deform_prob"]
    blur_prob = hyper_params["blur_prob"]
    jitter_prob = hyper_params["jitter_prob"]


    # We want to resize all input images to the same size and naturally have to change size of masks as well.
    # Depth of network is an hyper param and can be changed so for achieving this we calculate the new size based on depth of network
    img_new_w, img_new_h, mask_new_w, mask_new_h = find_image_mask_new_size(hyper_params) # for depth 3: 816, 564 / 776, 524

    #transforms.ColorJitter(brightness= brightness, contrast = brightness),
    # This how you sequence/compose transformations
    data_transform = lambda w, h: \
        transforms.Compose([transforms.Pad(padding=compute_padding(w, h, img_new_w, img_new_h), padding_mode='reflect'),
                            transforms.ToTensor()])

    # This is how you add onto an existing sequence/composition
    anno_transform = lambda w, h: \
        transforms.Compose([transforms.Pad(padding=compute_padding(w, h, mask_new_w, mask_new_h ), padding_mode='reflect'),
                            transforms.ToTensor(),
                            Binarize(threshold=0.001)])

    # Load train dataset
    GlaS_train_dataset = GlaSDataset(transform=data_transform,
                                     transform_anno=anno_transform,
                                     desired_dataset='train',
                                     flip_prob = flip_prob, rotate_prob = rotate_prob,
                                     elastic_deform_prob = elastic_deform_prob, blur_prob = blur_prob, jitter_prob = jitter_prob)

    GlaS_original_dataset = GlaSDataset(desired_dataset='train')

    # create data_loader
    train_loader = DataLoader(GlaS_train_dataset,
                              batch_size=batch_size,
                              shuffle= False,
                              num_workers=num_workers)

    orig_loader = DataLoader(GlaS_original_dataset,
                              batch_size=batch_size,
                              shuffle= False,
                              num_workers=num_workers)




    orig_sample = GlaS_original_dataset[1]

    trns = GlaS_train_dataset[1]

    print(trns['image'].numpy().shape, "----------------------")
    print(trns['image_anno'].numpy().shape, "----------------------")
    transformed_sample = trns['image'].numpy().transpose((1, 2, 0))
    transformed_anno = np.squeeze(trns['image_anno'].numpy().transpose((1, 2, 0)), axis=2)

    print(transformed_anno.shape)
    plt.figure()
    plt.imshow(orig_sample['image'])
    plt.title('orig_sample')

    plt.figure()
    plt.imshow(orig_sample['image_anno'])
    plt.title('orig_annotation')

    plt.figure()
    plt.imshow(transformed_sample)
    plt.title('transformed_sample')

    plt.figure()
    plt.imshow(transformed_anno)
    plt.title('transformed_anno')

    plt.show()

