# Author: Alexander Hustinx
# Date: 12-06-2018
#
# GlaS DataLoader example and transformation example
# Version: v1.1

from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from GlaS_dataset import GlaSDataset
from data_augmentation.HEStain import RandomHEStain
from data_augmentation.binarize import Binarize
from data_augmentation.elastic_deformation import ElasticDeformation
from data_augmentation.flip import Flip
from data_augmentation.gaussian_blur import GaussianBlur
from data_augmentation.normalise import Normalise
from data_augmentation.normalise_rgb import NormaliseRGB
from data_augmentation.pil_image import ToPILImage
from data_augmentation.random_gaussian_blur import RandomGaussianNoise
from data_augmentation.resize import Resize
from data_augmentation.rotation import Rotation
from data_augmentation.tensor import ToTensor
from data_augmentation.transpose_and_sqeeze import TransposeAndSqueeze


def imshow(sample, title=None):
    images_batch = sample['image']
    anno_images_batch = sample['image_anno']

    print('images_batch.shape: ', images_batch.shape)
    print('images_anno_batch.shape: ', anno_images_batch.shape)

    grid = torchvision.utils.make_grid(images_batch, nrow=batch_size)
    grid2 = torchvision.utils.make_grid(anno_images_batch, nrow=batch_size)

    print('grid.shape: ', grid.shape)
    print('grid T . shape: ', grid.numpy().transpose((1, 2, 0)).shape)

    print('grid2.shape: ', grid2.shape)
    print('grid2 T . shape: ', grid2.numpy().transpose((1, 2, 0)).shape)

    # plot image and image_anno
    ax = plt.subplot(2, 1, 1)
    ax.axis('off')
    ax.set_title('Input batch')
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    ax = plt.subplot(2, 1, 2)
    ax.axis('off')
    plt.imshow(grid2.numpy().transpose((1, 2, 0)))
    plt.title('Target segmentations')


# Example for the proof-of-concept:
#   Draws the first 4 images and their segmentation
#   Including their GlaS grade and (Sirinukunwattana et al. 2015) grade
if __name__ == '__main__':

    # For reproducible results
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Example batch size for proof of concept, our network will use batch_size = 1
    batch_size = 4

    # This how you sequence/compose transformations
    transformations = transforms.Compose([
        ToPILImage(),
        Resize((572, 572)),
        Rotation(),
        Flip(),
        ElasticDeformation(displacement=20),
        # GaussianBlur(sigma=[0.5, 0.7, 1, 1.3, 1.5, 1.7]),
        RandomGaussianNoise(),
        RandomHEStain(),
        NormaliseRGB(),
        Binarize(threshold=0.00001),
        ToTensor(),
        # Normalise(),
        # TransposeAndSqueeze()
    ])

    # load train dataset
    GlaS_train_dataset = GlaSDataset(transform=transformations, desired_dataset='train')

    # load test dataset (unused)
    GlaS_test_dataset = GlaSDataset(transform=transformations, desired_dataset='test')

    # create data_loader
    train_loader = DataLoader(GlaS_train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=1)

    # create data_loader (unused)
    test_loader = DataLoader(GlaS_test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=1)

    # loop over the set
    for batch_i, sampled_batch in enumerate(train_loader):
        print("Index #{}:\n\t"
              "Patient id:\t\t{}\n\t"
              "Image size:\t\t{}\n\t"
              "Annotated image size:\t{}\n\t"
              "GlaS grade:\t\t{}\n\t"
              "Other grade:\t\t{}"
              .format(batch_i,
                      sampled_batch['patient_id'],
                      sampled_batch['image'].shape,
                      sampled_batch['image_anno'].shape,
                      sampled_batch['GlaS'],
                      sampled_batch['grade']))

        # Observe the 3rd batch
        if batch_i == 2:
            plt.figure()
            imshow(sampled_batch)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
