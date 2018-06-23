# Author: Alexander Hustinx
# Date: 8-06-2018
#
# GlaS Dataset
# Version: v0.1

from __future__ import print_function, division

import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
from scipy import ndimage
from copy import copy
# My configuration/path
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

from PIL import Image

data_path = "../data/Glas/"
grade_file = "Grade.csv"


class GlaSDataset(Dataset):
    """ GlaS Dataset  """

    def __init__(self, csv_file=data_path + grade_file, root_dir=data_path, transform=lambda x: x,
                 desired_dataset=None, data_expansion_factor: int=1):
        """
        Arguments:
            csv_file: path to the grade csv-file
            root_dir: path to the map containing the images
            transform: (optional) transformation to be applied on sample['image'] and sample['image_anno']
            desired_dataset: (optional) rows where the name does not contains this keyword will be deleted
                    this allows you to split the dataset into 'train' and 'test'
        """

        self.data_expansion_factor = data_expansion_factor

        # File extensions *cough* hardcoded *cough*
        self.image_ext = '.bmp'
        self.annotation_label = '_anno'

        # Load csv-file into pandas
        self.framework = pd.read_csv(csv_file)

        # Get rid of those pesky whitespaces at the start and end of the grades
        self.framework[' grade (GlaS)'] = self.framework[' grade (GlaS)'].str.strip()
        self.framework[' grade (Sirinukunwattana et al. 2015)'] = self.framework[
            ' grade (Sirinukunwattana et al. 2015)'].str.strip()

        # Remove all rows not containing the given desired_dataset, allowing to split 'test' and 'train'
        if desired_dataset:
            self.framework = self.framework[self.framework['name'].str.contains(desired_dataset)]

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # artificially "clone" the dataset to get more images.
        return self.data_expansion_factor * len(self.framework)

    def __getitem__(self, index):
        """Sample format:
            image (torch.Tensor): image containing the to segment/grade cells
            image_anno (torch.Tensor): image containing the segmented cells
            patient_id: id of the patient the cell originated from
            GlaS: assigned GlaS grade (target #1)
            grade: assigned (Sirinukunwattana et al. 2015) grade (target #2)
        """
        # because if artificially 'cloned' the dataset, we need the below line to get the real index
        index = index // self.data_expansion_factor

        image_name = self.root_dir + self.framework.iloc[index, 0]
        image = io.imread(image_name + self.image_ext)
        image_anno = io.imread(image_name + self.annotation_label + self.image_ext)

        # Currently unused, but future-proofing
        patient_id = self.framework.iloc[index, 1]
        GlaS = self.framework.iloc[index, 2]
        grade = self.framework.iloc[index, 3]


        # Calculating Weight Map for Loss:
        sig = 5
        w_0 = 10
        glands = np.unique(image_anno) # should include zero for background, [0,1,2,...]
        gland_num = len(glands) - 1  # -1 for removing background from calculation
        if len(glands) == 1:
            if 0 in glands:# If all the image is just backgound and no gland:
                weight_map = np.zeros((1, image_anno.shape[0], image_anno.shape[1]))
            elif not(0 in glands):# If all the image is just 1 gland and there is no background:
                weight_map = 1 + w_0 * np.ones((1, image_anno.shape[0], image_anno.shape[1]))
        else:
            w = np.inf * np.ones((gland_num, image_anno.shape[0], image_anno.shape[1]))
            for g  in range(0, gland_num):
                filtered_anno = (~(image_anno == g + 1)).astype(int)# Filter out all glands except g th one.
                w[g] = ndimage.distance_transform_edt(filtered_anno)# Calculate the distance to nearest point in g th gland for every pixel:
            dist = -(np.sum(np.sort(w, axis=0)[0:min(2, gland_num)], axis=0) ** 2) / (2 * sig ** 2)# Find 2 minimum distances for every pixel:
            weight_map = 1 + w_0 * np.exp(dist)
        weight_map = 1000 * weight_map
        image, image_anno, weight_map = self.transform((image, image_anno,weight_map.astype(np.float32)))

        return {'image': image, 'image_anno': image_anno, 'loss_weight': weight_map, 'patient_id': patient_id, 'GlaS': GlaS,
                'grade': grade}


# Example for the proof-of-concept:
#   Draws the first 4 images and their segmentation
#   Including their GlaS grade and (Sirinukunwattana et al. 2015) grade
if __name__ == '__main__':

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
        TransposeAndSqueeze()
    ])

    # load dataset
    dataset = GlaSDataset(desired_dataset='test', transform=transformations)

    fig = plt.figure()

    for i in range(len(dataset)):
        # load a sample
        sample = dataset[i]

        print("Index #{}:\n\t"
              "Patient id:\t\t{}\n\t"
              "Image size:\t\t{}\n\t"
              "Annotated image size:\t{}\n\t"
              "GlaS grade:\t\t{}\n\t"
              "Other grade:\t\t{}"
              .format(i,
                      sample['patient_id'],
                      sample['image'].shape,
                      sample['image_anno'].shape,
                      sample['GlaS'],
                      sample['grade']))

        # plots: start
        ax = plt.subplot(2, 4, i + 1)
        plt.tight_layout()
        ax.axis('off')
        ax.set_title('Sample #{}'.format(i))
        plt.imshow(sample['image'])

        ax = plt.subplot(2, 4, i + 5)
        plt.tight_layout()
        ax.axis('off')
        plt.imshow(sample['image_anno'])
        # plots: end

        # we only show batch 3, proof-of-concept
        if i == 3:
            plt.show()
            break
