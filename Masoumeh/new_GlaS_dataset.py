# Transformations work fine with this file.

# Author: Alexander Hustinx
# Date: 8-06-2018
#
# GlaS Dataset

# few lines edited

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from torch.utils.data.sampler import SubsetRandomSampler
from copy import copy
import random
from PIL import Image, ImageFilter
import cv2
from scipy.interpolate import griddata
## My configuration/path

data_path = '../data/Glas/'
#grade_file = "Grade_s.csv"
grade_file = "Grade.csv"



class GlaSDataset(Dataset):
    """ GlaS Dataset  """

    def __init__(self, csv_file=data_path + grade_file, root_dir=data_path, transform=None, transform_anno=None,
                 desired_dataset=None, flip_prob= 0, rotate_prob = 0, elastic_deform_prob = 0, blur_prob = 0, jitter_prob = 0):
        """
		Arguments:
			csv_file: path to the grade csv-file
			root_dir: path to the map containing the images
			transform: (optional) transformation to be applied on sample['image']
			transform_anno: (optional) transformation to be applied on the sample['image_anno']
			desired_dataset: (optional) rows where the name does not contains this keyword will be deleted
					this allows you to split the dataset into 'train' and 'test'
		"""

        # File extensions *cough* hardcoded *cough*
        self.image_ext = '.bmp'
        self.annotation_label = '_anno'

        self.flip_prob = flip_prob # TODO: add it to param files
        self.rotate_prob = rotate_prob # TODO: add it to param files
        self.elastic_deform_prob = elastic_deform_prob # TODO: add it to param files
        self.blur_prob = blur_prob  # TODO: add it to param files
        self.jitter_prob = jitter_prob # TODO: add it to param files

        # Load csv-file into pandas
        self.framework = pd.read_csv(csv_file)

        # Get rid of those pesky whitespaces at the start and end of the grades
        self.framework[' grade (GlaS)'] = self.framework[' grade (GlaS)'].str.strip()
        self.framework[' grade (Sirinukunwattana et al. 2015)'] = self.framework[
            ' grade (Sirinukunwattana et al. 2015)'].str.strip()

        # Remove all rows not containing the given desired_dataset, allowing to split 'test' and 'train'
        if desired_dataset:
            self.framework = self.framework[self.framework['name'].str.contains(desired_dataset) == True]

        self.root_dir = root_dir
        self.transform = transform
        self.transform_anno = transform_anno

    def __len__(self):
        return len(self.framework)

    def __getitem__(self, index):
        """Sample format:
			image: image containing the to segment/grade cells
			image_anno: image containing the segmented cells
			patient_id: id of the patient the cell originated from
			GlaS: assigned GlaS grade (target #1)
			grade: assigned (Sirinukunwattana et al. 2015) grade (target #2)
		"""

        image_name = self.root_dir + self.framework.iloc[index, 0]
        image = io.imread(image_name + self.image_ext)
        image_anno = io.imread(image_name + self.annotation_label + self.image_ext)

        # Currently unused, but future-proofing
        patient_id = self.framework.iloc[index, 1]

        # !!!!!!!
        GlaS = 1 if self.framework.iloc[index, 2] == 'malignant' else 0
        grade = self.framework.iloc[index, 3]

        sample = {'image': image, 'image_anno': image_anno, 'patient_id': patient_id, 'GlaS': GlaS, 'grade': grade}

        # Currently unused, but future-proofing (This will be the supplied preprocessing/data augmentation)
        if self.transform:
            # PIL-image must be HxWxC, thus must have 3 dimensions
            if len(sample['image_anno'].shape) == 2:
                sample['image_anno'] = np.expand_dims(sample['image_anno'], axis=2)
            if len(sample['image_anno'].shape) == 2:
                sample['image_anno'] = np.expand_dims(sample['image_anno'], axis=2)
            [h, w, c] = sample['image_anno'].shape

            # transformations that are done on both image and labels go here...
            sample['image'], sample['image_anno'] = self.flip_aug(sample['image'], sample['image_anno'])

            sample['image'] = transforms.functional.to_pil_image(sample['image'])
            sample['image_anno'] = transforms.functional.to_pil_image(sample['image_anno'])

            # transform just on original image, since it get PIL as input, I defined it here:
            sample['image'] = self.blur_aug(sample['image'])
            sample['image'] = self.color_jitter(sample['image'])
            sample['image'], sample['image_anno'] = self.rotate_aug(sample['image'], sample['image_anno'])
            sample['image'], sample['image_anno'] = self.elastic_deformation(sample['image'], sample['image_anno'])

            instantiated_transform = self.transform(w, h)
            sample['image'] = instantiated_transform(sample['image'])

            instantiated_transform = self.transform_anno(w, h)
            sample['image_anno'] = instantiated_transform(sample['image_anno'])

        return sample

    #-------------- New helper func ------------------------
    # https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606
    # https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7
    def flip_aug(self, image, label):
        if random.random() < self.flip_prob:
            return image[::-1, :, :], label[::-1, :, :]
        else:
            return image, label

    def rotate_aug(self, image, label):
        if random.random() > self.rotate_prob:
            return image, label
        angle = random.choice([random.randint(-20,20),90,180,270])
        return image.rotate(angle), label.rotate(angle)

    def elastic_deformation(self, image, label):

        if random.random() > self.elastic_deform_prob:
            return image, label
        self._grid_size = 5
        self._displacement = 20

        width, height = image.size

        width_span = width / (self._grid_size * 2)
        height_span = height / (self._grid_size * 2)

        same_horizontal_border = np.array(np.meshgrid([0, height], np.arange(0, width, 1))).T.reshape(-1, 2)
        same_vertical_border = np.array(np.meshgrid(np.arange(0, height, 1), [0, width])).T.reshape(-1, 2)
        same_border = np.concatenate((same_horizontal_border, same_vertical_border), axis=0)

        displacement_point_y = np.arange(width_span, (2 * self._grid_size - 1) * width_span + 1, 2 * width_span)
        displacement_point_x = np.arange(height_span, (2 * self._grid_size - 1) * height_span + 1, 2 * height_span)
        source_points = np.array(np.meshgrid(displacement_point_y, displacement_point_x)).T.reshape(-1, 2)
        displaced_points = source_points + np.random.uniform(-self._displacement, self._displacement,
                                                             source_points.shape)

        source_points = np.concatenate((source_points, same_border), axis=0)
        displaced_points = np.concatenate((displaced_points, same_border), axis=0)

        grid_x, grid_y = np.mgrid[0:height - 1:1j * height, 0:width - 1:1j * width]

        grid_z = griddata(displaced_points, source_points, (grid_x, grid_y), method='cubic')
        map_x_32 = np.append([], [ar[:, 1] for ar in grid_z]).reshape(height, width).astype('float32')
        map_y_32 = np.append([], [ar[:, 0] for ar in grid_z]).reshape(height, width).astype('float32')

        return Image.fromarray(cv2.remap(np.array(image), map_x_32, map_y_32, cv2.INTER_CUBIC)), Image.fromarray(cv2.remap(np.array(label), map_x_32, map_y_32, cv2.INTER_CUBIC))

    def blur_aug(self, image):
        radius = 2
        if random.random() < self.blur_prob:
            return image.filter(ImageFilter.GaussianBlur(radius))
        else:
            return image

    def color_jitter(self,image):
        if random.random() < self.jitter_prob:
            # TODO: decide over how to choose these values
            brightness = random.uniform(0,1)
            contrast = random.uniform(0,1)
            return transforms.ColorJitter(brightness, contrast)(image)
        else:
            return image


## Example for the proof-of-concept:
## 		Draws the first 4 images and their segmentations
##		Including their GlaS grade and (Sirinukunwattana et al. 2015) grade
if __name__ == '__main__':

    # load dataset
    fig = plt.figure()
    dataset = GlaSDataset(desired_dataset='test')

    for i in range(len(dataset)):
        # load a sample
        sample = dataset[i]

        print(
            "Index #{}:\n\tPatient id:\t\t{}\n\tImage size:\t\t{}\n\tAnnotated image size:\t{}\n\tGlaS grade:\t\t{}\n\tOther grade:\t\t{}"
                .format(i, sample['patient_id'], sample['image'].shape, sample['image_anno'].shape, sample['GlaS'],
                        sample['grade']))

        ##plots: start
        ax = plt.subplot(2, 4, i + 1)
        plt.tight_layout()
        ax.axis('off')
        ax.set_title('Sample #{}'.format(i))
        plt.imshow(sample['image'])

        ax = plt.subplot(2, 4, i + 5)
        plt.tight_layout()
        ax.axis('off')
        plt.imshow(sample['image_anno'])
        ##plots: end

        # we only show 3, proof-of-concept
        if i == 3:
            plt.show()
            break
