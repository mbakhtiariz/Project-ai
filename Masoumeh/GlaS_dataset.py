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

## My configuration/path

data_path = '../data/Glas/'
#grade_file = "Grade_s.csv"
grade_file = "Grade.csv"



class GlaSDataset(Dataset):
    """ GlaS Dataset  """

    def __init__(self, csv_file=data_path + grade_file, root_dir=data_path, transform=None, transform_anno=None,
                 desired_dataset=None):
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

        GlaS = self.framework.iloc[index, 2]
        grade = self.framework.iloc[index, 3]

        sample = {'image': image, 'image_anno': image_anno, 'patient_id': patient_id, 'GlaS': GlaS, 'grade': grade}

        # Currently unused, but future-proofing (This will be the supplied preprocessing/data augmentation)
        if self.transform:
            # PIL-image must be HxWxC, thus must have 3 dimensions
            if len(sample['image_anno'].shape) == 2:
                sample['image_anno'] = np.expand_dims(sample['image_anno'], axis=2)
            [h, w, c] = sample['image'].shape
            sample['image'] = transforms.functional.to_pil_image(sample['image'])
            # !!
            #print("------------------", w, h)
            instantiated_transform = self.transform(w, h)
            sample['image'] = instantiated_transform(sample['image'])
        # No need ...
        if self.transform_anno:
            # PIL-image must be HxWxC, thus must have 3 dimensions

            if len(sample['image_anno'].shape) == 2:
                sample['image_anno'] = np.expand_dims(sample['image_anno'], axis=2)
            [h, w, c] = sample['image_anno'].shape
            #print("------------------", w, h)
            sample['image_anno'] = transforms.functional.to_pil_image(sample['image_anno'])
            #!!
            instantiated_transform = self.transform_anno(w, h)
            sample['image_anno'] = instantiated_transform(sample['image_anno'])

        return sample


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
