# Author: Alexander Hustinx
# Date: 8-06-2018
#
# GlaS Dataset
# Version: v0.1

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

## My configuration/path
data_path = "data\\GlaS\\"
grade_file = "Grade.csv"

class GlaSDataset(Dataset):
	""" GlaS Dataset  """
	def __init__(self, csv_file=data_path+grade_file, root_dir=data_path, transform=None, desired_dataset=None):
		"""
		Arguments:
			csv_file: path to the grade csv-file
			root_dir: path to the map containing the images
			transform: (optional) transformation to be applied on the sample
			desired_dataset: (optional) rows where the name does not contains this keyword will be deleted
					this allows you to split the dataset into 'train' and 'test'
		"""
		
		# File extensions *cough* hardcoded *cough*
		self.image_ext = '.bmp'
		self.annotation_label = '_anno'
		
		#Load csv-file into pandas
		self.framework = pd.read_csv(csv_file)
		
		#Get rid of those pesky whitespaces at the start and end of the grades
		self.framework[' grade (GlaS)'] = self.framework[' grade (GlaS)'].str.strip()
		self.framework[' grade (Sirinukunwattana et al. 2015)'] = self.framework[' grade (Sirinukunwattana et al. 2015)'].str.strip()
		
		#Remove all rows not containing the given desired_dataset, allowing to split 'test' and 'train'
		if desired_dataset:
			self.framework = self.framework[self.framework['name'].str.contains(desired_dataset) == True]
		
		self.root_dir = root_dir
		self.transform = transform	
		
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
		
		#Currently unused, but future-proofing
		patient_id = self.framework.iloc[index, 1]
		
		GlaS = self.framework.iloc[index, 2]
		grade = self.framework.iloc[index, 3]
		
		sample = {'image':image, 'image_anno':image_anno, 'patient_id':patient_id, 'GlaS':GlaS, 'grade':grade}
		
		#Currently unused, but future-proofing (This will be the supplied preprocessing/data augmentation)
		if self.transform:
			#sample['image'] = self.transform(sample['image'])
			sample = self.transform(sample)
			
		return sample
		
		
## Example for the proof-of-concept:
## 		Draws the first 4 images and their segmentations
##		Including their GlaS grade and (Sirinukunwattana et al. 2015) grade
if __name__ == '__main__':
	
	#load dataset
	dataset = GlaSDataset(desired_dataset='test')
	fig = plt.figure()
	
	for i in range(len(dataset)):
		#load a sample
		sample = dataset[i]
		
		print("Index #{}:\n\tPatient id:\t\t{}\n\tImage size:\t\t{}\n\tAnnotated image size:\t{}\n\tGlaS grade:\t\t{}\n\tOther grade:\t\t{}"
			.format(i, sample['patient_id'], sample['image'].shape, sample['image_anno'].shape, sample['GlaS'], sample['grade']))
		
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
		
		#we only show 3, proof-of-concept
		if i == 3:
			plt.show()
			break