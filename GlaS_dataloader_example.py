# Author: Alexander Hustinx
# Date: 12-06-2018
#
# GlaS DataLoader example and transformation example
# Version: v1.0

from __future__ import print_function, division
import os

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform

from GlaS_dataset import GlaSDataset

class Resize(object):
	"""Resize the image in a sample to a given size.

	Args:
		output_size (tuple or int): Desired output size. If tuple, output is
			matched to output_size. If int, smaller of image edges is matched
			to output_size keeping aspect ratio the same.
	"""

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size=output_size

	def __call__(self, sample):
		print('pre [Rescale] sample[\'image\'].shape: ', sample['image'].shape)
		image = sample['image']
		new_h, new_w = self.output_size
		new_h, new_w = int(new_h), int(new_w)
		img = transform.resize(image, (new_h, new_w))
		sample['image'] = img
		
		anno_image = sample['image_anno']
		new_h, new_w = self.output_size
		new_h, new_w = int(new_h), int(new_w)
		img = transform.resize(anno_image, (new_h, new_w))
		sample['image_anno'] = img
		
		print('post [Rescale] sample[\'image\'].shape: ', sample['image'].shape)
		
		return sample
		
class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		image, image_anno = sample['image'], sample['image_anno']

		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		sample['image'] = image.transpose((2, 0, 1))
		sample['image_anno'] = image.transpose((2, 0, 1))
		
		return sample

		
def imshow(input, title=None):	
	images_batch = input['image']
	
	print('images_batch.shape: ', images_batch.shape)
	
	grid = torchvision.utils.make_grid(images_batch)
	
	print('grid.shape: ', grid.shape)
	print('grid T . shape: ', grid.numpy().transpose((1, 2, 0)).shape)
	
	plt.imshow(grid.numpy().transpose((1,2,0)))
	plt.title('batch from dataloader')
		
## Example for the proof-of-concept:
## 		Draws the first 4 images and their segmentations
##		Including their GlaS grade and (Sirinukunwattana et al. 2015) grade
if __name__ == '__main__':
	batch_size = 4

	#load train dataset
	GlaS_train_dataset = GlaSDataset(transform=transforms.Compose([Resize((256,256)),ToTensor()]), 
								desired_dataset='train')

	#load test dataset
	GlaS_test_dataset = GlaSDataset(transform=Resize((256,256)),
								desired_dataset='test')
	
	train_loader = DataLoader(GlaS_train_dataset, 
							batch_size=batch_size,
							shuffle=True,
							num_workers=1)
							
	test_loader = DataLoader(GlaS_test_dataset,
							batch_size=batch_size,
							shuffle=False,
							num_workers=1)
							
	#loop over the set
	for batch_i, sampled_batch in enumerate(train_loader):
		print("Index #{}:\n\tPatient id:\t\t{}\n\tImage size:\t\t{}\n\tAnnotated image size:\t{}\n\tGlaS grade:\t\t{}\n\tOther grade:\t\t{}"
			.format(batch_i, sampled_batch['patient_id'], sampled_batch['image'].shape, sampled_batch['image_anno'].shape, sampled_batch['GlaS'], sampled_batch['grade']))
		
		
		
		#Observe the 3rd batch
		if batch_i == 2:
			plt.figure()
			imshow(sampled_batch)
			plt.axis('off')
			plt.ioff()
			plt.show()
			##plots: end
			
			break
	