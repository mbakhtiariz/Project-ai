# Author: Alexander Hustinx
# Date: 12-06-2018
#
# GlaS DataLoader example and transformation example
# Version: v1.1

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
	
	data_transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
	
	#load train dataset
	GlaS_train_dataset = GlaSDataset(transform=data_transform,
								transform_anno=data_transform, 
								desired_dataset='train')

	#load test dataset
	GlaS_test_dataset = GlaSDataset(transform=data_transform,
								transform_anno=data_transform,
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
	