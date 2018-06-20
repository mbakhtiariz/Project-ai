import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import utils, transforms

from UNet import UNet
from GlaS_dataloader_example import BinarizeExample
from GlaS_dataset import GlaSDataset

import cv2

import matplotlib.pyplot as plt
import time
import random

def jaccard_loss(input, target):
	eps = 1e-15
	intersection = (F.sigmoid(input) * target).sum()
	union = F.sigmoid(input).sum() + target.sum()
	return -torch.log((intersection+eps) / (union-intersection+eps))

#from https://github.com/pytorch/pytorch/issues/751
def stable_bce_loss(input, target):
	neg_abs = - input.abs()
	loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
	return loss.mean()

def prep_img(img):
	#PIL-image must be HxWxC, thus must have 3 dimensions
	if len(img.shape) == 2:
		img = np.expand_dims(img, axis=2)
	if len(img.shape) == 4:
		img = torch.squeeze(img, 0)
	img = transforms.functional.to_pil_image(img)
	return img

def train(model, device, train_loader, optimizer, epoch, log_interval=10):
	model.train()
	for batch_i, sample in enumerate(train_loader):
		data, target = sample['image'], sample['image_anno']
		augment = random.choice(data_augmentation)
		if augment:
			data, target = augment(prep_img(data)), augment(prep_img(target))
			data, target = transforms.ToTensor()(data), transforms.ToTensor()(target)
			data, target = torch.unsqueeze(data, 0), torch.unsqueeze(target, 0)

		#target = torch.squeeze(target, dim=0)
		data, target = data.to(device), target.to(device)		
		
		optimizer.zero_grad()
		output = model(data)
		
		loss = F.mse_loss(output, target)
		#loss = stable_bce_loss(output, target)
		#loss = F.binary_cross_entropy(output, target)	#Needs LongTensor, given FloatTensor
		#loss = F.l1_loss(output, target)		#Horribly high loss
		#loss = F.nll_loss(torch.exp(output), target)		#Needs LongTensor, given FloatTensor
		#loss = F.binary_cross_entropy_with_logits(output, target)
		#loss = jaccard_loss(output, target)
		loss.backward()
		optimizer.step()
		
		if batch_i % log_interval == 0:
			print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
						epoch, batch_i*len(data), len(train_loader.dataset),
						100. * batch_i / len(train_loader), loss.item()))
						
		if batch_i == 164:
			post_transform = transforms.Compose([BinarizeExample(threshold=output.mean())])
			thres = post_transform(output)
			#hist_eq = torch.histc(output.to(torch.device("cpu")))
			utils.save_image(data, "input_{}.bmp".format(epoch))
			utils.save_image(target, "target_{}.bmp".format(epoch))
			
			utils.save_image(output, "output_{}.bmp".format(epoch))
			utils.save_image(thres, "thres_{}.bmp".format(epoch))

###### Transfroms #######
class RandomHEStain(object):
	"""Transfer the given PIL.Image from rgb to HE, perturbate, transfer back to rgb """
	
	def _call_(self, img):
		img_he = skimage.color.rgb2hed(img)
		img_he[:, :, 0] = img_he[:, :, 0] * random.normal(1.0, 0.02, 1)  # H
		img_he[:, :, 1] = img_he[:, :, 1] * random.normal(1.0, 0.02, 1)  # E
		img_rgb = np.clip(skimage.color.hed2rgb(img_he), 0, 1)
		img = Image.fromarray(np.uint8(img_rgb*255.999), img.mode)
		return img
	
	
class Normalize_AMC(object):
	
	#def __init__():
		
	
	# input: np.array or PIL-Image
	# output: PIL-Image
	def __call__(self, image, target=None):
		"""Normalizing function we got from the cedars-sinai medical center"""
		
		#Expects np array, so cast to np 
		if type(image) is not np.ndarray:
			image = np.array(image)
		
		if target is None:
			target = np.array([[148.60, 41.56], [169.30, 9.01], [105.97, 6.67]])

		M, N = image.shape[:2]

		whitemask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		whitemask = whitemask > 215 ## TODO: Hard code threshold; replace with Otsu

		imagelab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

		imageL, imageA, imageB = cv2.split(imagelab)

		# mask is valid when true
		imageLM = np.ma.MaskedArray(imageL, whitemask)
		imageAM = np.ma.MaskedArray(imageA, whitemask)
		imageBM = np.ma.MaskedArray(imageB, whitemask)

		## Sometimes STD is near 0, or 0; add epsilon to avoid div by 0 -NI
		epsilon = 1e-11

		imageLMean = imageLM.mean()
		imageLSTD = imageLM.std() + epsilon

		imageAMean = imageAM.mean()
		imageASTD = imageAM.std() + epsilon

		imageBMean = imageBM.mean()
		imageBSTD = imageBM.std() + epsilon

		# normalization in lab
		imageL = (imageL - imageLMean) / imageLSTD * target[0][1] + target[0][0]
		imageA = (imageA - imageAMean) / imageASTD * target[1][1] + target[1][0]
		imageB = (imageB - imageBMean) / imageBSTD * target[2][1] + target[2][0]

		imagelab = cv2.merge((imageL, imageA, imageB))
		imagelab = np.clip(imagelab, 0, 255)
		imagelab = imagelab.astype(np.uint8)

		# Back to RGB space
		returnimage = cv2.cvtColor(imagelab, cv2.COLOR_LAB2RGB)
		# Replace white pixels
		returnimage[whitemask] = image[whitemask]
		
		return transforms.ToPILImage()(returnimage)

###### End Transfroms ######
			
			
			
if __name__ == '__main__':

	#For reproducable results
	seed = 2011
	np.random.seed(seed)
	torch.manual_seed(seed)

	#The paper specifies batch_size = 1
	batch_size = 1
	max_epochs = 500
	
	#List of data augmentations to be applied on the data
	# TODO ...
	data_augmentation = [None,
						transforms.RandomRotation((90,90)),
						transforms.RandomRotation((180,180)),
						transforms.RandomRotation((270,270)),
						transforms.RandomHorizontalFlip(1),
						transforms.Compose([transforms.RandomHorizontalFlip(1),
											transforms.RandomRotation((90,90))]),
						transforms.Compose([transforms.RandomHorizontalFlip(1),
											transforms.RandomRotation((180,180))]),
						transforms.Compose([transforms.RandomHorizontalFlip(1),
											transforms.RandomRotation((270,270))])]	
	
	#This how you sequence/compose transformations
	data_transform = transforms.Compose([Normalize_AMC(),
										transforms.CenterCrop((572,572)),
										transforms.Grayscale(),
										transforms.ToTensor(),
										transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
										
	#This is how you add onto an existing sequence/composition
	anno_transform = transforms.Compose([transforms.CenterCrop((388,388)),
										transforms.ToTensor(), 
										BinarizeExample(threshold=0.000001),
										transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
	
	#load train dataset
	GlaS_train_dataset = GlaSDataset(transform=data_transform,
								transform_anno=anno_transform)#, 
								#desired_dataset='train')

	#load test dataset (unused)
	GlaS_test_dataset = GlaSDataset(transform=data_transform,
								transform_anno=anno_transform,
								desired_dataset='test')
	
	#create train data_loader
	train_loader = DataLoader(GlaS_train_dataset, 
							batch_size=batch_size,
							shuffle=True,
							pin_memory=True,
							num_workers=1)
						
	#create test data_loader (unused)
	test_loader = DataLoader(GlaS_test_dataset,
							batch_size=batch_size,
							shuffle=False,
							num_workers=1)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	#model = UNet(upsample_mode='transpose').to(device)
	model = UNet(upsample_mode='bilinear').to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.00001)
	
	print(' ==== Now running training for {} epochs ==== '.format(max_epochs))
	start_time = time.time()
	for epoch in range(1, max_epochs+1):
		train(model, device, train_loader, optimizer, epoch, log_interval=11)
		
	print("Elapsed time: ", time.time()-start_time, "sec")