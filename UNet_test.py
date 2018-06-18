import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import utils, transforms

from UNet import UNet
from GlaS_dataloader_example import BinarizeExample
from GlaS_dataset import GlaSDataset

import matplotlib.pyplot as plt
import time
import random

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
			
			# #Failed attempt at showing them as plots once every epoch...
			# target = torch.squeeze(target)
			# output = torch.squeeze(output)
			# thres = torch.squeeze(thres)
			
			# ax = plt.subplot(1,3,1)
			# plt.tight_layout()
			# ax.axis('off')
			# plt.imshow(target)
			# ax = plt.subplot(1,3,2)
			# plt.tight_layout()
			# ax.axis('off')
			# plt.imshow(output.detach())
			# ax = plt.subplot(1,3,3)
			# plt.tight_layout()
			# ax.axis('off')
			# plt.imshow(thres)
			
			# fig.canvas.draw()
			
			#utils.save_image(hist_eq, "hist_eq_{}.bmp".format(epoch))

if __name__ == '__main__':

	#For reproducable results
	seed = 2011
	np.random.seed(seed)
	torch.manual_seed(seed)

	#The paper specifies batch_size = 1
	batch_size = 1
	max_epochs = 250
	
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
	data_transform = transforms.Compose([transforms.CenterCrop((572,572)),
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
	optimizer = optim.Adam(model.parameters(), lr=0.0001)
	
	print(' ==== Now running training for {} epochs ==== '.format(max_epochs))
	start_time = time.time()
	for epoch in range(1, max_epochs+1):
		train(model, device, train_loader, optimizer, epoch, log_interval=11)
		
	print("Elapsed time: ", time.time()-start_time, "sec")