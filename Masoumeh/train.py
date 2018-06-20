from __future__ import print_function, division
import numpy as np
from Util_functions import early_stopping, jaccard_loss
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
from dynamic_UNet import UNet
from new_GlaS_dataset import GlaSDataset
from torchvision import utils
from transform_img import Binarize
from torchvision import transforms
import random
def train_UNet(train_loader,valid_loader, hyper_params, device):

    epochs = hyper_params["epochs"]
    channels = hyper_params["channels"]
    learning_rate = hyper_params["lr"]
    lambda2 = hyper_params["lambda2"]
    cls_alpha = hyper_params["cls_alpha"] # Classification loss coef
    tolerance = hyper_params["tolerance"] # For early stopping

    '''
    # Build Network:
    net = UNet(hyper_params).to(device)


    # Define loss criterion:
    if (hyper_params['loss'] == "j"):
        seg_criterion = jaccard_loss()
    else:
        seg_criterion = F.binary_cross_entropy_with_logits

    cls_criterion = F.binary_cross_entropy_with_logits


    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=lambda2)
    
    # For saving losses:
    out = open("output/loss.txt", "w")
    best_config = open("output/best_config.txt", "w")
    minimum_loss = np.inf

    # Each epoch has a training and validation phase
    avg_loss = []
    '''


    data_augmentation = list(range(1,10))
    '''
        [None,
                         transforms.RandomRotation((90, 90)),
                         transforms.RandomRotation((180, 180)),
                         transforms.RandomRotation((270, 270)),
                         transforms.RandomHorizontalFlip(1),
                         transforms.Compose([transforms.RandomHorizontalFlip(1),
                                             transforms.RandomRotation((90, 90))]),
                         transforms.Compose([transforms.RandomHorizontalFlip(1),
                                             transforms.RandomRotation((180, 180))]),
                         transforms.Compose([transforms.RandomHorizontalFlip(1),
                                          transforms.RandomRotation((270, 270))])]
                                          
    '''


    for epoch in range(epochs):
        print("Epoch %d : " % epoch)
        finish = False
        for phase in ['train', 'val']:
            if phase == 'train':
                print("*************** Train phase *************")
                loader = train_loader
            elif phase == 'val':
                print("*************** Valid phase *************")
                loader = valid_loader
            # Iterate over data.
            one_epoch_losses = []
            for batch_index, sampled_batch in enumerate(loader):

                #augment = random.choice(data_augmentation)
                #for aug in range(1,10)data_augmentation:
                    #if aug:

                    images = Variable(sampled_batch['image'].to(device).float())
                    seg_labels = Variable(sampled_batch['image_anno'].to(device).float())
                    cls_labels = Variable(sampled_batch['GlaS'].to(device).float())
                    # Forward pass
                    utils.save_image(images, "output/input_{}_{}.bmp".format(epoch,batch_index))
                    utils.save_image(seg_labels, "output/target_{}_{}.bmp".format(epoch,batch_index))



            '''
                    seg_out, cls_out = net(images)
                    cls_out = torch.squeeze(cls_out, dim = 1).to(device)
                    # Find loss
                    cls_loss = cls_criterion(cls_out, cls_labels)
                    seg_loss = seg_criterion(seg_out, seg_labels)
                    batch_loss = seg_loss + cls_alpha*cls_loss
                    # Backward and optimize
                    if phase == 'train':
                        optimizer.zero_grad()
                        batch_loss.backward()
                        optimizer.step()
                    out.write(str(batch_loss.item()) + "\n")
                    one_epoch_losses.append(batch_loss.item())
    
            
                if phase == 'val':
                    print("1 epoch losses:", one_epoch_losses)
                    last_avg_loss = np.mean(one_epoch_losses)
                    avg_loss.append(last_avg_loss)
                    print("avg losses:", avg_loss)
                    if last_avg_loss < minimum_loss:
                        minimum_loss = last_avg_loss
                        print("Minimum Average Loss so far:", minimum_loss)
                        torch.save(net.state_dict(), 'output/best_model.pth')
                        #print("Final result dimension- seg:", seg_out.size())
                        #print("Final result dimension- cls:", cls_out.size())
                        print("Final result dimension- seg:", F.sigmoid(seg_out))
                        print("labels of seg:", seg_labels)
                        post_transform = transforms.Compose([Binarize(threshold=seg_out.mean())])
                        thres = post_transform(seg_out)
                        utils.save_image(thres, "output/thres_{}.bmp".format(epoch))
                        utils.save_image(images, "output/input_{}.bmp".format(epoch))
                        utils.save_image(F.sigmoid(seg_out), "output/output_{}.bmp".format(epoch))
                        utils.save_image(seg_labels, "output/target_{}.bmp".format(epoch))
                    if early_stopping(epoch, avg_loss, tolerance):
                        finish = True
                        break
            if finish == True:
                break
    
            '''