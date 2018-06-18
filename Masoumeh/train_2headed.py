# Training for both segmentation and classification

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils, datasets
from torch.utils.data.sampler import SubsetRandomSampler

from hyperparam import load_hyperparams
from masUNet_2headed import UNet
from losses import Jaccard_loss
from new_GlaS_dataset import GlaSDataset


# Example of a transformation that can be executed on the sample image
class Binarize(object):
    """
        Binarize the image given a certain threshold

    Args:
        threshold (float or int): Desired threshold to binarize the image with

    """

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        img = img > self.threshold
        return img.float()


def compute_padding(w, h, target_w, target_h):
    x_pad = (target_w - w) // 2
    y_pad = (target_h - h) // 2
    # print ("+++++++++", x_pad, y_pad)
    return (x_pad, y_pad, x_pad + (target_w - w) % 2, y_pad + (target_h - h) % 2)


def imshow(original, predection, mask):

    images_batch = original
    anno_images_batch = mask
    pred_batch = predection

    grid = torchvision.utils.make_grid(images_batch, nrow=batch_size)
    grid2 = torchvision.utils.make_grid(anno_images_batch, nrow=batch_size)
    grid3 = torchvision.utils.make_grid(pred_batch, nrow=batch_size)

    print('grid.shape: ', grid.shape)
    print('grid T . shape: ', grid.numpy().transpose((1, 2, 0)).shape)

    # plot image and image_anno
    ax = plt.subplot(3, 1, 1)
    ax.axis('off')
    ax.set_title('Input batch')
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    # plot image and image_anno
    ax = plt.subplot(3, 1, 2)
    ax.axis('off')
    ax.set_title('mask')
    plt.imshow(100*grid2.numpy().transpose((1, 2, 0)))

    # plot image and image_anno
    ax = plt.subplot(3, 1, 3)
    ax.axis('off')
    ax.set_title('Input batch')
    plt.imshow(100*grid3.numpy().transpose((1, 2, 0)))
    plt.title('Pred')


def early_stopping(avg_loss, tolerance):

    # if condition is not improving, we have to finish
    if np.argmin(avg_loss) < tolerance:
        return True
    else:
        return False

def find_image_mask_new_size(hyper_params):
    img_w = hyper_params["img_w"]
    img_h = hyper_params["img_h"]

    # For x > 4 (x represents the size of feature map at the end of the down sampling layer)
    # Based on x, for d-layer network input size should be 2^(d-1)x + 4(2^(d)-1)
    # The margin between input & output 12*2^(d-1) - 8
    d = hyper_params["depth"]

    a = 2**(d - 1)
    b = 12 * (2**(d - 1)) - 4*(2**d) + 4

    # Find new mask size in a way to be slightly larger than original image

    mask_new_w = a * math.ceil(float(img_w + b) / a) - b
    mask_new_h = a * math.ceil(float(img_h + b) / a) - b

    diff = 12 * (2**(d - 1)) - 8
    img_new_w =  mask_new_w + diff
    img_new_h = mask_new_h + diff

    print(img_new_w, img_new_h, mask_new_w, mask_new_h)
    return img_new_w, img_new_h, mask_new_w, mask_new_h

if __name__ == '__main__':

    # Load cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # For reproducable results
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)


    # Load PARAMS:
    hyper_params = load_hyperparams("hyper_params")

    epochs = hyper_params["epochs"]
    batch_size = hyper_params["batchSize"]
    channels = hyper_params["channels"]
    learning_rate = hyper_params["lr"]
    lambda2 = hyper_params["lambda2"]


    #---------- Multi loss param ---------------
    cls_alpha = hyper_params["cls_alpha"]
    #---------- Validation split params----------
    valid_size = hyper_params["valid_size"]
    shuffle = hyper_params["shuffle"]
    #If using CUDA, num_workers should be set to 1 and pin_memory to True.
    pin_memory = hyper_params["pin_memory"] #False: cpu, True:cuda
    num_workers = hyper_params["num_workers"] # 0 if cpu 1 or more if cuda
    #---------- Early Stopping Param------------
    tolerance = hyper_params["tolerance"]
    #-------------------------------------------

    #------------- prob of each inner transformation ----------------
    flip_prob = hyper_params["flip_prob"]
    rotate_prob = hyper_params["rotate_prob"]
    elastic_deform_prob = hyper_params["elastic_deform_prob"]
    blur_prob = hyper_params["blur_prob"]
    jitter_prob = hyper_params["jitter_prob"]


    # We want to resize all input images to the same size and naturally have to change size of masks as well.
    # Depth of network is an hyper param and can be changed so for achieving this we calculate the new size based on depth of network
    img_new_w, img_new_h, mask_new_w, mask_new_h = find_image_mask_new_size(hyper_params) # for depth 3: 816, 564 / 776, 524


    # This how you sequence/compose transformations
    data_transform = lambda w, h: \
        transforms.Compose([transforms.Pad(padding=compute_padding(w, h, img_new_w, img_new_h), padding_mode='reflect'),
                            transforms.ToTensor()])

    # This is how you add onto an existing sequence/composition
    anno_transform = lambda w, h: \
        transforms.Compose([transforms.Pad(padding=compute_padding(w, h, mask_new_w, mask_new_h ), padding_mode='reflect'),
                            transforms.ToTensor(),
                            Binarize(threshold=0.001)])

    # Load train dataset
    GlaS_train_dataset = GlaSDataset(transform=data_transform,
                                     transform_anno=anno_transform,
                                     desired_dataset='train',
                                     flip_prob = flip_prob, rotate_prob = rotate_prob,
                                     elastic_deform_prob = elastic_deform_prob, blur_prob = blur_prob)

    # load valid dataset
    GlaS_valid_dataset = GlaSDataset(transform=data_transform,
                                     transform_anno=anno_transform,
                                     desired_dataset='train',
                                     flip_prob=flip_prob, rotate_prob=rotate_prob,
                                     elastic_deform_prob=elastic_deform_prob, blur_prob=blur_prob)

    # Load test dataset
    GlaS_test_dataset = GlaSDataset(transform=data_transform,
                                    transform_anno=anno_transform,
                                    desired_dataset='test')
#--------------------------------------------------------------------------------------------
# Code from:
# https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb#file-data_loader-py-L92
    num_train = len(GlaS_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
# --------------------------------------------------------------------------------------------

    # create data_loader
    valid_loader = DataLoader(GlaS_valid_dataset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              pin_memory=pin_memory,
                              num_workers=num_workers)


    # create data_loader
    train_loader = DataLoader(GlaS_train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              pin_memory=pin_memory,
                              num_workers=num_workers)

    # create data_loader (unused)
    test_loader = DataLoader(GlaS_test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)


    print("Length of train loader: ",len(train_loader))
    print("Length of valid loader: ", len(valid_loader))
    print("Length of test loader: ", len(test_loader))

    # Build Network:
    net = UNet(hyper_params).to(device)


    # ---------- Define loss criterion -------------
    if (hyper_params['loss'] == "jaccard"):
        seg_criterion = Jaccard_loss()
    else:
        seg_criterion = nn.CrossEntropyLoss()

    cls_criterion = nn.CrossEntropyLoss()


    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=lambda2)

    # For saving losses:
    out = open("loss.txt", "w")
    best_config = open("best_config.txt", "w")
    minimum_loss = np.inf
    # Each epoch has a training and validation phase
    for epoch in range(epochs):
        epoch_loss = []
        finish = False
        for phase in ['train', 'val']:
            if phase == 'train':
                print("*************** Train phase *************")
                loader = train_loader
            elif phase == 'val':
                print("*************** Valid phase *************")
                loader = valid_loader

            # Iterate over data.
            avg_loss = []
            for batch_index, sampled_batch in enumerate(loader):
                print("Epoch %d, Iteration %d: sampling images.. " % (epoch, batch_index))
                images = sampled_batch['image'].to(device)
                seg_labels = torch.squeeze(sampled_batch['image_anno'],dim = 1).to(device).long()
                cls_labels = sampled_batch['GlaS'].to(device)


                print("*******", images.size())
                #if images.size()[0] == 1:
                #    continue
                print("Iteration %d: computing forward pass.." % batch_index)
                # Forward pass
                seg_out, cls_out = net(images)
                print("Iteration %d: calculating loss..." % batch_index)

                cls_loss = cls_criterion(cls_out, cls_labels)
                seg_loss = seg_criterion(seg_out, seg_labels)

                loss = seg_loss + cls_alpha*cls_loss
                print("Iteration %d: doing backward pass..." % batch_index)
                # Backward and optimize
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    print("Iteration %d: now updating...." % batch_index)
                    optimizer.step()


                print("---------------------loss: %f", loss.item())
                out.write(str(loss.item()) + "\n")
                epoch_loss.append(loss.item())

            avg_loss.append(np.mean(epoch_loss))
            print("--------------------- average loss: %f", avg_loss[-1])

            if phase == 'val':
                if avg_loss[-1] < minimum_loss:
                    minimum_loss = avg_loss[-1]
                    torch.save(net.state_dict(), 'best_model.pth')
                    # later probably save the file of best config including hyperparams.
                    print("--------------------- min loss so far: %f", avg_loss[-1])
                if early_stopping(avg_loss, tolerance):
                    finish = True
                    break
        if finish == True:
            break

    #---------------------EVAL ----------------------
    # have to change this later to read from best_hyper_param file
    model = UNet(hyper_params).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    seg_correct = seg_total = seg_intersection = seg_union = 0.
    cls_correct = cls_total = 0.
    draw_flag = False
    with torch.no_grad():
        for batch_index, sampled_batch in enumerate(test_loader):
            images = sampled_batch['image'].to(device)
            seg_labels = torch.squeeze(sampled_batch['image_anno']).to(device)
            cls_labels = sampled_batch['GlaS'].to(device)
            seg_out, cls_out = model(images)
            _, seg_pred = torch.max(seg_out.data, 1)
            _, cls_pred = torch.max(cls_out.data, 1)

            seg_total += seg_labels.size(0) * seg_labels.size(1) * seg_labels.size(2)
            seg_correct += (seg_pred == seg_labels).sum().item()

            cls_total += cls_labels.size(0)
            cls_correct += (cls_pred == cls_labels).sum().item()

            seg_intersection += (seg_pred * seg_labels).sum()
            seg_union += seg_pred.sum() + seg_labels.sum()


            if batch_index == 2:
                draw_flag = True
                plt.figure()
                imshow(images[0], seg_pred[0], seg_labels[0])
                plt.axis('off')
                plt.ioff()



    seg_acc_test = 100.0 * seg_correct / seg_total
    seg_dice_test = 2.0 * float(seg_intersection) / float(seg_union)

    cls_acc_test = 100.0 * cls_correct / cls_total
    print('intersection: ', seg_intersection)
    print('union: ', seg_union)
    print('Accuracy of the network on the test images: ' , seg_acc_test)
    print('Dice of the network on the test images:', seg_dice_test)
    print('classification Accuracy of the network on the test images: ', cls_acc_test)

    if draw_flag:
        plt.show()