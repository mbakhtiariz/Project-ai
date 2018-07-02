import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import utils, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from GlaS_dataset import GlaSDataset
from UNet import UNet
from data_augmentation.HEStain import RandomHEStain
from data_augmentation.binarize import Binarize, Binarize_Output
from data_augmentation.center_crop import CenterCrop
from data_augmentation.elastic_deformation import ElasticDeformation
from data_augmentation.flip import Flip
from data_augmentation.grayscale import Grayscale
from data_augmentation.normalise import Normalise
from data_augmentation.normalise_rgb import NormaliseRGB
from data_augmentation.pil_image import ToPILImage
from data_augmentation.random_gaussian_blur import RandomGaussianNoise
from data_augmentation.rotation import Rotation
from data_augmentation.tensor import ToTensor
import pickle
import os
import random
import sys


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)



def early_stopping(epoch_num, avg_loss, tolerance):
    # if condition is not improving, we have to finish
    if epoch_num - np.argmin(avg_loss) > tolerance:
        return True
    else:
        return False

def jaccard_loss(input, target):
    eps = 1e-11
    intersection = (input * target).sum()
    union = input.sum() + target.sum()
    return -torch.log((intersection + eps) / (union - intersection + eps))

# from https://github.com/pytorch/pytorch/issues/751
def stable_bce_loss(input, target):
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def train(train_loader,valid_loader, loss_type, act_type, tolerance, result_path, log_interval=10, lr=0.000001,max_epochs = 500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = UNet(upsample_mode='transpose').to(device)
    model = UNet(upsample_mode='bilinear').to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_model_path = result_path + '/best_model.pth'
    model_path = result_path + '/model_epoch'


    train_batch_loss_file = open(result_path + "/train_batch_loss.txt", "w")
    valid_batch_loss_file = open(result_path + "/valid_batch_loss.txt", "w")

    train_all_epochs_loss_file = open(result_path + "/train_all_epochs_loss.txt", "w")
    train_all_epochs_loss = []

    valid_all_epochs_loss_file = open(result_path + "/valid_all_epochs_loss.txt", "w")
    valid_all_epochs_loss = []

    minimum_loss = np.inf
    finish = False

    for epoch in range(1, max_epochs + 1):
        for phase in ['train', 'val']:
            if phase == 'train':
                idx = list(range(0, len(train_loader)))
                train_smpl = random.sample(idx, 1)
                #train_smpl.append(len(train_loader)-1)
                loader = train_loader
                model.train()
            elif phase == 'val':
                idx = list(range(0, len(valid_loader)))
                val_smpl = random.sample(idx, 1)
                #val_smpl.append(len(valid_loader) - 1)
                loader = valid_loader
                model.eval()

            all_batches_losses = []

            for batch_i, sample in enumerate(loader):
                data, target, loss_weight = sample['image'], sample['image_anno'], sample['loss_weight']#/1000
                data, target, loss_weight = data.to(device), target.to(device), loss_weight.to(device)

                optimizer.zero_grad()
                loss_weight = loss_weight/1000

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(data)
                    # Set activation type:
                    if act_type == 'sigmoid':
                        activation = torch.nn.Sigmoid().cuda()
                    elif act_type == 'tanh':
                        activation = torch.nn.Tanh().cuda()
                    elif act_type == 'soft':
                        activation = torch.nn.Softmax().cuda()

                    # Calculate loss:
                    if loss_type == 'wbce':
                        # Weighted BCE with averaging:
                        criterion = torch.nn.BCELoss(weight=loss_weight).cuda()#,size_average=False).cuda()
                        loss = criterion(activation(output), target).cuda()
                        #loss = criterion(output, target).cuda()
                    elif loss_type == 'bce':
                        # BCE with averaging:
                        criterion = torch.nn.BCELoss().cuda()  # ,size_average=False).cuda()
                        loss = criterion(activation(output), target).cuda()
                    elif loss_type == 'mse':
                        # MSE:
                        loss = F.mse_loss(output, target).cuda()
                    else:# loss_type == 'jac':
                        loss = jaccard_loss(activation(output), target).cuda()


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if phase == 'train':
                    train_batch_loss_file.write(str(loss.item()) + "\n")
                    train_batch_loss_file.close()
                    train_batch_loss_file = open(result_path + "/train_batch_loss.txt", "a")
                else:
                    valid_batch_loss_file.write(str(loss.item()) + "\n")
                    valid_batch_loss_file.close()

                    valid_batch_loss_file = open(result_path + "/valid_batch_loss.txt", "a")

                all_batches_losses.append(loss.item())

                if batch_i % log_interval == 0:
                    print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(phase,
                        epoch, batch_i * len(data), len(loader.dataset),
                               100. * batch_i / len(loader), loss.item()))

                if phase == 'train' and batch_i in train_smpl:
                    post_transform = transforms.Compose([Binarize_Output(threshold=output.mean())])
                    thres = post_transform(output)
                    post_transform_weight = transforms.Compose([Binarize_Output(threshold=loss_weight.mean())])
                    weight_tresh = post_transform_weight(loss_weight)
                    utils.save_image(data, "{}/train_input_{}_{}.png".format(result_path, epoch, batch_i))
                    utils.save_image(target, "{}/train_target_{}_{}.png".format(result_path, epoch, batch_i))
                    utils.save_image(output, "{}/train_output_{}_{}.png".format(result_path, epoch, batch_i))
                    utils.save_image(thres, "{}/train_thres_{}_{}.png".format(result_path, epoch, batch_i))
                    utils.save_image(weight_tresh, "{}/train_weights_{}_{}.png".format(result_path, epoch, batch_i))
                    
                    if epoch % 25 == 0:
                        torch.save(model.state_dict(), model_path+'_{}.pth'.format(epoch))

                if phase == 'val' and batch_i in val_smpl:
                    post_transform = transforms.Compose([Binarize_Output(threshold=output.mean())])
                    thres = post_transform(output)

                    post_transform_weight = transforms.Compose([Binarize_Output(threshold=loss_weight.mean())])
                    weight_tresh = post_transform_weight(loss_weight)

                    utils.save_image(data, "{}/valid_input_{}_{}.png".format(result_path, epoch, batch_i))
                    utils.save_image(target, "{}/valid_target_{}_{}.png".format(result_path, epoch, batch_i))
                    utils.save_image(output, "{}/valid_output_{}_{}.png".format(result_path, epoch, batch_i))
                    utils.save_image(thres, "{}/valid_thres_{}_{}.png".format(result_path, epoch, batch_i))
                    utils.save_image(weight_tresh, "{}/valid_weights_{}_{}.png".format(result_path, epoch, batch_i))

            if phase == 'train':
                train_last_avg_loss = np.mean(all_batches_losses)
                print("------average %s loss %f" % (phase, train_last_avg_loss))
                train_all_epochs_loss_file.write(str(train_last_avg_loss) + "\n")
                train_all_epochs_loss_file.close()
                train_all_epochs_loss_file = open(result_path + "/train_all_epochs_loss.txt", "a")
            if phase == 'val':
                valid_last_avg_loss = np.mean(all_batches_losses)
                print("------average %s loss %f" % (phase, valid_last_avg_loss))
                valid_all_epochs_loss_file.write(str(valid_last_avg_loss) + "\n")
                valid_all_epochs_loss_file.close()
                valid_all_epochs_loss_file = open(result_path + "/valid_all_epochs_loss.txt", "a")
                valid_all_epochs_loss.append(valid_last_avg_loss)
                if valid_last_avg_loss < minimum_loss:
                    minimum_loss = valid_last_avg_loss
                    #--------------------- Saving the best found model -----------------------
                    torch.save(model.state_dict(), best_model_path)
                    print("Minimum Average Loss so far:", minimum_loss)
                if early_stopping(epoch, valid_all_epochs_loss, tolerance):
                    finish = True
                    break

        if finish == True:
            break


if __name__ == '__main__':

    # Important: in every run change exp_num, loss_type and you may change the act_type
    exp_num = int(sys.argv[1])
    loss_type = sys.argv[2]#'wbce'  # 'mse'  # , 'wbce' , 'jac', 'bce'
    print(loss_type)
    act_type = 'sigmoid'  # 'none', 'sigmoid', 'tanh', 'soft'
    tolerance = 100 # some int
    #folder = "final_results"
    folder = "trained_models"
    result_path = folder+"/results_" + str(exp_num)+ "_"+loss_type + "_" + act_type

    if not os.path.exists(folder):
        os.makedirs(folder)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # For reproducible results
    seed = 2011
    np.random.seed(seed)
    torch.manual_seed(seed)

    # The paper specifies batch_size = 1
    batch_size = 1
    max_epochs = 500
    valid_portion = 0.1
    shuffle = True
    pin_memory = True # False if using cpu, True for Gpu
    num_workers = 2 # 0 if using cpu


    lr = float(sys.argv[3])
    print("learning rate",lr)


    # Save setting in a flie:
    obj = {'batch_size':batch_size,'max_epochs':max_epochs,'valid_portion':valid_portion,'shuffle':shuffle,'pin_memory':pin_memory
        ,'num_workers':num_workers,'loss_type':loss_type,'act_type':act_type,'tolerance':tolerance, 'lr':lr}
    save_obj(obj, result_path+"/hyper_param")


    # List of data augmentations to be applied on the data
    transformations = transforms.Compose([
        ToPILImage(),
        # Resize((572, 572)),
        Rotation(),
        Flip(),
        # GaussianBlur(sigma=[0.5, 0.7, 1, 1.3, 1.5, 1.7]),
        RandomGaussianNoise(),
        RandomHEStain(),
        ElasticDeformation(displacement=20),
        NormaliseRGB(),
        CenterCrop(image_crop=(572, 572), mask_crop=(388, 388)),
        Grayscale(),
        Binarize(threshold=0.000001),
        ToTensor(),
        Normalise(),
        # TransposeAndSqueeze()
    ])



    # load train dataset
    GlaS_train_dataset = GlaSDataset(transform=transformations, desired_dataset='train', data_expansion_factor=4)

    # For spliting validation set and training set:
    num_train = len(GlaS_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_portion * num_train))
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    
    # create data_loader    
    valid_loader = DataLoader(GlaS_train_dataset,
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


    print("Length of train loader: ", len(train_loader))
    print("Length of valid loader: ", len(valid_loader))


    print(' ==== Now running training for {} epochs ==== '.format(max_epochs))
    start_time = time.time()
    train(train_loader, valid_loader,loss_type, act_type, tolerance, result_path, log_interval=10, lr=lr,max_epochs =max_epochs)
    print("Elapsed time: ", time.time() - start_time, "sec")
