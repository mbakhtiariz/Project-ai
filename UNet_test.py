import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import utils, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('GTKAgg')
#matplotlib.use('TkAgg')
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
    eps = 1e-15
    intersection = (F.sigmoid(input) * target).sum()
    union = F.sigmoid(input).sum() + target.sum()
    return -torch.log((intersection + eps) / (union - intersection + eps))

# from https://github.com/pytorch/pytorch/issues/751
def stable_bce_loss(input, target):
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def train(model, device, train_loader, optimizer, loss_type, act_type, tolerance, result_path, log_interval=10):

    best_model_path = result_path + '/best_model.pth'

    train_batch_loss_file = open("output/train_batch_loss.txt", "w")
    valid_batch_loss_file = open("output/valid_batch_loss.txt", "w")

    train_all_epochs_loss_file = open("output/train_all_epochs_loss.txt", "w")
    train_all_epochs_loss = []

    valid_all_epochs_loss_file = open("output/valid_all_epochs_loss.txt", "w")
    valid_all_epochs_loss = []

    minimum_loss = np.inf
    finish = False

    for epoch in range(1, max_epochs + 1):
        model.train()

        for phase in ['train', 'val']:
            if phase == 'train':
                print("## Train phase ##")
                loader = train_loader
            elif phase == 'val':
                print("## Valid phase ##")
                loader = valid_loader

            all_batches_losses = []


            for batch_i, sample in enumerate(loader):
                data, target, loss_weight = sample['image'], sample['image_anno'], sample['loss_weight']#/1000
                data, target, loss_weight = data.to(device), target.to(device), loss_weight.to(device)


                loss_weight = loss_weight/1000
                if phase == 'train':
                    optimizer.zero_grad()
                output = model(data)

                # Set activation type:
                if act_type == 'sigmoid':
                    print("Sigmoid")
                    activation = torch.nn.Sigmoid()
                elif act_type == 'tanh':
                    print("Tanh")
                    activation = torch.nn.Tanh()
                elif act_type == 'soft':
                    print("Soft")
                    activation = torch.nn.Softmax()

                # Calculate loss:
                if loss_type == 'wbce':
                    print("W-BCE LOSS")
                    # Weighted BCE with averaging:
                    criterion = torch.nn.BCELoss(weight=loss_weight).cuda()#,size_average=False).cuda()
                    loss = criterion(activation(output), target)
                elif loss_type == 'bce':
                    print("BCE LOSS")
                    # BCE with averaging:
                    criterion = torch.nn.BCELoss().cuda()  # ,size_average=False).cuda()
                    loss = criterion(activation(output), target)
                elif loss_type == 'mse':
                    print("MSE LOSS")
                    # MSE:
                    #loss = F.mse_loss(activation(output), target)
                    loss = F.mse_loss(output, target)
                else:# loss_type == 'jac':
                    print("JAC LOSS")
                    loss = jaccard_loss(output, target)
                    # loss = stable_bce_loss(output, target)
                    # loss = F.binary_cross_entropy(output, target)	#Needs LongTensor, given FloatTensor
                    # loss = F.l1_loss(output, target)		#Horribly high loss
                    # loss = F.nll_loss(torch.exp(output), target)		#Needs LongTensor, given FloatTensor
                    # loss = F.binary_cross_entropy_with_logits(output, target)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    train_batch_loss_file.write(str(loss.item()) + "\n")
                else:
                    valid_batch_loss_file.write(str(loss.item()) + "\n")

                all_batches_losses.append(loss.item())

                if batch_i % log_interval == 0:
                    print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_i * len(data), len(loader.dataset),
                               100. * batch_i / len(loader), loss.item()))

            last_avg_loss = np.mean(all_batches_losses)
            if phase == 'train':
                train_all_epochs_loss_file.write(str(last_avg_loss) + "\n")
            if phase == 'val':
                valid_all_epochs_loss_file.write(str(last_avg_loss) + "\n")
                valid_all_epochs_loss.append(last_avg_loss)
                if last_avg_loss < minimum_loss:
                    minimum_loss = last_avg_loss

                    torch.save(model.state_dict(), best_model_path)

                    print("Minimum Average Loss so far:", minimum_loss)
                    post_transform = transforms.Compose([Binarize_Output(threshold=output.mean())])
                    thres = post_transform(output)

                    post_transform_weight = transforms.Compose([Binarize_Output(threshold=loss_weight.mean())])
                    weight_tresh = post_transform_weight(output)

                    utils.save_image(data, "{}/input_{}_{}.bmp".format(result_path,epoch, batch_i))
                    utils.save_image(target, "{}/target_{}_{}.bmp".format(result_path,epoch, batch_i))
                    utils.save_image(output, "{}/output_{}_{}.bmp".format(result_path,epoch, batch_i))
                    utils.save_image(thres, "{}/thres_{}_{}.bmp".format(result_path,epoch, batch_i))
                    utils.save_image(weight_tresh, "{}/weights_{}_{}.bmp".format(result_path,epoch, batch_i))
                if early_stopping(epoch, valid_all_epochs_loss, tolerance):
                    finish = True
                    break
        if finish == True:
            break


if __name__ == '__main__':

    if not os.path.exists("results"):
        os.makedirs("results")

    if not os.path.exists("results/results_mse"):
        os.makedirs("results/results_mse")
    if not os.path.exists("results/results_jac"):
        os.makedirs("results/results_jac")
    if not os.path.exists("results/results_wbce"):
        os.makedirs("results/results_wbce")
    if not os.path.exists("results/results_bce"):
        os.makedirs("results/results_bce")

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
    num_workers = 1 # 0 if using cpu

    loss_type = 'wbce'#'mse'  # , 'wbce' , 'jac'
    act_type = 'sigmoid'#'none'#'sigmoid' #''sigmoid'
    tolerance = 5  # some int
    result_path = "results/results_" + loss_type

    if loss_type == 'wbce':
        lr = 0.01
    if loss_type == 'bce':
        lr = 0.01
    elif loss_type == 'mse':
        lr = 0.000001
    else:
        lr = 0.001



    # Save setting in a flie:
    obj = {'max_epochs':max_epochs,'valid_portion':valid_portion,'shuffle':shuffle,'pin_memory':pin_memory
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

    # List of data augmentations to be applied on the data
    test_transformations = transforms.Compose([
        ToPILImage(),
        # Resize((572, 572)),
        #Rotation(),
        #Flip(),
        # GaussianBlur(sigma=[0.5, 0.7, 1, 1.3, 1.5, 1.7]),
        #RandomGaussianNoise(),
        #RandomHEStain(),
        #ElasticDeformation(displacement=20),
        #NormaliseRGB(),
        CenterCrop(image_crop=(572, 572), mask_crop=(388, 388)),
        Grayscale(),
        Binarize(threshold=0.000001),
        ToTensor(),
        Normalise(),
        # TransposeAndSqueeze()
    ])



    # load train dataset
    GlaS_train_dataset = GlaSDataset(transform=transformations, data_expansion_factor=5)


    # load valid dataset
    GlaS_valid_dataset = GlaSDataset(transform=transformations, data_expansion_factor=5)


    # load test dataset (unused)
    GlaS_test_dataset = GlaSDataset(transform=test_transformations,
                                    desired_dataset='test',
                                    data_expansion_factor=5)


    # For spliting validation set and training set:
    num_train = 10#len(GlaS_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_portion * num_train))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

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

    # create test data_loader (unused)
    test_loader = DataLoader(GlaS_test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = UNet(upsample_mode='transpose').to(device)
    model = UNet(upsample_mode='bilinear').to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.000001)

    print(' ==== Now running training for {} epochs ==== '.format(max_epochs))
    start_time = time.time()

    train(model, device, train_loader, optimizer, loss_type, act_type, tolerance, result_path, log_interval=1)
    print("Elapsed time: ", time.time() - start_time, "sec")
