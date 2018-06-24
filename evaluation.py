from UNet import UNet
import torch
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import utils, transforms
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
import sys


def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def eval_UNet(test_loader,model_path, test_output_path, act_type = 'sigmoid'):
    saving_counter = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dice_file = open(test_output_path + "/test_dice.txt", "w")
    test_acc_file = open(test_output_path + "/test_acc.txt", "w")
    model = UNet(upsample_mode='bilinear').to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    correct = total = intersection = union = 0.

    if act_type == 'sigmoid':
        activation = F.sigmoid().cuda()#torch.nn.Sigmoid().cuda()
        tresh = 0.5
    elif act_type == 'tanh':
        activation = torch.nn.Tanh().cuda()
        tresh = 0
    elif act_type == 'soft':
        activation = torch.nn.Softmax().cuda()
        tresh = 0.5

    draw_flag = False
    with torch.no_grad():
        for batch_i, sample in enumerate(test_loader):

            data, target = sample['image'], sample['image_anno']
            data, target = data.to(device), target.to(device)

            output = model(data)
            #pred = (F.sigmoid(output) > 0.5).float()
            pred = (activation(output) > tresh).float()

            total += target.size(0) * target.size(2) * target.size(3)
            correct += (pred == target).sum().item()

            intersection += (pred * target).sum()
            union += pred.sum() + target.sum()

            acc = 100.0 * correct / total
            dice = 2.0 * float(intersection) / float(union)

            test_dice_file.write(str(dice) + "\n")
            test_dice_file.close()
            test_dice_file = open(test_output_path + "/test_dice.txt", "a")


            test_acc_file.write(str(acc) + "\n")
            test_acc_file.close()
            test_acc_file = open(test_output_path + "/test_acc.txt", "a")

            print('intersection: ', intersection)
            print('union: ', union)
            print('Accuracy of the network on the test images: ', acc)
            print('Dice of the network on the test images:', dice)


            utils.save_image(data, "{}/input_{}.png".format(test_output_path, batch_i))
            utils.save_image(target, "{}/target_{}.png".format(test_output_path, batch_i))
            utils.save_image(F.sigmoid(output), "{}/output_{}.png".format(test_output_path, batch_i))
            utils.save_image(pred, "{}/thres_{}.png".format(test_output_path, batch_i))




if __name__ == '__main__':
    # Important: in every run change exp_num, loss_type and you may change the act_type
    exp_num = int(sys.argv[1]) #2
    loss_type = sys.argv[2]#'wbce'  # 'mse'  # , 'wbce' , 'jac'
    act_type = 'sigmoid'  # 'none', 'sigmoid', 'tanh', 'soft'
    result_path = "results/results_" + str(exp_num) + "_" + loss_type + "_" + act_type
    obj_name = result_path + "/hyper_param"
    hyper_params = load_obj(obj_name)
    best_model_path = result_path + '/best_model.pth'

    test_output_path = result_path + '/test'
    if not os.path.exists(test_output_path):
        os.makedirs(test_output_path)



    #obj = {'batch_size'=batch_size,'max_epochs':max_epochs,'valid_portion':valid_portion,'shuffle':shuffle,'pin_memory':pin_memory
    #    ,'num_workers':num_workers,'loss_type':loss_type,'act_type':act_type,'tolerance':tolerance, 'lr':lr}


    batch_size = hyper_params['batch_size']

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

    # load test dataset (unused)
    GlaS_test_dataset = GlaSDataset(transform=test_transformations,
                                    desired_dataset='test',
                                    data_expansion_factor=5)

    # create test data_loader (unused)
    test_loader = DataLoader(GlaS_test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=1)

    eval_UNet(test_loader, best_model_path, test_output_path, act_type)