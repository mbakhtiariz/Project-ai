# Training for both segmentation and classification

from __future__ import print_function, division

import torch
from Util_functions import load_hyperparams, prepare_data
from train import train_UNet
from evaluation import eval_UNet
#matplotlib.use('GTKAgg')

if __name__ == '__main__':

# TODO: loop over different hyper param files and run all
# for sub_folder in output_folders:
#   read hyper_param_file
#   do the training
#   save output in sub_folder


    # Load cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Cuda is loaded...")
    # Load PARAMS:
    hyper_params = load_hyperparams("hyper_params")
    print("Hyper params are loaded...")
    # Load data:
    train_loader, valid_loader, test_loader = prepare_data(hyper_params, device)
    print("data loaders are built ...")
    # train network and save results
    train_UNet(train_loader,valid_loader, hyper_params,device)
    print("training is finished!")
    # load best found model and evaluate it:
    #eval_UNet()

