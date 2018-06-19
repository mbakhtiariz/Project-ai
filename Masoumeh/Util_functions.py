from __future__ import print_function, division
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from transform_img import Binarize
from new_GlaS_dataset import GlaSDataset
import math
import numpy as np

def load_hyperparams(param_path):
    param_file = open(param_path, "r")

    hyperparams = defaultdict()

    keywords = ["loss",'filterNumStart', "lr", "epochs",
                "lambda2", "batchSize", "doBatchNorm", "channels",
                "dropout", "depth", "valid_size", "shuffle",
                "pin_memory", "num_workers", "tolerance","cls_alpha",
                "img_w", "img_h", "mask_w", "mask_h",
                "flip_prob", "rotate_prob", "elastic_deform_prob", "blur_prob",
                "jitter_prob"]


    types = ["string", "int", "float", "int",
             "float", "int", "int", "int",
             "float", "int","float", "string",
             "string", "int", "float","float",
             "int", "int", "int", "int",
             "float", "float", "float", "float",
             "float"]


    key_type = {}
    for i in range(len(keywords)):
        key_type[keywords[i]] = types[i]
    #print(key_type)

    for line in param_file:
        info = line.replace(' ', '').strip().split('=')
        if (key_type[info[0]] in ["float"]):
            print(info)
            hyperparams[info[0]] = float(info[1])
        elif (key_type[info[0]] in ["int"]):
            hyperparams[info[0]] = int(info[1])
        else:
            hyperparams[info[0]] = info[1]

    #print(hyperparams)
    return hyperparams

#-------------------------------------------------------------------------------
class jaccard_loss():

    def __call__(self, prediction, label):
        eps = 1e-5
        intersection = (F.sigmoid(prediction) * label).sum()
        union = F.sigmoid(prediction).sum() + label.sum()
        return -torch.log((intersection + eps) / (union - intersection + eps))

#-------------------------------------------------------------------------------
def histog(dataset):
    hist = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        x, y, z = sample['image'].shape
        #x, y = sample['image_anno'].shape
        if (x, y) not in hist:
            hist[(x, y)] = 1
        else:
            hist[(x, y)] += 1
    return hist
#------------------------------------------------------------------------------

def compute_padding(w, h, target_w, target_h):
    x_pad = (target_w - w) // 2
    y_pad = (target_h - h) // 2
    # print ("+++++++++", x_pad, y_pad)
    return (x_pad, y_pad, x_pad + (target_w - w) % 2, y_pad + (target_h - h) % 2)

def early_stopping(epoch_num, avg_loss, tolerance):

    # if condition is not improving, we have to finish
    if epoch_num - np.argmin(avg_loss) > tolerance:
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



def prepare_data(hyper_params, device):

    # For reproducable results
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)


    valid_size = hyper_params["valid_size"]
    shuffle = hyper_params["shuffle"]
    batch_size = hyper_params["batchSize"]

    if torch.cuda.is_available():
        pin_memory = True  # hyper_params["pin_memory"]  # False: cpu, True:cuda
        num_workers = 1  # hyper_params["num_workers"]  # 0 if cpu 1 or more if cuda
    else:
        pin_memory = False  # hyper_params["pin_memory"]  # False: cpu, True:cuda
        num_workers = 0  # hyper_params["num_workers"]  # 0 if cpu 1 or more if cuda

    # ------------- prob of each inner transformation ----------------
    flip_prob = hyper_params["flip_prob"]
    rotate_prob = hyper_params["rotate_prob"]
    elastic_deform_prob = hyper_params["elastic_deform_prob"]
    blur_prob = hyper_params["blur_prob"]
    jitter_prob = hyper_params["jitter_prob"]

    # We want to resize all input images to the same size and naturally have to change size of masks as well.
    # Depth of network is an hyper param and can be changed so for achieving this we calculate the new size based on depth of network
    img_new_w, img_new_h, mask_new_w, mask_new_h = find_image_mask_new_size(
        hyper_params)  # for depth 3: 816, 564 / 776, 524

    # This how you sequence/compose transformations
    data_transform = lambda w, h: \
        transforms.Compose([transforms.Pad(padding=compute_padding(w, h, img_new_w, img_new_h), padding_mode='reflect'),
                            transforms.ToTensor()])

    # This is how you add onto an existing sequence/composition
    anno_transform = lambda w, h: \
        transforms.Compose(
            [transforms.Pad(padding=compute_padding(w, h, mask_new_w, mask_new_h), padding_mode='reflect'),
             transforms.ToTensor(),
             Binarize(threshold=0.001)])

    # Load train dataset
    GlaS_train_dataset = GlaSDataset(transform=data_transform,
                                     transform_anno=anno_transform,
                                     desired_dataset='train',
                                     flip_prob=flip_prob, rotate_prob=rotate_prob,
                                     elastic_deform_prob=elastic_deform_prob, blur_prob=blur_prob)

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
    # --------------------------------------------------------------------------------------------
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

    print("Length of train loader: ", len(train_loader))
    print("Length of valid loader: ", len(valid_loader))
    print("Length of test loader: ", len(test_loader))

    return train_loader, valid_loader, test_loader

#-----------------------------------------------------
def prep_img(img):
    #PIL-image must be HxWxC, thus must have 3 dimensions
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    if len(img.shape) == 4:
        img = torch.squeeze(img, 0)
    img = transforms.functional.to_pil_image(img)
    return img