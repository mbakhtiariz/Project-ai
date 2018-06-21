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


def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    for batch_i, sample in enumerate(train_loader):
        data, target = sample['image'], sample['image_anno']

        #print(sample['loss_weight'])
        #print(torch.max(sample['loss_weight']))
        #print(torch.min(sample['loss_weight']))
        # target = torch.squeeze(target, dim=0)
        data, target = data.to(device), target.to(device)


        optimizer.zero_grad()
        output = model(data)
        activation = torch.nn.Sigmoid()
        criterion = torch.nn.BCELoss(weight=sample['loss_weight'],size_average=False).cuda()
        loss = criterion(activation(output), target)
        #loss_weights = torch.squeeze(sample['loss_weight']).to(device)
        #_target = torch.squeeze(target).to(device)
        #_output = torch.squeeze(output).to(device)
        #print("lllllllllllllllllloooooooooooooooosssssssss:",loss_weights.size())
        #print("lllllllllllllllllloooooooooooooooosssssssss:",_output.size())
        #print("lllllllllllllllllloooooooooooooooosssssssss:",_target.size())
        #loss = F.cross_entropy(_output, _target ,weight = loss_weights)
        #loss = F.mse_loss(output, target)
        # loss = stable_bce_loss(output, target)
        # loss = F.binary_cross_entropy(output, target)	#Needs LongTensor, given FloatTensor
        # loss = F.l1_loss(output, target)		#Horribly high loss
        # loss = F.nll_loss(torch.exp(output), target)		#Needs LongTensor, given FloatTensor
        # loss = F.binary_cross_entropy_with_logits(output, target)
        # loss = jaccard_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_i % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_i * len(data), len(train_loader.dataset),
                       100. * batch_i / len(train_loader), loss.item()))



        if batch_i == 1:


            post_transform = transforms.Compose([Binarize_Output(threshold=output.mean())])
            thres = post_transform(output)
            # hist_eq = torch.histc(output.to(torch.device("cpu")))
            utils.save_image(data, "output/input_{}.bmp".format(epoch))
            utils.save_image(target, "output/target_{}.bmp".format(epoch))

            utils.save_image(output, "output/output_{}.bmp".format(epoch))
            utils.save_image(thres, "output/thres_{}.bmp".format(epoch))


if __name__ == '__main__':

    # For reproducible results
    seed = 2011
    np.random.seed(seed)
    torch.manual_seed(seed)

    # The paper specifies batch_size = 1
    batch_size = 1
    max_epochs = 500

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
    GlaS_train_dataset = GlaSDataset(transform=transformations, data_expansion_factor=5)

    # load test dataset (unused)
    GlaS_test_dataset = GlaSDataset(transform=transformations,
                                    desired_dataset='test',
                                    data_expansion_factor=5)

    # create train data_loader
    train_loader = DataLoader(GlaS_train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=1)

    # create test data_loader (unused)
    test_loader = DataLoader(GlaS_test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = UNet(upsample_mode='transpose').to(device)
    model = UNet(upsample_mode='bilinear').to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    print(' ==== Now running training for {} epochs ==== '.format(max_epochs))
    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval=1)

    print("Elapsed time: ", time.time() - start_time, "sec")
