import cv2
import torch
import torch.nn.functional as F
from torchvision import utils, transforms
from torch.utils.data import DataLoader
from UNet import UNet
from data_augmentation.binarize import Binarize, Binarize_Output
from data_augmentation.center_crop import CenterCrop
from data_augmentation.grayscale import Grayscale
from data_augmentation.normalise import Normalise
from data_augmentation.pil_image import ToPILImage
from data_augmentation.normalise_rgb import NormaliseRGB
from data_augmentation.tensor import ToTensor
from  GlaS_dataset import GlaSDataset
from UNet_test import jaccard_loss
import pickle
import os
import sys
import numpy as np
from collections import Counter


def connected_components(img, display):
    ret, labels = cv2.connectedComponents(img, connectivity=4)
    labels_return = labels

    if display:
        # Map component labels to hue val
        label_hue = np.uint8(179 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # cvt to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue == 0] = 0

        cv2.imshow('labeled.png', labeled_img)
        cv2.waitKey()
    return ret, labels_return


def find_TP_FP_FN(output, target, bin_output, bin_target):
    # For adding to TP & FN counter:
    #   Binarize output and loop on glands on target.
    #       Then check how many percent of target gland is detected as tumor.
    #       If the obtained value is more than 50% :
    #           Add 1 to the TP counter.
    #       Else:
    #           Add 1 to the FN counter.

    TP_counter = 0
    FN_counter = 0
    glands = np.unique(target)  # should include zero for background, [0,1,2,...]

    for i in glands:
        if i != 0:
            filtered_target = ((target == i)).astype(int)  # Filter out all glands except g th one.
            TP_pixels = sum(sum(filtered_target * bin_output))
            all_pixels_i = sum(sum(filtered_target))
            if float(TP_pixels) / float(all_pixels_i) >= 0.5:
                TP_counter += 1
            else:
                FN_counter += 1

    # For adding to FP:
    #       Binarize target and loop on glands on output.
    #           Then check how many percent of output doesnt have over lap with any target
    # ------> if sum((~bin_mask)*output[g]) > 0.5 sum(output[g]):
    # The reason of ding this second loop on output glands if because of possible output glands without no overlap with target

    FP_counter = 0
    glands = np.unique(output)  # should include zero for background, [0,1,2,...]

    for j in glands:
        if j != 0:
            filtered_output = ((output == j)).astype(int)  # Filter out all glands except g th one.
            TP_pixels = sum(sum(filtered_output * bin_target))
            all_pixels_j = sum(sum(filtered_output))
            if float(TP_pixels) / float(all_pixels_j) <= 0.5:
                FP_counter += 1
        print(TP_counter, FP_counter, FN_counter)
    return TP_counter, FP_counter, FN_counter


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------------------------
def calc_overlap(_target_component, _output_component):
    """
    Here target and object do not necessarily represent the real real target and output of network.
    # Target is the clustered image which we intend to loop over its glands.
    # Output will be the clustered image that we desire to find the best match from it.
    """
    weighted_target_dice = 0
    glands = np.unique(_target_component)  # should include zero for background, [0,1,2,...]
    eps = 1e-15

    for i in glands:
        if i != 0:
            filtered_target = ((_target_component == i)).astype(int)  # Filter out all glands except i th one.
            num_gland_pixels = sum(sum(filtered_target))

            # find best matching gland for i th gland in target:
            gland_matches = filtered_target * _output_component

            seg_freq = Counter(list(gland_matches.flatten()))
            if 0 in seg_freq: del seg_freq[0]

            if len(seg_freq) > 0:
                # best_match = seg_freq.most_common(1)[0][0]
                intersection = seg_freq.most_common(1)[0][1]
                filtered_match = ((_output_component == seg_freq.most_common(1)[0][0])).astype(int)
                # filtered_match = ((_output_component == best_match)).astype(int)

                num_match_pixels = sum(sum(filtered_match))
                target_dice = (float(intersection) + eps) / (float(num_gland_pixels) + float(num_match_pixels) + eps)
                weighted_target_dice += float(num_gland_pixels) * target_dice

            else:
                weighted_target_dice += 0

    return weighted_target_dice


# ----------------------------------------------------------------------------------------------
def eval_UNet(test_loader, model_path, test_output_path, act_type='sigmoid', loss_type='mse'):
    """
    In this function we find scores of F1, Jaccard Index (IoU) and object level dice index:

    Steps of object-level-dice-index:
        # for every image load target and output
        #   add number of nonzero pixels to Gp and Sq
        #   loop over every gland in target.
        #       find the best match.
        #       calc dice between them.
        #       multiply by pixel num of that target gland
        #       add this value to gt_overlap

        #   do the same in reverse for output.
        #       add the result to seg_overlap

        #   final dice = 0.5 * [(gt_overlap/Gp) + (seg_overlap/Sq)]

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    test_loss_file = open(test_output_path + "/test_loss.txt", "w")
    test_F1_file = open(test_output_path + "/test_F1.txt", "w")
    test_IoU_file = open(test_output_path + "/test_IoU.txt", "w")
    test_precision_file = open(test_output_path + "/test_precision.txt", "w")
    test_recall_file = open(test_output_path + "/test_recall.txt", "w")
    test_objDice_file = open(test_output_path + "/test_objDice.txt", "w")


    model = UNet(upsample_mode='bilinear').to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    TP = FP = FN = 0
    Gp = Sq = 0  # Total target pixels & Total output pixels
    gt_overlap = seg_overlap = 0

    with torch.no_grad():
        for batch_i, sample in enumerate(test_loader):

            # Loading every image and its annotation:
            data, mask, loss_weight = sample['image'], sample['image_anno'], sample['loss_weight']
            data, mask, loss_weight = data.to(device), mask.to(device), loss_weight.to(device)
            loss_weight = loss_weight / 1000
            output = model(data)

            # Calculate loss:
            if loss_type == 'wbce':
                # Weighted BCE with averaging:
                activation = torch.nn.Sigmoid().cuda()
                criterion = torch.nn.BCELoss(weight=loss_weight).cuda()
                loss = criterion(activation(output), mask).cuda()
                pred = torch.squeeze(activation(output) > 0.5, dim=0).cpu().numpy().astype(
                    np.uint8)  # pred is binarized output with treshhold of 0.5
            elif loss_type == 'bce':
                # BCE with averaging:
                activation = torch.nn.Sigmoid().cuda()
                criterion = torch.nn.BCELoss().cuda()
                loss = criterion(activation(output), mask).cuda()
                pred = torch.squeeze(activation(output) > 0.5, dim=0).cpu().numpy().astype(np.uint8)
            elif loss_type == 'mse':
                # MSE:
                loss = F.mse_loss(output, mask).cuda()
                post_transform = transforms.Compose([Binarize_Output(threshold=output.mean())])
                pred = post_transform(output)
                pred = torch.squeeze(pred, dim=0).cpu().numpy().astype(np.uint8)  # binarized output
            else:
                activation = torch.nn.Sigmoid().cuda()
                loss = jaccard_loss(activation(output), mask).cuda()
                pred = torch.squeeze(activation(output) > 0.5, dim=0).cpu().numpy().astype(np.uint8)  # binarized output

            print("Batch: ", batch_i, ", Loss:", loss.item())
            bin_pred = np.squeeze(pred.transpose(1, 2, 0))  # covert to numpy for connecteComponent
            pred_ret, pred_component = connected_components(bin_pred, display=False)  # Find output connected components

            target = torch.squeeze(mask, dim=0).cpu().numpy().astype(np.uint8)
            bin_target = np.squeeze(target.transpose(1, 2, 0))
            target_ret, target_component = connected_components(bin_target, display=False)


            # ============================ Saving Images ===============================
            if loss_type == 'mse':
                trsh = output.mean()
                utils.save_image(output, "{}/test_output_{}.png".format(test_output_path, batch_i))
            else:
                trsh = 0.5
                utils.save_image(F.sigmoid(output), "{}/test_output_{}.png".format(test_output_path, batch_i))

            post_transform = transforms.Compose([Binarize_Output(threshold= trsh)])
            thres = post_transform(output)

            post_transform_weight = transforms.Compose([Binarize_Output(threshold=loss_weight.mean())])
            weight_tresh = post_transform_weight(loss_weight)

            utils.save_image(data, "{}/test_input_{}.png".format(test_output_path, batch_i))
            utils.save_image(mask, "{}/test_target_{}.png".format(test_output_path, batch_i))
            utils.save_image(thres, "{}/test_thres_{}.png".format(test_output_path, batch_i))
            utils.save_image(weight_tresh, "{}/test_weights_{}.png".format(test_output_path, batch_i))


            # ============================= F1 and Jaccard ============================
            # Find TP, FP, FN for every image
            _TP, _FP, _FN = find_TP_FP_FN(np.array(pred_component), np.array(target_component), bin_pred, bin_target)
            # Add up all of those local TP, FP, FN to the global ones:
            TP += _TP
            FP += _FP
            FN += _FN


            test_loss_file.write(str(loss.item()) + "\n")
            test_loss_file.close()
            test_loss_file = open(test_output_path + "/test_loss.txt", "a")


            # ============================ object level dice ==========================
            # We have to calculate dice for both side g->s & s->g
            Gp += sum(sum(bin_target))  # sum(sum(bin_target))
            Sq += sum(sum(bin_pred))  # sum(sum(bin_output))

            # g->s:
            # For every gland on target find the best match in output, Then calculate dice between them
            gt_overlap += calc_overlap(target_component, pred_component)
            # s->g:
            # For every gland on output find the best match in target, Then calculate dice between them
            seg_overlap += calc_overlap(pred_component, target_component)


    # ============================ Final step of F1 & Jac =========================
    eps = 1e-15  # To avoid devision by zero:
    Jac_index = float(TP + eps) / float(TP + FP + FN + eps)
    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    F1_score = 2 * (precision * recall + eps) / (precision + recall + eps)
    print("F1 score,Jac_index, precision, recall")
    print(F1_score, Jac_index, precision, recall)

    # =========================== Final object dice ===============================
    obj_dice = 0.5 * ((float(gt_overlap) / float(Gp)) + (float(seg_overlap) / float(Sq)))
    print("obj dice is:", obj_dice)

    # =========================== Saving results ===============================
    test_F1_file.write(str(F1_score) + "\n")
    test_F1_file.close()
    test_F1_file = open(test_output_path + "/test_F1.txt", "a")

    test_IoU_file.write(str(Jac_index) + "\n")
    test_IoU_file.close()
    test_IoU_file = open(test_output_path + "/test_IoU.txt", "a")

    test_precision_file.write(str(precision) + "\n")
    test_precision_file.close()
    test_precision_file = open(test_output_path + "/test_precision.txt", "a")

    test_recall_file.write(str(recall) + "\n")
    test_recall_file.close()
    test_recall_file = open(test_output_path + "/test_recall.txt", "a")

    test_objDice_file.write(str(obj_dice) + "\n")
    test_objDice_file.close()
    test_objDice_file = open(test_output_path + "/test_objDice.txt", "a")


    return F1_score, Jac_index, precision, recall, obj_dice



if __name__ == '__main__':
    # Important: in every run change exp_num, loss_type and you may change the act_type
    exp_num = int(sys.argv[1])  # 2
    loss_type = sys.argv[2]  # 'wbce'  # 'mse'  # , 'wbce' , 'jac'
    act_type = 'sigmoid'  # 'none', 'sigmoid', 'tanh', 'soft'
    result_path = "prj-ai-results-26june/results_" + str(exp_num) + "_" + loss_type + "_" + act_type
    #result_path = "results/results_" + str(exp_num) + "_" + loss_type + "_" + act_type
    obj_name = result_path + "/hyper_param"
    hyper_params = load_obj(obj_name)

    print(hyper_params)
    best_model_path = result_path + '/best_model.pth'

    test_output_path = result_path + '/test'
    if not os.path.exists(test_output_path):
        os.makedirs(test_output_path)

    batch_size = 1  # hyper_params['batch_size']

    # List of data augmentations to be applied on the data
    test_transformations = transforms.Compose([
        ToPILImage(),
        NormaliseRGB(),
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
                                    data_expansion_factor=1)

    # create test data_loader (unused)
    test_loader = DataLoader(GlaS_test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=1)

    eval_UNet(test_loader, best_model_path, test_output_path, act_type, loss_type)