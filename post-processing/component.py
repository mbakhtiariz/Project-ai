import torch
import cv2
import numpy as np
from torchvision.transforms import transforms

from GlaS_dataset import GlaSDataset
from data_augmentation.binarize import Binarize
from data_augmentation.pil_image import ToPILImage
from data_augmentation.tensor import ToTensor
from data_augmentation.transpose_and_sqeeze import TransposeAndSqueeze


def connected_components(img: (torch.Tensor, np.ndarray), display: bool = False):
    img = img.astype(np.uint8)
    max_value = 1
    threshold = 0.5
    img = cv2.threshold(img, threshold, max_value, cv2.THRESH_BINARY)[1]  # ensure binary

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


# Example for the proof-of-concept:
#   Draws the first 4 images and their segmentation
#   Including their GlaS grade and (Sirinukunwattana et al. 2015) grade

if __name__ == '__main__':
    transformations = transforms.Compose([
        ToPILImage(),
        Binarize(threshold=0.00001),
        ToTensor(),
        TransposeAndSqueeze()
    ])

    # load dataset
    dataset = GlaSDataset(desired_dataset='test', transform=transformations)
    sample = dataset[1]

    image_annotation = sample['image_anno']
    ret, labels = connected_components(image_annotation, display=False)
