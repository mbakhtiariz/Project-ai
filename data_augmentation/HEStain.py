import numpy as np
import skimage
from PIL import Image
from numpy import random


class RandomHEStain(object):
    """Transfer the given PIL.Image from rgb to HE, perturbate, transfer back to rgb """

    def __call__(self, sample: tuple) -> tuple:
        return self.he_stain(sample[0], sample[1],sample[2])

    def he_stain(self, image: Image, mask: Image,weight: Image) -> tuple:
        img_he = skimage.color.rgb2hed(image)
        img_he[:, :, 0] = img_he[:, :, 0] * random.normal(1.0, 0.02, 1)  # H
        img_he[:, :, 1] = img_he[:, :, 1] * random.normal(1.0, 0.02, 1)  # E
        img_rgb = np.clip(skimage.color.hed2rgb(img_he), 0, 1)
        image = Image.fromarray(np.uint8(img_rgb * 255.999), image.mode)
        return image, mask,weight


# if __name__ == '__main__':
#     dataset = GlaSDataset(csv_file="..\\data\\GlaS\\Grade.csv", root_dir="..\\data\\GlaS\\")
#     img = transforms.ToPILImage()(dataset[1]['image'])
#
#     he_img, he_mask = RandomHEStain()(transforms.ToPILImage()((img, img)))
#     plt.figure()
#     plt.imshow(he_img)
#     plt.show()
#     np.array(he_img)
