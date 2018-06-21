import random

import numpy as np
import cv2
from PIL.Image import Image, fromarray
from scipy.interpolate import griddata


class ElasticDeformation(object):
    """
    Deforms an image using bi-cubic interpolation.

    Args:
        grid_size (int): image will be divided in a grid of  'grid_size' by 'grid_size'
        displacement (int): how much will the center of a square in the grid will be shifted(in both x and y directions)
    """

    def __init__(self, probability: (int, float) = 1, grid_size: int = 3, displacement: int = 10) -> None:
        if not isinstance(grid_size, int):
            raise ValueError("Argument 'grid_size' should be int, but instead got " + str(type(grid_size)) + ".")
        else:
            self._grid_size = grid_size

        if not isinstance(displacement, int):
            raise ValueError("Argument 'displacement' should be int, but instead got " + str(type(displacement)) + ".")
        else:
            self._displacement = displacement

        if not isinstance(probability, (int, float)):
            raise ValueError(
                "Argument 'probability' should be int or float, but instead got " + str(type(probability)) + ".")
        else:
            self._probability = probability

    def __call__(self, sample: tuple) -> tuple:
        """
        I hope this function is suitable for our project(Currently, the border pixels to remain the same, because I
        didn't manage to do reflection padding on the image, so that I don't get black(NaN) pixes at the border).
        Another mention: this function is a little slow. :/
        """
        grid_size = 3
        displacement = 15
        return self.elastic_transform(sample[0], sample[1], sample[2], grid_size=grid_size, displacement=displacement)

    def elastic_transform(self, image: Image, mask: Image,weight: Image, grid_size: int, displacement: (int, float)) -> tuple:
        assert isinstance(image, Image) and isinstance(mask, Image) and isinstance(weight, Image)

        if random.random() > self._probability:
            return image, mask, weight

        width, height = image.size

        width_span = width / (grid_size * 2)
        height_span = height / (grid_size * 2)

        same_horizontal_border = np.array(np.meshgrid([0, height], np.arange(0, width, 1))).T.reshape(-1, 2)
        same_vertical_border = np.array(np.meshgrid(np.arange(0, height, 1), [0, width])).T.reshape(-1, 2)
        same_border = np.concatenate((same_horizontal_border, same_vertical_border), axis=0)

        displacement_point_y = np.arange(width_span, (2 * grid_size - 1) * width_span + 1, 2 * width_span)
        displacement_point_x = np.arange(height_span, (2 * grid_size - 1) * height_span + 1, 2 * height_span)
        source_points = np.array(np.meshgrid(displacement_point_y, displacement_point_x)).T.reshape(-1, 2)
        displaced_points = source_points + np.random.uniform(-displacement, displacement, source_points.shape)

        source_points = np.concatenate((source_points, same_border), axis=0)
        displaced_points = np.concatenate((displaced_points, same_border), axis=0)

        grid_x, grid_y = np.mgrid[0:height - 1:1j * height, 0:width - 1:1j * width]

        grid_z = griddata(displaced_points, source_points, (grid_x, grid_y), method='cubic')
        map_x_32 = np.append([], [ar[:, 1] for ar in grid_z]).reshape(height, width).astype('float32')
        map_y_32 = np.append([], [ar[:, 0] for ar in grid_z]).reshape(height, width).astype('float32')

        return fromarray(cv2.remap(np.array(image), map_x_32, map_y_32, cv2.INTER_CUBIC)), \
               fromarray(cv2.remap(np.array(mask), map_x_32, map_y_32, cv2.INTER_CUBIC)), \
               fromarray(cv2.remap(np.array(weight), map_x_32, map_y_32, cv2.INTER_CUBIC))

# if __name__ == '__main__':
#
#     transformations = transforms.Compose([
#         ToPILImage(),
#         ElasticDeformation(displacement=30),
#         ToTensor()])
#
#     dataset = GlaSDataset(csv_file="..\\data\\GlaS\\Grade.csv", root_dir="..\\data\\GlaS\\", transform=transformations)
#     sample = dataset[1]
#
#     image = sample['image'].numpy().transpose((1, 2, 0))
#     image_anno = np.squeeze(sample['image_anno'].numpy().transpose((1, 2, 0)), axis=2)
#
#     plt.figure()
#     plt.imshow(image)
#     plt.figure()
#     plt.imshow(image_anno)
#     plt.show()
