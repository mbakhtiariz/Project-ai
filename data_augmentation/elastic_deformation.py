import numpy as np
import cv2
from PIL import Image
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from torchvision.transforms import functional

from GlaS_dataset import GlaSDataset


# def elastic_transform(image, alpha, sigma, random_state=None):
#     """Elastic deformation of images as described in [Simard2003]_.
#     .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
#        Convolutional Neural Networks applied to Visual Document Analysis", in
#        Proc. of the International Conference on Document Analysis and
#        Recognition, 2003.
#     """
#     if random_state is None:
#         random_state = np.random.RandomState(None)
#
#     shape = image.shape
#
#     dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
#     dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
#
#     x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
#
#     indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
#
#     distored_image = map_coordinates(image, indices, order=3, mode='reflect')
#     return distored_image.reshape(image.shape)


class ElasticDeformation(object):
    """
    Deforms an image using bi-cubic interpolation.

    Args:
        grid_size (int): image will be divided in a grid of  'grid_size' by 'grid_size'
        displacement (int): how much will the center of a square in the grid will be shifted(in both x and y directions)
    """

    def __init__(self, grid_size: int = 3, displacement: int = 10) -> None:
        if not isinstance(grid_size, int):
            raise ValueError("Argument 'grid_size' should be int, but instead got " + str(type(grid_size)) + ".")
        else:
            self._grid_size = grid_size

        if not isinstance(displacement, int):
            raise ValueError("Argument 'displacement' should be int, but instead got " + str(type(displacement)) + ".")
        else:
            self._displacement = displacement

    def __call__(self, image: Image) -> Image:
        """
        I hope this function is suitable for our project(Currently, the border pixels to remain the same, because I
        didn't manage to do reflection padding on the image, so that I don't get black(NaN) pixes at the border).
        Another mention: this function is a little slow. :/
        """
        width, height = image.size

        width_span = width / (self._grid_size * 2)
        height_span = height / (self._grid_size * 2)

        same_horizontal_border = np.array(np.meshgrid([0, height], np.arange(0, width, 1))).T.reshape(-1, 2)
        same_vertical_border = np.array(np.meshgrid(np.arange(0, height, 1), [0, width])).T.reshape(-1, 2)
        same_border = np.concatenate((same_horizontal_border, same_vertical_border), axis=0)

        displacement_point_y = np.arange(width_span, (2 * self._grid_size - 1) * width_span + 1, 2 * width_span)
        displacement_point_x = np.arange(height_span, (2 * self._grid_size - 1) * height_span + 1, 2 * height_span)
        source_points = np.array(np.meshgrid(displacement_point_y, displacement_point_x)).T.reshape(-1, 2)
        displaced_points = source_points + np.random.uniform(-self._displacement, self._displacement,
                                                             source_points.shape)

        source_points = np.concatenate((source_points, same_border), axis=0)
        displaced_points = np.concatenate((displaced_points, same_border), axis=0)

        grid_x, grid_y = np.mgrid[0:height - 1:1j * height, 0:width - 1:1j * width]

        grid_z = griddata(displaced_points, source_points, (grid_x, grid_y), method='cubic')
        map_x_32 = np.append([], [ar[:, 1] for ar in grid_z]).reshape(height, width).astype('float32')
        map_y_32 = np.append([], [ar[:, 0] for ar in grid_z]).reshape(height, width).astype('float32')

        return Image.fromarray(cv2.remap(np.array(image), map_x_32, map_y_32, cv2.INTER_CUBIC))


if __name__ == '__main__':
    dataset = GlaSDataset(csv_file="..\\data\\GlaS\\Grade.csv", root_dir="..\\data\\GlaS\\")
    sample = dataset[1]
    plt.figure()
    plt.imshow(ElasticDeformation(grid_size=3, displacement=15)(functional.to_pil_image(sample['image'])))
    plt.figure()
    plt.imshow(sample['image'])
    plt.show()
