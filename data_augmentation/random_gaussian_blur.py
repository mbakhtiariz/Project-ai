from PIL import ImageFilter
from PIL.Image import Image
import numpy.random as random


class RandomGaussianNoise(object):

    def __call__(self, sample: tuple) -> tuple:
        return self.gaussian_blur(sample[0], sample[1])

    def gaussian_blur(self, image: Image, mask: Image) -> tuple:
        assert isinstance(image, Image) and isinstance(mask, Image)

        sigma = random.normal(0.0, 0.5, 1)

        return image.filter(ImageFilter.GaussianBlur(sigma)), mask
