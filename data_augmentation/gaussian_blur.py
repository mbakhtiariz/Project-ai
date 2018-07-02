import random

from PIL.Image import Image
from PIL import ImageFilter


class GaussianBlur(object):
    """
    Flips the image given a certain orientation

    Args:
        sigma (int, float, tuple, list).
        probability (int, float).
    """

    def __init__(self, probability: float = 1, sigma: (int, float, tuple, list) = 1.5) -> None:
        if not isinstance(probability, (int, float)):
            raise ValueError(
                "Argument 'probability' should be int or float, but instead got " + str(type(probability)) + ".")
        else:
            self._probability = probability

        if not isinstance(sigma, (int, float, tuple, list)):
            raise ValueError(
                "Argument 'sigma' should be int, float, tuple or list, but instead got " + str(type(sigma)) + ".")
        else:
            self._sigma = sigma

    def __call__(self, sample: tuple) -> tuple:
        return self.gaussian_blur(sample[0], sample[1], sample[2])

    def gaussian_blur(self, image: Image, mask: Image, weight: Image) -> tuple:
        assert isinstance(image, Image) and isinstance(mask, Image) and isinstance(weight, Image)

        if random.random() > self._probability:
            return image, mask, weight
        else:
            if not isinstance(self._sigma, (tuple, list)):
                self._sigma = [self._sigma]

            sigma = random.choices(self._sigma)
            if isinstance(sigma, list):
                sigma = sigma[0]
            return image.filter(ImageFilter.GaussianBlur(sigma)), mask, weight
