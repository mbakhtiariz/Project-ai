from PIL import Image
from torchvision.transforms import transforms


class Grayscale(object):
    """Transforms the given PIL.Image from rgb to grayscale"""

    def __init__(self) -> None:
        self._greyscale = transforms.Grayscale()

    def __call__(self, sample: tuple) -> tuple:
        return self.grayscale(sample[0], sample[1],sample[2])

    def grayscale(self, image: Image, mask: Image,weight: Image) -> tuple:
        return self._greyscale(image), mask,weight
