from PIL import Image
from torchvision.transforms import transforms
import collections


class Resize(object):
    """
    Randomly flips an image and its mask.
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self._interpolation = interpolation
        self._size = size

    def __call__(self, sample: tuple) -> tuple:
        return self.resize(sample[0], sample[1],sample[2])

    def resize(self, image, mask, weight) -> tuple:
        resize = transforms.Resize(self._size, self._interpolation)
        return resize(image), resize(mask),resize(weight)
