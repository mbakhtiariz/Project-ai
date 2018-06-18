from PIL import Image

class Binarize(object):
    """
        Binarize the image given a certain threshold

    Args:
        threshold (int or float): Desired threshold to binarize the image with

    """

    def __init__(self, threshold: (int, float) = 128) -> None:
        self._threshold = threshold

    def __call__(self, sample: tuple) -> tuple:
        return sample[0], self.binarize(sample[1])

    def binarize(self, image: Image) -> Image:
        return image.point(lambda x: 0 if x < self._threshold else 255)
