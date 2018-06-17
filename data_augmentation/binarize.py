from PIL import Image


class BinarizeExample(object):
    """
        Binarize the image given a certain threshold

    Args:
        threshold (int or float): Desired threshold to binarize the image with

    """

    def __init__(self, threshold: (int, float) = 128) -> None:
        self.threshold = threshold

    def __call__(self, img: Image) -> Image:
        img = img > self.threshold
        img = img.float()

        return img
