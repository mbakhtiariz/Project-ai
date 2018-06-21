from PIL import Image
'''
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

'''

class Binarize(object):
    """
        Binarize the image given a certain threshold

    Args:
        threshold (int or float): Desired threshold to binarize the image with

    """

    def __init__(self, threshold: (int, float) = 128) -> None:
        self._threshold = threshold

    def __call__(self, sample: tuple) -> tuple:
        return sample[0], self.binarize(sample[1]), sample[2]

    def binarize(self, image: Image) -> Image:
        return image.point(lambda x: 0 if x < self._threshold else 255)


# Example of a transformation that can be executed on the sample image
class Binarize_Output(object):
    """
        Binarize the image given a certain threshold

    Args:
        threshold (float or int): Desired threshold to binarize the image with

    """
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        img = img > self.threshold
        return img.float()

