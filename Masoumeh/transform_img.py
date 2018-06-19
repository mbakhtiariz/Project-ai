# Example of a transformation that can be executed on the sample image
class Binarize(object):
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