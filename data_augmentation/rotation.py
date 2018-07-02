import random
from PIL.Image import Image


class Rotation(object):
    """
    Rotates the image given a certain angle

    Args:
        angle (int): Angle of image rotation. Should be one of these: [0, 90, 180, 270]
    """

    def __init__(self, angles: (tuple, list) = (0, 90, 180, 270)) -> None:
        for angle in angles:
            if angle not in (0, 90, 180, 270):
                raise ValueError(
                    "Angle should be one of [0, 90, 180, 270], but instead angle=" + str(angle) + " was given.")
        self._angles = angles

    def __call__(self, sample: tuple) -> tuple:
        return self.rotate(sample[0], sample[1], sample[2])

    def rotate(self, image: Image, mask: Image, weight: Image) -> tuple:
        assert isinstance(image, Image) and isinstance(mask, Image) and isinstance(weight, Image)
        angle = random.choices(self._angles)
        if isinstance(angle, list):
            angle = angle[0]

        return image.rotate(angle, expand=1), mask.rotate(angle, expand=1), weight.rotate(angle, expand=1)
