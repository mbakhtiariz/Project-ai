import random
from PIL.Image import Image, FLIP_TOP_BOTTOM, FLIP_LEFT_RIGHT


class Flip(object):
    """
    Randomly flips an image and its mask.
    """

    def __call__(self, sample: tuple) -> tuple:
        return self.flip(sample[0], sample[1], sample[2])

    def flip(self, image: Image, mask: Image, weight: Image) -> tuple:
        assert isinstance(image, Image) and isinstance(mask, Image) and isinstance(weight,Image)

        probability = random.random()
        if probability < 0.33:
            return image, mask, weight
        elif probability < 0.66:
            return image.transpose(FLIP_TOP_BOTTOM), mask.transpose(FLIP_TOP_BOTTOM), weight.transpose(FLIP_TOP_BOTTOM)
        else:
            return image.transpose(FLIP_LEFT_RIGHT), mask.transpose(FLIP_LEFT_RIGHT), weight.transpose(FLIP_LEFT_RIGHT)
