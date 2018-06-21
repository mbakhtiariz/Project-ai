import numpy as np
from torchvision.transforms import transforms


class ToPILImage(object):
    """
    Randomly flips an image and its mask.
    """

    def __init__(self) -> None:
        self._to_pil = transforms.ToPILImage()

    def __call__(self, sample: tuple) -> tuple:
        return self.to_pil_image(sample[0], sample[1], sample[2])

    def to_pil_image(self, image, mask, weight) -> tuple:
        """
        Args:
            image (Tensor or numpy.ndarray): Image to be converted to PIL Image.
            mask (Tensor or numpy.ndarray): Image to be converted to PIL Image.

        Returns:
            pill_image: Converted PIL image.
        """
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
        if len(weight.shape) == 2:
            weight = np.expand_dims(weight, axis=2)
        return self._to_pil(image), self._to_pil(mask), self._to_pil(weight)
