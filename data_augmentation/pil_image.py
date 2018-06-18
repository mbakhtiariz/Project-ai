import numpy as np
from torchvision.transforms import transforms


class ToPILImage(object):
    """
    Randomly flips an image and its mask.
    """

    def __call__(self, sample: tuple) -> tuple:
        return self.to_pil_image(sample[0], sample[1])

    def to_pil_image(self, image, mask) -> tuple:
        """
        Args:
            image (Tensor or numpy.ndarray): Image to be converted to PIL Image.
            mask (Tensor or numpy.ndarray): Image to be converted to PIL Image.

        Returns:
            pill_image: Converted PIL image.
        """
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
        to_pil_image = transforms.ToPILImage()
        return to_pil_image(image), to_pil_image(mask)
