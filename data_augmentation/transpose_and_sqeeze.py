import numpy as np
from PIL.Image import Image
from torchvision.transforms import transforms


class TransposeAndSqueeze(object):
    """
    Transposes the color channel on the last dimensions, and gets rid of last dimension for mask.
    """

    def __init__(self) -> None:
        self.to_tens = transforms.ToTensor()

    def __call__(self, sample: tuple) -> tuple:
        return self.transpose_and_squeeze(sample[0], sample[1])

    def transpose_and_squeeze(self, image, mask) -> tuple:
        """
        Args:
            image (PIL Image or torch.Tensor): Tensor to be transposed
            mask (PIL Image or torch.Tensor): Tensor to be transposed and squeezed.

        Returns:
            numpy.ndarray: Transposed image and mask.
        """
        if isinstance(image, Image):
            image = self.to_tens(image)
            mask = self.to_tens(mask)

        return image.numpy().transpose((1, 2, 0)), np.squeeze(mask.numpy().transpose((1, 2, 0)), axis=2)
