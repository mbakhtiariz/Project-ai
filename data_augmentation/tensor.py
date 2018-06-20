from torchvision.transforms import transforms


class ToTensor(object):
    """
    Randomly flips an image and its mask.
    """

    def __call__(self, sample: tuple) -> tuple:
        return self.toTensor(sample[0], sample[1])

    def toTensor(self, image, mask) -> tuple:
        """
        Args:
            image (PIL Image or numpy.ndarray): Image to be converted to tensor.
            mask (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            ToTensor: Converted image.
        """
        # assert isinstance(image, Image) and isinstance(mask, Image)

        to_tens = transforms.ToTensor()
        return to_tens(image), to_tens(mask)
