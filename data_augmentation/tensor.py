from torchvision.transforms import transforms


class ToTensor(object):
    """
    Randomly flips an image and its mask.
    """

    def __init__(self) -> None:
        self.to_tens = transforms.ToTensor()

    def __call__(self, sample: tuple) -> tuple:
        return self.toTensor(sample[0], sample[1], sample[2])

    def toTensor(self, image, mask,weight) -> tuple:
        """
        Args:
            image (PIL Image or numpy.ndarray): Image to be converted to tensor.
            mask (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            ToTensor: Converted image.
        """
        #print("before tensor:")
        #print(weight.getextrema())
        #print("after tensore")
        return self.to_tens(image), self.to_tens(mask), self.to_tens(weight)
