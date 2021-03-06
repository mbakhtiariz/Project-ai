import torch

from torchvision.transforms import transforms


class Normalise(object):
    """
    Randomly flips an image and its mask.
    """

    def __init__(self) -> None:
        self._normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __call__(self, sample: tuple) -> tuple:
        return self.normalise(sample[0], sample[1],sample[2])

    def normalise(self, image: (torch.FloatTensor, torch.LongTensor), mask: (torch.FloatTensor, torch.LongTensor),weight: (torch.FloatTensor, torch.LongTensor)) -> tuple:
        assert isinstance(image, (torch.FloatTensor, torch.LongTensor)) and isinstance(mask, (torch.FloatTensor, torch.LongTensor)) and isinstance(weight, (torch.FloatTensor, torch.LongTensor))

        return self._normalize(image), mask, weight
