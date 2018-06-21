from PIL.Image import Image
from torchvision.transforms import transforms


class CenterCrop(object):
    """
    Crops an image and its mask.
    """

    def __init__(self, image_crop, mask_crop) -> None:
        self._image_crop_size = image_crop
        self._image_crop = transforms.CenterCrop(self._image_crop_size)
        self._mask_crop_size = mask_crop
        self._mask_crop = transforms.CenterCrop(self._mask_crop_size)

    def __call__(self, sample: tuple) -> tuple:
        return self.centerCrop(sample[0], sample[1], sample[2])

    def centerCrop(self, image: Image, mask: Image, weight: Image) -> tuple:
        assert isinstance(image, Image) and isinstance(mask, Image) and isinstance(weight, Image)
        return self._image_crop(image), self._mask_crop(mask), self._mask_crop(weight)
