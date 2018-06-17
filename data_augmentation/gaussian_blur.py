from PIL import Image, ImageFilter


class GaussianBlur(object):
    """
    Flips the image given a certain orientation

    Args:
        sigma (int, float).
    """

    def __init__(self, sigma: (int, float) = 2) -> None:
        if not isinstance(sigma, (int, float)):
            raise ValueError("Argument 'sigma' should be int or float, but instead got " + str(type(sigma)) + ".")
        else:
            self._sigma = sigma

    def __call__(self, image: Image) -> Image:
        return image.filter(ImageFilter.GaussianBlur(self._sigma))
