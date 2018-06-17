from PIL import Image


class Flip(object):
    """
    Flips the image given a certain orientation

    Args:
        vertical (bool): If should be vertical or horizontal. false=horizontal, true=vertical.
    """

    def __init__(self, vertical: bool = False) -> None:
        if not isinstance(vertical, bool):
            raise ValueError("Argument 'vertical' should be boolean, but instead got " + str(type(vertical)) + ".")
        else:
            self._vertical = vertical

    def __call__(self, image: Image) -> Image:
        if self._vertical:
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
