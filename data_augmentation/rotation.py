
class Rotation(object):
    """
    Rotates the image given a certain angle

    Args:
        angle (Angle): Angle of image rotation. Should be one of these: [0, 90, 180, 270]
    """

    def __init__(self, angle=0) -> None:
        if angle not in [0, 90, 180, 270]:
            raise ValueError("Angle should be one of [0, 90, 180, 270], but instead angle=" + str(angle) + " was given.")
        else:
            self._angle = angle

    def __call__(self, image):
        return image.rotate(self._angle)
