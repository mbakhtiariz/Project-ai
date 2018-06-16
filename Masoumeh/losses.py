import torch

class Jaccard_loss():

    def __call__(self, prediction, label):
        eps = 1e-15

        intersection = (prediction * label).sum()
        union = prediction.sum() + label.sum()

        return torch.log((intersection + eps) / (union - intersection + eps))

