import torch
import torch.nn.functional as F

class Jaccard_loss():

    def __call__(self, prediction, label):
        eps = 1e-15

        intersection = (F.sigmoid(prediction) * label).sum()
        union = F.sigmoid(prediction).sum() + label.sum()
        #print("mmmmmmmmmmmmmmmmmmmmmmmmm Intersection mmmmmmmmmmmmmmmmmmmmmmm: ", intersection)
        #print("mmmmmmmmmmmmmmmmmmmmmmmmm Union mmmmmmmmmmmmmmmmmmmmmmm: ", union)
        return -torch.log((intersection + eps) / (union - intersection + eps))

