import copy
import logging

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

log = logging.getLogger(__name__)


class FashionMNIST(datasets.FashionMNIST):

    UNLABLED = -1

    def __init__(self, root, percent_unlabeled, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)
        assert percent_unlabeled >= 0.0 and percent_unlabeled <= 1.0
        if not train:
            # no unlabled data in the test set
            assert percent_unlabeled == 0.0

        self.true_targets = copy.deepcopy(self.targets)
        self.percent_unlabeled = percent_unlabeled

        log.info("Setting {}% of the targets to UNLABELED".format(
            self.percent_unlabeled * 100))

        self.unlabeled_idx = np.random.permutation(
            np.arange(0, len(self.targets)))[:int(self.percent_unlabeled * len(self.targets))]

        self.targets[self.unlabeled_idx] = self.UNLABLED

        self.n_classes = len(self.classes)

    def sample_labels(self, n):
        """Sample n targets from the labeled data

        Arguments:
            n {int} -- Number of samples
        """
        pass

    @staticmethod
    def separate_unlabeled(x_raw, y_raw):
        unlabeled_idx = y_raw == FashionMNIST.UNLABLED
        x, y = x_raw[~unlabeled_idx], y_raw[~unlabeled_idx]
        x_unlab, y_unlab = x_raw[unlabeled_idx], y_raw[unlabeled_idx]
        return x, y, x_unlab, y_unlab


if __name__ == '__main__':

    fmnist_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    fmnist = FashionMNIST("./fashion-mnist", 0.5,
                          transform=fmnist_transforms, download=True)

    for x_raw, y_raw in DataLoader(fmnist, batch_size=10):
        x, y, x_unlab, y_unlab = FashionMNIST.separate_unlabeled(x_raw, y_raw)

        print(x.size(), y.size())
        print(x_unlab.size(), y_unlab.size())

        break

    fmnist = FashionMNIST("./fashion-mnist", 0.0, train=False,
                          transform=fmnist_transforms, download=True)

    for x_raw, y_raw in DataLoader(fmnist, batch_size=10):
        x, y, x_unlab, y_unlab = FashionMNIST.separate_unlabeled(x_raw, y_raw)

        print(y, y_unlab)
        print(x.size(), y.size())
        print(x_unlab.size(), y_unlab.size())

        break
