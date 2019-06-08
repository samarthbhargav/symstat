import copy
import logging

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

log = logging.getLogger(__name__)


def balanced_batches(dataset, batch_size):
    unlabled_idx = dataset.unlabeled_idx
    labeled_idx = list(filter(lambda _: _ not in unlabled_idx,
                              np.arange(len(dataset.targets))))
    labeled_idx = np.array(labeled_idx)

    # construct batches - half of them should be from unlabled, half from labeled
    n_batches = (len(unlabled_idx) // (batch_size//2)) + 1
    for ulb in np.array_split(unlabled_idx, n_batches):
        batch_idx = list(ulb)
        lb = np.random.choice(labeled_idx, size=(
            batch_size // 2), replace=True)
        batch_idx.extend(lb)
        x_batch = []
        y_batch = []
        for idx in batch_idx:
            x, y = dataset[idx]
            x_batch.append(x)
            y_batch.append(y)
        yield torch.stack(x_batch), torch.LongTensor(y_batch)


def balanced_batches_heirarchy(dataset, heirarchy, batch_size):
    unlabled_idx = dataset.unlabeled_idx
    labeled_idx = list(filter(lambda _: _ not in unlabled_idx,
                              np.arange(len(dataset.targets))))
    labeled_idx = np.array(labeled_idx)

    # construct batches - half of them should be from unlabled, half from labeled
    n_batches = (len(unlabled_idx) // (batch_size//2)) + 1
    for ulb in np.array_split(unlabled_idx, n_batches):
        batch_idx = list(ulb)
        lb = np.random.choice(labeled_idx, size=(
            batch_size // 2), replace=True)
        batch_idx.extend(lb)
        x_batch = []
        y_batch = []
        for idx in batch_idx:
            x, y = dataset[idx]
            x_batch.append(x)
            y_batch.append(y)

        y_batch = heirarchy.to_vec(torch.LongTensor(y_batch))
        yield torch.stack(x_batch), y_batch


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
        if y_raw.ndimension() == 2:
            unlabeled_idx = (y_raw == -1).sum(1) > 0
        else:
            unlabeled_idx = y_raw == FashionMNIST.UNLABLED
        x, y = x_raw[~unlabeled_idx], y_raw[~unlabeled_idx]
        x_unlab, y_unlab = x_raw[unlabeled_idx], y_raw[unlabeled_idx]
        return x, y, x_unlab, y_unlab


class Hierarchy:

    def __init__(self, fmnist):
        self.org_class_to_idx = fmnist.class_to_idx
        self.org_idx_to_class = {
            v: k for (k, v) in self.org_class_to_idx.items()}
        self.org_n_classes = len(self.org_class_to_idx)

        Top = {"T-shirt/top", "Pullover", "Coat", "Shirt"}
        Shoes = {"Sandal", "Sneaker", "Ankle boot"}
        # simple one level heirarchy for now
        self.heirarchy = {
            "Top": Top,
            "Shoes": Shoes,
            "Bag": "Bag",
            "Dress": "Dress",
            "Trouser": "Trouser",
        }

        self.class_to_idx = copy.deepcopy(self.org_class_to_idx)
        # add new top level classes
        self.class_to_idx["Top"] = len(self.class_to_idx)
        self.class_to_idx["Shoes"] = len(self.class_to_idx)

        self.idx_to_class = {v: k for (k, v) in self.class_to_idx.items()}
        self.n_classes = len(self.class_to_idx)

        assoc_idx = {}
        neg_assoc_idx = {}
        all_idx = set(range(self.n_classes))
        for clz in self.class_to_idx:
            cls_idx = self.class_to_idx[clz]
            assoc_idx[cls_idx] = [self.class_to_idx[c]
                                  for c in self.find_classes(clz)]
            neg_assoc_idx[cls_idx] = []
            for idx in all_idx:
                if idx not in assoc_idx[cls_idx]:
                    neg_assoc_idx[cls_idx].append(idx)

        self.assoc_idx = assoc_idx
        self.neg_assoc_idx = neg_assoc_idx

    def find_classes(self, y, new_classes=None):
        if new_classes is None:
            new_classes = set()
        for k, v in self.heirarchy.items():
            if isinstance(v, set) and y in v:
                new_classes.add(k)
                new_classes.add(y)
            elif k == y:
                new_classes.add(k)
        return new_classes

    def to_vec(self, y):
        new_y = torch.zeros(y.size(0), self.n_classes)
        for idx, y_sub in enumerate(y.detach().numpy()):
            if y_sub == -1:
                new_y[idx, :] = -1.0
                continue
            classes = self.find_classes(self.org_idx_to_class[y_sub])
            classes_idx = [self.class_to_idx[c] for c in classes]
            new_y[idx, classes_idx] = 1.0
        return new_y

    def from_vector(self, v):
        pass


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

    fmnist = FashionMNIST("./fashion-mnist", 0.998, train=True,
                          transform=fmnist_transforms, download=True)

    for x, y in balanced_batches(fmnist, 16):
        print(x.size(), y)
    # for x_raw, y_raw in DataLoader(fmnist, batch_size=10):
    #     x, y, x_unlab, y_unlab = FashionMNIST.separate_unlabeled(x_raw, y_raw)

    #     print(y, y_unlab)
    #     print(x.size(), y.size())
    #     print(x_unlab.size(), y_unlab.size())
