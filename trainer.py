import os
import time
import math
import logging

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from models import SemanticLossModule
from fashion_mnsit import FashionMNIST

log = logging.getLogger(__name__)


class Multilabel:

    @staticmethod
    def f1_score(y_true, y_pred):
        # Compute F1 score per class and return as a dictionary
        # why micro? : https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-text-classification-1.html
        # "Microaveraged results are a measure of effectiveness on the large classes in a test collection"
        return f1_score(y_true, y_pred, average="micro")

    @staticmethod
    def recall_score(y_true, y_pred):
        return recall_score(y_true, y_pred, average="micro")

    @staticmethod
    def precision_score(y_true, y_pred):
        return precision_score(y_true, y_pred, average="micro")

    @staticmethod
    def accuracy_score(y_true, y_pred):
        return accuracy_score(y_true, y_pred)


def gather_outputs(forward_func, loader, threshold=0.5):
    y_true = []
    y_pred = []
    log.info("Gathering outputs")
    with torch.no_grad():
        for index, (x_raw, y_raw) in enumerate(loader):

            output = F.log_softmax(forward_func(x_raw), dim=1)
            pred   = output.argmax(dim=1, keepdim=True)

            y_pred.extend(pred.cpu().view(-1).numpy())
            y_true.extend(y_raw.cpu().view(-1).numpy())

            if (index + 1) % 1000 == 0:
                log.info("Eval loop: {} done".format(index + 1))

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return y_true, y_pred


def mkdir(path):
    if os.path.exists(path):
        return
    os.mkdir(path)


class Trainer(object):
    MODEL_WTS_DIR = "models"

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.device = torch.device(args.device)
        # self.dataset = args.dataset
        self.model_type = args.model
        assert self.model_type in {"sl"}
        self.model_id = args.model_id
        self.learning_rate = args.learning_rate
        self.unlabeled = args.unlabeled

        # load data
        self.dataset_sizes = {}
        self.datasets = {}
        self.dataloaders = {}
        self._load_data(args)

        log.info("Device: {}".format(self.device))

        # load model
        self.model = None
        self._create_model(args)

        mkdir(self._get_save_path())
        mkdir(os.path.join(self._get_save_path(), self.MODEL_WTS_DIR))

    def _get_save_path(self):
        return os.path.join("results", self.model_id)

    def _load_data(self, args):
        fmnist_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        train_idx = list(range(0, 50000))
        val_idx = list(range(50000, 60000))
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        self.datasets["training"] = FashionMNIST("./fashion-mnist", self.unlabeled,
                                                 transform=fmnist_transforms, download=True)
        self.dataloaders["training"] = DataLoader(self.datasets["training"],
                                                  batch_size=self.batch_size,
                                                  num_workers=self.num_workers, sampler=train_sampler)
        self.dataset_sizes["training"] = len(self.datasets["training"])

        # set train = False, unlabeled_data = 0.0 for val set
        self.datasets["val"] = FashionMNIST("./fashion-mnist", 0.0,
                                            transform=fmnist_transforms, download=True)
        self.dataloaders["val"] = DataLoader(self.datasets["val"],
                                             batch_size=self.batch_size,
                                             num_workers=self.num_workers, sampler=val_sampler)
        self.dataset_sizes["val"] = len(self.datasets["val"])

        # set train = False, unlabeled_data = 0.0 for test set
        self.datasets["test"] = FashionMNIST("./fashion-mnist", 0.0, train=False,
                                             transform=fmnist_transforms, download=True)
        self.dataloaders["test"] = DataLoader(self.datasets["test"],
                                              batch_size=self.batch_size, shuffle=False,
                                              num_workers=self.num_workers)
        self.dataset_sizes["test"] = len(self.datasets["test"])

        self.n_classes = self.datasets["training"].n_classes

    def _create_model(self, args):
        if self.model_type == "sl":
            self.model = SemanticLossModule(self.device, self.n_classes, args)
        else:
            raise ValueError("unknown model")

    def run_epoch(self, epoch, phase, device, optimizer):
        log.info("Phase: {}".format(phase))
        if phase == 'training':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_n = 0.0

        n_batches = (self.dataset_sizes[phase] // self.batch_size) + 1
        # Iterate over data.
        for batch_idx, (x_raw, y_raw) in enumerate(self.dataloaders[phase], 1):
            x, y, x_unlab, y_unlab = FashionMNIST.separate_unlabeled(
                x_raw, y_raw)

            if phase == "training":
                # zero the parameter gradients
                optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'training'):
                loss = self.model.compute_loss(x, y, x_unlab, y_unlab)

                # backward + optimize only if in training phase
                if phase == 'training':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * (len(x) + len(x_unlab))  # TODO: change?
            running_n += (len(x) + len(x_unlab))
            if batch_idx % 50 == 0:
                log.info("\t[{}/{}] Batch {}/{}: Loss: {:.4f}".format(phase,
                                                                      epoch,
                                                                      batch_idx,
                                                                      n_batches,
                                                                      running_loss / running_n))

        epoch_loss = running_loss / self.dataset_sizes[phase]

        if phase != 'training':
            log.info("Computing scores")
            y_true, y_pred = gather_outputs(
                self.model.forward, self.dataloaders[phase])

            scores = {
                "accuracy": Multilabel.accuracy_score(y_true, y_pred)
                # "f1": Multilabel.f1_score(y_true, y_pred),
                # "recall": Multilabel.recall_score(y_true, y_pred),
                # "precision": Multilabel.precision_score(y_true, y_pred)
            }

            log.info("Scores: {}".format(scores))

        log.info('{} Loss: {:.4f}'.format(
            phase, epoch_loss))

        return epoch_loss

    def train(self, num_epochs):

        root_path = self._get_save_path()

        model_path = os.path.join(root_path, "best_model.pkl")

        device = torch.device(self.device)

        self.model = self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        since = time.time()

        for epoch in range(1, num_epochs + 1):
            log.info('Epoch {}/{}'.format(epoch, num_epochs))

            train_loss = self.run_epoch(epoch,
                                        "training", device, optimizer)

            if math.isnan(train_loss):
                raise ValueError("NaN loss encountered")

            val_loss = self.run_epoch(epoch, "val", device, None)

            if math.isnan(val_loss):
                raise ValueError("NaN loss encountered")

            save_path = os.path.join(
                self._get_save_path(), self.MODEL_WTS_DIR, "model_{}.wts".format(epoch))

            torch.save(self.model.state_dict(), save_path)

        time_elapsed = time.time() - since
        log.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    def test(self):
        device = torch.device(self.device)
        self.model = self.model.to(device)
        self.run_epoch(0, "test", device, None)
