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
from torch import autograd

from models import SemanticLossModule
from fashion_mnsit import FashionMNIST, balanced_batches

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

            output = torch.sigmoid(forward_func(x_raw))
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

    def run_epoch(self, epoch, phase, device, optimizer, w_s_weight):
        log.info("Phase: {}".format(phase))
        if phase == 'training':
            self.model.train()
        else:
            self.model.eval()

        running_loss       = 0.0
        running_loss_lab   = 0.0
        running_loss_unlab = 0.0
        running_n          = 0.0

        n_batches = (self.dataset_sizes[phase] // self.batch_size) + 1
        # Iterate over data.
        # for batch_idx, (x_raw, y_raw) in enumerate(self.dataloaders[phase], 1):
        for batch_idx, (x_raw, y_raw) in enumerate(balanced_batches(self.datasets[phase], self.batch_size)):
            x, y, x_unlab, y_unlab = FashionMNIST.separate_unlabeled(x_raw, y_raw)

            if phase == "training":
                # zero the parameter gradients
                optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'training'):

                if phase == 'training':
                    with autograd.detect_anomaly():
                        ce, sl = self.model.compute_loss(x, y, x_unlab, y_unlab)
                        #loss = (x.size(0) * ce + x_unlab.size(0) * sl) / (x.size(0) + x_unlab.size(0))
                        loss = ce + w_s_weight * sl
                        #loss = torch.add(ce, sl)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                        optimizer.step()
                else:
                    ce, sl = self.model.compute_loss(x, y, x_unlab, y_unlab)
                    #loss = (x.size(0) * ce + x_unlab.size(0) * sl) / (x.size(0) + x_unlab.size(0))
                    loss = ce + w_s_weight * sl

            # statistics
            running_loss       += loss.item() * (len(x) + len(x_unlab))  # TODO: change?
            running_loss_lab   += ce.item() * (len(x) + len(x_unlab))
            running_loss_unlab += float(sl) * (len(x) + len(x_unlab))
            running_n          += (len(x) + len(x_unlab))

            if batch_idx % 50 == 0:
                log.info("\t[{}/{}] Batch {}/{}: lab Loss: {:.4f} Unlab Loss: {:.4f}".format(phase,
                                                                                             epoch,
                                                                                             batch_idx,
                                                                                             n_batches,
                                                                                             running_loss_lab / running_n,
                                                                                             running_loss_unlab / running_n))

        epoch_loss = running_loss / self.dataset_sizes[phase]

        log.info("Computing scores")
        y_true, y_pred = gather_outputs(
            self.model.forward, self.dataloaders[phase])

        scores = {
            "accuracy": Multilabel.accuracy_score(y_true, y_pred)
        }

        log.info("{} Scores: {}".format(phase, scores))

        log.info('{} Loss: {:.4f}'.format(
            phase, epoch_loss))

        return epoch_loss

    def train(self, num_epochs):

        root_path = self._get_save_path()

        model_path = os.path.join(root_path, "best_model.pkl")

        device = torch.device(self.device)

        self.model = self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        anneal_rate = 1. / num_epochs
        w_s_weight  = 0.

        since = time.time()

        for epoch in range(1, num_epochs + 1):
            log.info('Epoch {}/{}'.format(epoch, num_epochs))

            train_loss = self.run_epoch(epoch, "training", device, optimizer, w_s_weight)

            if math.isnan(train_loss):
                raise ValueError("NaN loss encountered")

            val_loss = self.run_epoch(epoch, "val", device, None, w_s_weight)

            if math.isnan(val_loss):
                raise ValueError("NaN loss encountered")

            w_s_weight = min(1., anneal_rate + w_s_weight)

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
