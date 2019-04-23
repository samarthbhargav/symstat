import os
import time
import logging

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from vocabulary import Vocabulary
from models import SemanticLossModule
from dataloader import ReutersDataset, ReutersDatasetIterator
from sklearn.metrics import f1_score, precision_score, recall_score

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


def gather_outputs(forward_func, loader, threshold=0.5):
    y_true = []
    y_pred = []
    log.info("Gathering outputs")
    with torch.no_grad():
        for index, (_id, labels, text, _,  _, _) in enumerate(loader):
            output = torch.sigmoid(forward_func(text))
            output[output >= threshold] = 1
            output[output < threshold] = 0
            y_pred.append(output.cpu().view(-1).numpy())
            y_true.append(labels.cpu().view(-1).numpy())

            if (index + 1) % 1000 == 0:
                log.info("Eval loop: {} done".format(index + 1))

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return y_true, y_pred


class Trainer(object):

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.device = torch.device(args.device)
        # self.dataset = args.dataset
        self.model_type = args.model
        assert self.model_type in {"sl"}
        self.model_id = args.model_id
        self.learning_rate = args.learning_rate

        # load data
        self.dataset_sizes = {}
        self.datasets = {}
        self.dataloaders = {}
        self._load_data(args)

        log.info("Device: {}".format(self.device))

        # load model
        self.model = None
        self._create_model(args)

    def _get_save_path(self):
        return os.path.join("results", self.model_id)

    def _load_data(self, args):
        self.vocabulary = Vocabulary(True, 5, True, "./reuters/stopwords")
        train_iter = ReutersDatasetIterator("reuters", "training")
        self.vocabulary.build(train_iter)
        for split in {"training", "test"}:
            self.datasets[split] = ReutersDataset(
                "reuters", split, self.vocabulary)
            self.dataset_sizes[split] = len(self.datasets[split])
            # the code only supports batch-size = 1 at the moment
            self.dataloaders[split] = DataLoader(self.datasets[split],
                                                 batch_size=1, shuffle=True, num_workers=self.num_workers)

        self.n_classes = self.datasets["training"].n_classes

    def _create_model(self, args):
        if self.model_type == "sl":
            self.model = SemanticLossModule(
                self.device, self.n_classes, self.vocabulary, args)
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
        for batch_idx, (_id, labels, text, _,  _, _) in enumerate(self.dataloaders[phase], 1):

            # id_doc = id_doc.to(device)

            if phase == "training":
                # zero the parameter gradients
                optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'training'):
                loss = self.model.compute_loss(text, labels)

                # backward + optimize only if in training phase
                if phase == 'training':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * len(text)
            running_n += len(text)
            if batch_idx % 50 == 0:
                log.info("\t[{}/{}] Batch {}/{}: Loss: {:.4f}".format(phase,
                                                                      epoch,
                                                                      batch_idx,
                                                                      n_batches,
                                                                      running_loss / running_n))

        epoch_loss = running_loss / self.dataset_sizes[phase]

        log.info("Computing scores")
        y_true, y_pred = gather_outputs(
            self.model.forward, self.dataloaders[phase])

        scores = {
            "f1": Multilabel.f1_score(y_true, y_pred),
            "recall": Multilabel.recall_score(y_true, y_pred),
            "precision": Multilabel.precision_score(y_true, y_pred)
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
