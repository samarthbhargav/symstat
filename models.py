import logging
from collections import OrderedDict

import torch
from torch import nn

log = logging.getLogger(__name__)


class SemanticLossModule(nn.Module):
    def __init__(self, device, n_classes, vocabulary, args, embedding_dim=200):
        super().__init__()
        self.device = device
        self.vocabulary = vocabulary
        self.embedding_dim = embedding_dim
        log.info("Creating an embedding matrix with vocab size: {}".format(
            len(vocabulary)))
        self.embedding = nn.Embedding(len(vocabulary), self.embedding_dim)

        layers = OrderedDict()
        layers["fc_1"] = nn.Linear(self.embedding_dim, 128)
        layers["relu_1"] = nn.ReLU(True)
        layers["fc_2"] = nn.Linear(128, 128)
        layers["relu_2"] = nn.ReLU(True)
        layers["fc_3"] = nn.Linear(128, n_classes)
        self.layers = nn.Sequential(layers)

    def forward(self, x):
        x = torch.LongTensor(x)
        x = self.embedding(x)
        # perform MoT pooling
        x, _ = x.max(dim=0)
        return self.layers(x)

    def compute_loss(self, text, labels):
        criterion = nn.BCEWithLogitsLoss()
        labels = torch.FloatTensor(labels)
        out = self(text)
        # warn: this only works for a single sample
        loss = criterion(out, labels.view(-1))

        return loss
