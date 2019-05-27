import logging
from collections import OrderedDict

import torch
from torch import nn

log = logging.getLogger(__name__)


class Flatten(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)


class PrintShape(nn.Module):
    def forward(self, x):
        print(x.size())
        return x


def get_layers(n_classes):
    layers = OrderedDict()
    layers["flatten"] = Flatten()
    layers["fc_1"] = nn.Linear(28 * 28, 512)
    layers["relu_1"] = nn.ReLU(inplace=True)

    layers["fc_2"] = nn.Linear(512, 128)
    layers["relu_2"] = nn.ReLU(inplace=True)

    layers["fc_3"] = nn.Linear(128, n_classes)
    layers["log_sf"] = nn.LogSoftmax(dim=1)
    return nn.Sequential(layers)


class SemanticLossModule(nn.Module):
    def __init__(self, device, n_classes, args):
        super().__init__()
        self.device = device
        self.layers = get_layers(n_classes)

        log.info("Classifier: {}".format(self.layers))
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        return self.layers(x)

    def compute_loss(self, x, y, x_unlab, u_unlab):
        out = self(x)
        loss = self.criterion(out, y)
        return loss
