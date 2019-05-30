import logging
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

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
    # layers["log_sf"] = nn.LogSoftmax(dim=1)
    return nn.Sequential(layers)


class SemanticLossModule(nn.Module):
    def __init__(self, device, n_classes, args):
        super().__init__()
        self.device = device
        self.layers = get_layers(n_classes)
        self.n_classes = n_classes
        self.w_s    = args.w_s

        log.info("Classifier: {}".format(self.layers))
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.layers(x)

    def pack_y(self, y):
        yy = torch.zeros(y.size(0), self.n_classes)
        for idx, _ in enumerate(y):
            yy[idx, _] = 1.0
        return yy

    def compute_loss(self, x, y, x_unlab, u_unlab):

        loss = 0.
        sl   = 0.

        if x.size(0) > 0:
            out = self(x)
            loss = self.criterion(out, self.pack_y(y))

        if x_unlab.size(0) > 0:
            out_unlab = self(x_unlab)
            sl = torch.mean(self.sem_loss(torch.sigmoid(out_unlab)))

        return loss, sl
        # return (x.size(0) * loss + x_unlab.size(0) * sl) / (x.size(0) + x_unlab.size(0))
        # return sl

    def sem_loss(self, probs):
        s = torch.zeros(probs.size(1))

        for i, p_i in enumerate(probs):
            s_part = p_i
            for j, p_j in enumerate(probs):
                if i == j:
                    continue
                s_part = torch.mul(s_part, (1 - p_j))
            s = torch.add(s, s_part)

        return -1 * torch.log(s)

    def compute_semantic_loss(self, norm_probs, num_classes=10):
        '''
        Return semantic loss for exactly one constraint
        '''
        semantic_loss = torch.tensor(0.)

        for i in range(num_classes):
            one_situation    = [1.] * num_classes
            one_situation[i] = 0.
            one_situation    = torch.tensor(one_situation)

            semantic_loss   += torch.prod(one_situation - norm_probs)

        return -torch.log(torch.abs(semantic_loss))
