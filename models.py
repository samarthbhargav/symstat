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

        print("---")
        print(n_classes)

        log.info("Classifier: {}".format(self.layers))
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.layers(x)

    def pack_y(self, y):
        yy = torch.zeros(y.size(0), self.n_classes)
        for idx, _ in enumerate(y):

            _ = _.type(torch.LongTensor)

            # idx = torch.tensor(idx).type(torch.LongTensor)
            yy[idx, _] = 1.0
        return yy

    def compute_loss(self, x, y, x_unlab, y_unlab, hierarchy):

        loss = 0.
        sl   = 0.

        if x.size(0) > 0:
            out    = self(x)
            ce_lab = self.criterion(out, self.pack_y(y))
            sl_lab = torch.mean(self.sem_loss_hierarchy(torch.sigmoid(out), hierarchy))

            loss   = torch.add(loss, ce_lab)
            sl     = torch.add(sl, sl_lab)

        if x_unlab.size(0) > 0:
            out_unlab = self(x_unlab)
            sl_unlab  = torch.mean(self.sem_loss_hierarchy(torch.sigmoid(out_unlab), hierarchy))

            sl        = torch.add(sl, sl_unlab)

        return loss, sl

    def sem_loss(self, probs, hierarchy=None):

        s = torch.zeros(probs.size(0))

        probs = probs.permute(1, 0)

        for i, p_i in enumerate(probs):
            s_part = torch.log(p_i + 1e-9)

            for j, p_j in enumerate(probs):
                if i == j:
                    continue

                s_part = torch.add(s_part, torch.log(1 - p_j + 1e-9))
                s_part = torch.clamp(s_part, max=5)
            s = torch.add(s, torch.exp(s_part))

        return -1 * torch.log(s)

    def sem_loss_hierarchy(self, p_probs, hierarchy):

        s_batch = torch.zeros(p_probs.size(0))

        for j, probs in enumerate(p_probs):
            s_1 = torch.zeros(1)
            s_2 = torch.zeros(1)

            for i, p_i in enumerate(probs):
                # get 'positive' locs - we want to maximize these probs
                assoc = hierarchy.assoc_idx[i]
                # get 'negative' locs - we want to minimuize these probs
                neg_assoc = hierarchy.neg_assoc_idx[i]

                # _s_1 = probs[assoc].prod()
                # _s_2 = (1 - probs[neg_assoc]).prod()

                _s_1 = torch.sum(torch.log(probs[assoc] + 1e-9))
                _s_1 = torch.clamp(_s_1, max=5)
                _s_2 = torch.sum(torch.log(1 - probs[neg_assoc] + 1e-9))
                _s_2 = torch.clamp(_s_2, max=5)

                s_1 = torch.add(s_1, torch.exp(_s_1))
                s_2 = torch.add(s_2, torch.exp(_s_2))

                s_1 = torch.clamp(s_1, max=1)
                s_2 = torch.clamp(s_2, max=1)

                # s_1 = s_1 + _s_1
                # s_2 = s_2 + _s_2

            s_batch[j] = - torch.log(s_1) - torch.log(s_2)

        return s_batch.mean()
