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
        self.w_s    = args.w_s

        log.info("Classifier: {}".format(self.layers))
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        return self.layers(x)

    def compute_loss(self, x, y, x_unlab, u_unlab):

        if x.size(0) > 0:
            out = self(x)
            loss = self.criterion(out, y)
        else:
            loss = 0.

        if x_unlab.size(0) > 0:
            out_unlab = self(x_unlab)
            sl = self.compute_semantic_loss(out_unlab)
        else:
            sl = 0.

        return loss + self.w_s * sl

    def compute_semantic_loss(self, probs):
        '''
        Return semantic loss for exactly one constraint
        '''
        # num_classes   = probs.size()[0]
        num_classes   = 10

        norm_probs    = torch.sigmoid(probs)
        semantic_loss = torch.tensor(0.)

        for i in range(num_classes):
            one_situation    = [1.] * num_classes
            one_situation[i] = 0.
            one_situation    = torch.tensor(one_situation)

            semantic_loss   += torch.prod(one_situation - norm_probs)

        semantic_loss = torch.clamp(semantic_loss, min=1e-5)

        return -torch.log(torch.abs(semantic_loss))
