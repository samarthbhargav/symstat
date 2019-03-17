# -*- coding: utf-8 -*-
from __future__ import print_function as print_future
from __future__ import unicode_literals

import os
import random
import numpy as np

from collections import defaultdict

from constants   import *


def sco(object):
    return SCO.get(object, 0)


def make_context(jumps):
    if jumps is 0:
        return []
    else:
        positives = [[i, jumps - i] for i in range(jumps + 1)]
        left_negatives = [[offset[0], offset[1] * -1] for offset in positives if offset[1] is not 0]
        right_negatives = [[offset[0] * -1, offset[1]] for offset in positives if offset[0] is not 0]
        double_negatives = [[offset[0] * -1, offset[1] * -1] for offset in positives if offset[0] is not 0 and offset[1] is not 0]
        return make_context(jumps - 1) + positives + left_negatives + right_negatives + double_negatives


def display_context(context):
    maximum = max([tup[0] for tup in context])
    size = maximum * 2 + 1
    square = np.zeros((size, size), dtype=np.int_)

    middle = np.asarray([maximum, maximum], dtype=np.int_)

    for offset in context:
        square[tuple(middle + offset)] = 1

    print(square)



def unpy(numpy_array):
    return [int(long) for long in numpy_array.tolist()]


def avg(list):
    return sum(list) / float(len(list))


def CON(context):
    if type(context) is int:
        return "CONTEXT-{}".format(context)
    else:
        return "CONTEXT-{}".format(max([item[0] for item in context]) if context is not [] else 0)


def chance(probability):
    return random.random() < probability


def initialize_q():
    return defaultdict(lambda: defaultdict(int))


def clear():
    os.system('clear')
    print("")



def accuracy(positives, negatives):
    return percentage(positives, negatives)


def percentage(positives, negatives):
    if (positives == 0 and negatives == 0):
        return 1
    else:
        return positives / float(negatives + positives)


def update_string(current_epoch, amount_of_epochs, score_per_epoch, rate_per_epoch, print_interval):
    result = ""
    result += "Epoch {}/{}".format(current_epoch, amount_of_epochs) + os.linesep
    if (len(score_per_epoch) > 0):
        result += "Average score: {}".format(sum(score_per_epoch) / float(len(score_per_epoch))) + os.linesep
    if (len(rate_per_epoch) > 0):
        result += "Average rate: {}".format(sum(rate_per_epoch) / float(len(rate_per_epoch))) + os.linesep

    if (len(score_per_epoch[::print_interval]) < 10):
        scores = score_per_epoch[::print_interval]
    else:
        scores = score_per_epoch[::print_interval][-10:]

    if (len(rate_per_epoch[::print_interval]) < 10):
        rates = rate_per_epoch[::print_interval]
    else:
        rates = rate_per_epoch[::print_interval][-10:]

    result += "Scores: {}".format(["%.2f" % member for member in scores]) + os.linesep
    result += "Rates: {}".format(["%.2f" % member for member in rates])

    return result
