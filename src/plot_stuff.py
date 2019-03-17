import itertools
import math
import matplotlib
import operator
import os
import random
import subprocess
import sys
import time

import numpy             as np
import matplotlib.pyplot as plt

from time              import sleep
from collections       import defaultdict
from datetime          import datetime
from scipy.interpolate import spline



def redo_plots():
    plot_compare(epochs=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
                 a_scores=[0.10499999999999998, 1.75, 1.7849999999999997, 1.625, 1.735, 1.455, 1.25, 1.4200000000000002, 1.12, 1.1800000000000002, 1.03, 0.96, 1.2999999999999998, 1.315, 1.5849999999999997, 1.2499999999999998, 1.275, 1.38, 1.23, 1.4049999999999998, 1.2699999999999998],
                 a_accuracies=[0.5140535714285714, 0.6691607142857142, 0.6913762626262626, 0.7096935425685426, 0.745440476190476, 0.7692777777777777, 0.7314761904761904, 0.7582857142857142, 0.7082559523809524, 0.7366130952380953, 0.6919464285714287, 0.7316626984126984, 0.7535059523809523, 0.7556210317460319, 0.7497777777777778, 0.7301666666666666, 0.7309801587301588, 0.7612519841269842, 0.7361071428571427, 0.7812559523809525, 0.7298690476190476],
                 a_label='First Order',
                 b_scores=[0.185, 3.53, 4.110000000000001, 4.659999999999999, 4.445, 4.325, 4.559999999999999, 4.5649999999999995, 4.4, 4.1049999999999995, 4.039999999999999, 3.85, 3.7050000000000005, 3.8250000000000006, 3.625, 3.689999999999999, 3.71, 3.3950000000000005, 3.3649999999999998, 3.5149999999999997, 3.185],
                 b_accuracies=[0.5320912698412699, 0.7657094155844156, 0.8133306277056278, 0.843648088023088, 0.8355270562770564, 0.8421751443001444, 0.8490972222222222, 0.8745290404040403, 0.8783948412698412, 0.8706448412698414, 0.848988095238095, 0.857313492063492, 0.8352393578643579, 0.8579801587301586, 0.8406686507936509, 0.8522638888888888, 0.8762083333333335, 0.8502175324675326, 0.8460297619047619, 0.8608412698412697, 0.8539464285714287],
                 b_label="Higher Order",
                 experiment_name='Con7_Gam0.8_comp_2018-04-13_08-41-06')

def test_plot_compare():
    plot_compare(epochs=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
                 a_scores=[-0.18, 3.5700000000000003, 3.65, 3.6799999999999997, 3.78, 3.84, 3.2599999999999993, 3.54, 3.2400000000000007, 3.34, 3.3300000000000005, 3.1799999999999997, 3.41, 3.2700000000000005, 3.3299999999999996, 3.69, 3.6300000000000003, 3.3900000000000006, 3.2399999999999998, 3.5700000000000003, 3.9099999999999993],
                 a_accuracies=[0.5017622655122655, 0.8318239538239538, 0.8546547619047619, 0.8572857142857142, 0.8075357142857144, 0.8401230158730157, 0.8493556998556999, 0.836626984126984, 0.8192352092352092, 0.7974285714285713, 0.8353874458874457, 0.8033477633477635, 0.8252857142857142, 0.8266984126984127, 0.8364152236652236, 0.8455277777777779, 0.8455541125541124, 0.8398769841269841, 0.8365753968253969, 0.8347222222222224, 0.8877788600288602],
                 a_label='First Order',
                 b_scores=[-0.13999999999999999, 4.18, 4.3100000000000005, 4.38, 4.44, 4.970000000000001, 4.7, 4.4, 4.83, 4.73, 4.87, 4.959999999999999, 4.800000000000001, 4.84, 4.24, 4.55, 4.790000000000001, 4.829999999999999, 4.5, 4.76, 4.5],
                 b_accuracies=[0.4839545454545456, 0.8591572871572872, 0.8348080808080807, 0.8784628427128428, 0.8887261904761905, 0.907111111111111, 0.8910238095238094, 0.8811547619047617, 0.8754390331890332, 0.9018373015873017, 0.8814325396825398, 0.8992023809523811, 0.8906785714285712, 0.8958398268398268, 0.862047619047619, 0.8891507936507936, 0.8850396825396825, 0.8996309523809524, 0.8516071428571428, 0.8939761904761905, 0.8749325396825396],
                 b_label="Higher Order",
                 experiment_name='HOho_fo_comparison_2018-04-13_04-04-53')


def test_plot():
    plot(epochs=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
         scores=[-0.059999999999999984, 2.4099999999999997, 2.2950000000000004, 2.58, 2.5050000000000003, 2.2700000000000005, 2.3600000000000003, 2.5050000000000003, 2.5550000000000006, 2.5249999999999995, 2.51, 2.4449999999999994, 2.2950000000000004, 2.38, 2.3299999999999996, 2.2849999999999997, 2.4999999999999996, 2.585, 2.0399999999999996, 2.22, 2.74],
         accuracies=[0.49536392773892757, 0.6731928210678213, 0.664515748140748, 0.6928562964812964, 0.6848259657009654, 0.658162518037518, 0.6744926046176045, 0.6780985125985126, 0.6856897824397824, 0.7012778610278609, 0.6806134837384838, 0.6837360833610833, 0.6645035103785102, 0.6615000555000554, 0.679305375180375, 0.6837449772449772, 0.6850644216894216, 0.6848250360750361, 0.6559255605505605, 0.6608497474747475, 0.692664862914863],
         experiment_name='experiment_2018-04-13_03-14-39',
         color='r')


def plot(epochs, scores, accuracies, experiment_name, show=False, color='k'):
    figure, score_axis = plt.subplots()
    score_axis.set_xlabel('epoch')

    score_axis.set_ylabel('average score')
    score_axis.plot(epochs, scores, linewidth=2.25, color=color)
    score_axis.tick_params(axis='y')
    score_axis.set_ylim(0, 7)
    # score_axis.set_ylim(top=7)

    accuracy_axis = score_axis.twinx()

    accuracy_axis.set_ylabel('average accuracy')
    accuracy_axis.plot(epochs, accuracies, alpha=0.5, linestyle='dashed', color=color)
    accuracy_axis.tick_params(axis='y')
    accuracy_axis.set_ylim(0.5, 1)
    # accuracy_axis.set_ylim(top=1)

    sc = matplotlib.lines.Line2D([], [], color=color, linewidth=2.25, alpha=1.0, linestyle='solid', label='scores')
    ac = matplotlib.lines.Line2D([], [], color=color, linewidth=1, alpha=1.0, linestyle='dashed', label='accuracies')

    plt.legend(handles=[sc, ac], loc='lower right', ncol=1, frameon=False)

    plt.suptitle(experiment_name)
    # figure.tight_layout()

    plt.savefig("../results/{}.png".format(experiment_name))
    if (show): plt.show()


def plot_compare(epochs, a_scores, a_accuracies, a_label, b_scores, b_accuracies, b_label, experiment_name, show=False):
    figure, score_axis = plt.subplots()
    score_axis.set_xlabel('epoch')

    a_color = 'b'
    b_color = 'r'

    score_axis.set_ylabel('average score')
    score_axis.plot(epochs, a_scores, color=a_color, linewidth=2.25)
    score_axis.plot(epochs, b_scores, color=b_color, linewidth=2.25)
    score_axis.tick_params(axis='y')
    score_axis.set_ylim(0, 7)
    # score_axis.set_ylim(top=7)

    accuracy_axis = score_axis.twinx()

    accuracy_axis.set_ylabel('average accuracy')
    accuracy_axis.plot(epochs, a_accuracies, color=a_color, alpha=0.5, linestyle='dashed')
    accuracy_axis.plot(epochs, b_accuracies, color=b_color, alpha=0.5, linestyle='dashed')
    accuracy_axis.tick_params(axis='y')
    accuracy_axis.set_ylim(0.5, 1)
    # accuracy_axis.set_ylim(top=1)

    a_c = matplotlib.patches.Patch(color=a_color, label=a_label)
    b_c = matplotlib.patches.Patch(color=b_color, label=b_label)
    sc = matplotlib.lines.Line2D([], [], color='k', linewidth=2.25, alpha=1.0, linestyle='solid', label='scores')
    ac = matplotlib.lines.Line2D([], [], color='k', linewidth=1, alpha=1.0, linestyle='dashed', label='accuracies')

    # plt.legend(handles=[a_c, b_c, sc, ac], loc='lower right', ncol=1, frameon=False)

    plt.suptitle(experiment_name)
    # figure.tight_layout()

    plt.savefig("../results/{}.png".format(experiment_name))
    if (show): plt.show()
