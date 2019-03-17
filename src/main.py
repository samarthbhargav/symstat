# -*- coding: utf-8 -*-
from __future__ import print_function as print_future
from __future__ import unicode_literals

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

from constants         import *
from plot_stuff        import *
from game              import *


def experiment(context_jumps, size, n_guys, n_agents, n_epochs, n_train_steps, test_interval,
               n_test_games, n_test_steps, gamma=GAMMA, higher_order=False,
               progress=1, preamble=None, name=None, do_plot=True):

    agents = [initialize_q() for agent in range(n_agents)]
    context = make_context(context_jumps)

    progress_modulo = math.floor(n_epochs * progress)

    test_epochs = []
    test_scores = []
    test_accuracies = []

    for epoch in range(0, n_epochs + 1):
        # PRINT PROGRESS
        if (progress != 1 and (progress == 0 or epoch % progress_modulo == 0)):
            clear()
            if preamble: print(preamble)
            print("{}{:3.0f}% ({:0{digits}d}/{})".format("   " if preamble else "", epoch / float(n_epochs) * 100, epoch, n_epochs, digits=len(str(n_epochs))))

        # TRAIN AGENTS
        if (epoch != 0):
            for q in agents:
                play_game(q, context, n_train_steps, size, n_guys, learn=True, gamma=gamma, higher_order=higher_order)

        # TEST
        if (epoch % test_interval == 0):
            agent_scores = []
            agent_accuracies = []

            for q in agents:
                game_scores = []
                game_accuracies = []

                for game in range(n_test_games):
                    (score, accuracy) = play_game(q, context, n_test_steps, size, n_guys, learn=False, gamma=gamma, higher_order=higher_order)
                    game_scores.append(score)
                    game_accuracies.append(accuracy)

                agent_scores.append(avg(game_scores))
                agent_accuracies.append(avg(game_accuracies))

            test_scores.append(avg(agent_scores))
            test_accuracies.append(avg(agent_accuracies))
            test_epochs.append(epoch)

    # OUTPUT TO FILE
    now = datetime.now()
    experiment_name = name if name else 'experiment_{}'.format(now.strftime('%Y-%m-%d_%H-%M-%S'))
    text_file_name = '../results/{}.txt'.format(experiment_name)

    with open(text_file_name, "w") as file:
        output = lambda text: print(text, file=file)
        output("Experiment '{}' on {}".format(experiment_name, now.strftime('%Y %b %d %H:%M:%S')))

        output("")
        output("EXPERIMENT SETUP")
        output("  {}".format('*Higher Order*' if higher_order else '*First Order*'))
        output("  Agents: {}".format(n_agents))
        output("  Epochs: {}".format(n_epochs))
        output("  Training steps p/e: {}".format(n_train_steps))
        output("  Testing: at every {}th epoch".format(test_interval))
        output("  Testing games: {}".format(n_test_games))
        output("  Testing steps p/g {}".format(n_test_steps))

        output("")
        output("PARAMETERS")
        output("  Learning rate: {}".format(LEARNING_RATE))
        output("  Gamma: {}".format(gamma))
        output("  Explore rate: {}".format(EXPLORE_RATE))
        output("  Haste: {}".format(HASTE))

        output("")
        output("GAME SETTINGS")
        output("  Dimensions: {0}x{0}".format(size))
        output("  Bad guys and good guys: both {}".format(n_guys))
        output("  Context jumps: {}".format(context_jumps))

        output("")
        output("RESULTS")
        output("  Final epoch {}: average score = {}, average accuracy = {}".format(test_epochs[-1], test_scores[-1], test_accuracies[-1]))

        output("")
        output("  Test epochs: {}".format(test_epochs))

        output("")
        output("  Test scores: {}".format("[{}]".format(", ".join(["{:.1f}".format(score) for score in test_scores]))))

        output("")
        output("  Test accuracies: {}".format("[{}]".format(", ".join(["{:.1f}%".format(accuracy * 100) for accuracy in test_accuracies]))))

        output("")
        output("  Test scores: {}".format(test_scores))

        output("")
        output("  Test accuracies: {}".format(test_accuracies))

    if (do_plot):
        color = 'r' if higher_order else 'b'
        plot(test_epochs, test_scores, test_accuracies, experiment_name, color=color)

    return (test_epochs, test_scores, test_accuracies)



def full_run():
    EPOCHS = 200
    INTERVAL = 10
    AGENTS = 20
    CONTEXTS = [2, 4, 6, 8]
    GAMMAS = [0.1, 0.3, 0.6, 0.9]

    for context in CONTEXTS:
        for gamma in GAMMAS:
            experiment(context_jumps=context,
                       size=8,
                       n_guys=7,
                       n_agents=AGENTS,
                       n_epochs=EPOCHS,
                       n_train_steps=100,
                       test_interval=INTERVAL,
                       n_test_games=10,
                       n_test_steps=200,
                       progress=0.1,
                       gamma=gamma,
                       preamble="Context: {}. Gamma: {}.".format(context, gamma))


def compare_higher_order():
    GAMMAS = [0.1, 0.3, 0.6, 0.8]
    CONTEXTS = [2, 3, 5, 7]
    agents = 20
    epochs = 100
    interval = 5
    train_steps = 75
    test_steps = train_steps

    for gamma in GAMMAS:
        for context in CONTEXTS:
            series = 'Con{}_Gam{}'.format(context, gamma)

            fo_name = '{}_fo_{}'.format(series, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            (test_epochs, fo_scores, fo_accuracies) = experiment(context_jumps=context,
                                                                 size=8,
                                                                 n_guys=7,
                                                                 n_agents=agents,
                                                                 n_epochs=epochs,
                                                                 n_train_steps=train_steps,
                                                                 test_interval=interval,
                                                                 n_test_games=10,
                                                                 n_test_steps=test_steps,
                                                                 progress=0.1,
                                                                 gamma=gamma,
                                                                 higher_order=False,
                                                                 preamble=series + "_fo",
                                                                 name=fo_name)

            ho_name = '{}_ho_{}'.format(series, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            (test_epochs, ho_scores, ho_accuracies) = experiment(context_jumps=context,
                                                                 size=8,
                                                                 n_guys=7,
                                                                 n_agents=agents,
                                                                 n_epochs=epochs,
                                                                 n_train_steps=train_steps,
                                                                 test_interval=interval,
                                                                 n_test_games=10,
                                                                 n_test_steps=test_steps,
                                                                 progress=0.1,
                                                                 gamma=gamma,
                                                                 higher_order=True,
                                                                 preamble=series + "_ho",
                                                                 name=ho_name)

            end_name = '{}_comp_{}'.format(series, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            plot_compare(epochs=test_epochs,
                         a_scores=fo_scores,
                         a_accuracies=fo_accuracies,
                         a_label='First Order',
                         b_scores=ho_scores,
                         b_accuracies=ho_accuracies,
                         b_label="Higher Order",
                         experiment_name=end_name)


def train(steps_per_epoch, amount_of_epochs, inspection_interval, print_interval, context_jumps, mode):
    score_per_epoch = []
    rate_per_epoch = []
    q = initialize_q()

    current_epoch = 0
    progress_string = ""
    context = make_context(context_jumps)

    while (current_epoch < amount_of_epochs):
        inspected_epoch = inspection_interval is not None and (current_epoch + 1) % inspection_interval == 0
        printing_epoch = inspected_epoch or current_epoch % print_interval == 0

        if (printing_epoch):
            clear()
            progress_string = update_string(current_epoch, amount_of_epochs, score_per_epoch, rate_per_epoch, print_interval)
            print(progress_string)

        field = initialize_game(HEIGHT, WIDTH, N_GOOD_GUYS, N_BAD_GUYS)

        score = 0
        negatives = 0
        positives = 0

        current_time_step = 0

        while (current_time_step < steps_per_epoch):
            (reward, given_string) = time_step(field, q, context, give_string=True)

            if (reward == -1):
                negatives += 1

            if (reward == 1):
                positives += 1

            score += reward

            if (inspected_epoch):
                clear()
                print(display(field, score, percentage(positives, negatives), current_time_step, steps_per_epoch, progress_string))
                print(given_string)
                if input("Type 'skip' to skip or any key to continue: ") == "skip":
                    break

            current_time_step += 1

        score_per_epoch.append(score)
        rate_per_epoch.append(percentage(positives, negatives))
        current_epoch += 1

    progress_string = update_string(current_epoch, amount_of_epochs, score_per_epoch, rate_per_epoch, print_interval)
    print(progress_string)
    output_to_file(progress_string, steps_per_epoch, amount_of_epochs, context, mode, score_per_epoch, rate_per_epoch)



if (len(sys.argv) > 1 and sys.argv[1] == "interactive"):
    interactive()
elif (len(sys.argv) > 1 and sys.argv[1] == "steps"):
    step_by_step(3)
elif (len(sys.argv) > 1 and sys.argv[1] == "experiments"):
    train(steps_per_epoch=100,
          amount_of_epochs=5000,
          inspection_interval=None,
          print_interval=1000,
          context_jumps=2,
          mode="regular")
    train(steps_per_epoch=100,
          amount_of_epochs=5000,
          inspection_interval=None,
          print_interval=1000,
          context_jumps=3,
          mode="regular")
    train(steps_per_epoch=100,
          amount_of_epochs=5000,
          inspection_interval=None,
          print_interval=1000,
          context_jumps=4,
          mode="regular")
else:
    redo_plots()
