# -*- coding: utf-8 -*-
from __future__ import print_function as print_future
from __future__ import unicode_literals

import numpy as np
import os
from time import sleep
import sys
import subprocess
from collections import defaultdict
import random
import itertools
import operator
import time
from datetime import datetime
import math
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import spline

NOTHING = 0
WALL = 1
PLAYER = 2
BAD_GUY = 3
GOOD_GUY = 4
TEST_GUY = 5
SHADOW = 6

AVA = {
    NOTHING: ' ',
    WALL: '#',
    PLAYER: 'M',
    BAD_GUY: '~',
    GOOD_GUY: 'o',
    TEST_GUY: '?',
    SHADOW: '·'
}

SCO = {
    BAD_GUY: -1,
    GOOD_GUY: 1
}


def sco(object):
    return SCO.get(object, 0)


LEFT = 'a'
RIGHT = 'd'
UP = 'w'
DOWN = 's'

ACTIONS = [LEFT, RIGHT, UP, DOWN]

DIR = {
    LEFT: [0, -1],
    RIGHT: [0, 1],
    UP: [-1, 0],
    DOWN: [1, 0]
}

MOV = {
    LEFT: "le",
    RIGHT: "ri",
    UP: "up",
    DOWN: "do"
}

NARROW = "narrow"
MEDIUM = "medium"
WIDE = "wide"

CONTEXT1 = [[0, -1], [0, 1], [-1, 0], [1, 0]]
CONTEXT2 = CONTEXT1 + [[-1, -1], [1, -1], [-1, 1], [1, 1]]
CONTEXT3 = CONTEXT2 + [[0, -2], [0, 2], [-2, 0], [2, 0]]

CONTEXT = {
    NARROW: CONTEXT1,
    MEDIUM: CONTEXT2,
    WIDE: CONTEXT3
}


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


def CON(context):
    if type(context) is int:
        return "CONTEXT-{}".format(context)
    else:
        return "CONTEXT-{}".format(max([item[0] for item in context]) if context is not [] else 0)


# HEIGHT = 10
# WIDTH = 30
# N_BAD_GUYS = 40
# N_GOOD_GUYS = 40

HEIGHT = 8
WIDTH = 8
N_BAD_GUYS = 7
N_GOOD_GUYS = 7

LEARNING_RATE = 1.0
GAMMA = 0.9
EXPLORE_RATE = 0.1
HASTE = 0


def chance(probability):
    return random.random() < probability


def display(field, score=None, rate=None, time_step=None, until=None, preamble=None):
    string = ""

    if preamble is not None:
        string += preamble + os.linesep + os.linesep

    if time_step is not None and until is not None:
        string += "Time step: {}/{}".format(time_step, until) + os.linesep + os.linesep

    string += '_' * (field.shape[1] + 2) + os.linesep

    height = field.shape[0]
    width = field.shape[1]

    y_range = range(height)
    x_range = range(width)

    WRAP = False
    if (WRAP):
        position = find_player(field)

        y_shift = position[0] - height / 2
        y_range = y_range[y_shift:] + y_range[:y_shift]

        x_shift = position[1] - width / 2
        x_range = x_range[x_shift:] + x_range[:x_shift]

    for y in y_range:
        string += '|'
        for x in x_range:
            string += AVA[field[y, x]]
        string += '|'
        string += os.linesep

    if score is not None:
        string += "{:#^{width}}".format(" SCORE: {} ".format(score), width=WIDTH + 2)
        if rate is not None:
            string += os.linesep

    if rate is not None:
        string += "{:#^{width}}".format(" RATE: {}% ".format(rate * 100), width=WIDTH + 2)

    return string.rstrip() + os.linesep


def display_interaction(interaction):
    return "{}: {}".format(AVA[interaction[0]], interaction[1])


def initialize_q():
    return defaultdict(lambda: defaultdict(int))


def display_interactions(interactions, context=None):
    if context is None:
        context = ""
    else:
        context = CON(context) + ": "

    interaction_strings = [display_interaction(interaction) for interaction in interactions]
    interaction_string = ""

    if (len(interaction_strings) > 1):
        for i in range(len(interaction_strings) - 1):
            interaction_string += interaction_strings[i] + " | "

    if (len(interaction_strings) > 0):
        interaction_string += interaction_strings[-1]

    return "{}{}".format(context, interaction_string)


def clear():
    os.system('clear')
    print("")


def render(field, score=None, rate=None):
    clear()
    print(display(field, score, rate))


def pick_random_spot(field):
    return (np.random.randint(0, field.shape[0]), np.random.randint(0, field.shape[1]))


def insert_objects(field, object, amount):
    while (np.count_nonzero(field == object) < amount):
        candidate_spot = pick_random_spot(field)
        if (field[candidate_spot] == NOTHING):
            field[candidate_spot] = object


def insert_player(field):
    insert_objects(field, PLAYER, 1)


def find_player(field):
    return np.argwhere(field == PLAYER)[0]


def wrap(field, position, walls=False):
    if walls:
        return position
    return [position[i] % maximum for (i, maximum) in enumerate(field.shape)]


# FOR USE WITH WALLS
# def move(field, direction):
#     player_position = find_player(field)
#     new_position = wrap(field, player_position + DIR[direction])
#     encountered_object = field[tuple(new_position)]
#
#     if (encountered_object != WALL):
#         field[tuple(player_position)] = NOTHING
#         field[tuple(new_position)] = PLAYER
#
#     return sco(encountered_object)

def move(field, direction):
    shadow_position = np.argwhere(field == SHADOW)
    if (shadow_position.size): field[tuple(shadow_position[0])] = NOTHING

    player_position = find_player(field)
    new_position = wrap(field, player_position + DIR[direction])
    encountered_object = field[tuple(new_position)]
    field[tuple(player_position)] = SHADOW
    field[tuple(new_position)] = PLAYER
    return sco(encountered_object)


def initialize_game(height, width, n_good_guys, n_bad_guys, walls=False):
    field = np.zeros((height, width), dtype=np.int_)
    if walls:
        field[:, 0], field[0, :], field[:, WIDTH - 1], field[HEIGHT - 1, :] = [WALL] * 4
    insert_player(field)
    insert_objects(field, GOOD_GUY, N_GOOD_GUYS)
    insert_objects(field, BAD_GUY, N_BAD_GUYS)
    return field


def spot_interactions(field, context, walls=False):
    player_position = find_player(field)
    interactions = []
    for offset in context:
        try:
            observed_object = field[tuple(wrap(field, player_position + offset, walls))]
            if (observed_object != NOTHING and observed_object != WALL and observed_object != SHADOW):
                interactions.append([observed_object, offset])
        except IndexError:
            pass

    return interactions


def make_combinations(interactions):
    combinations = []

    for (left, right) in itertools.combinations(interactions, 2):
        sortd = sorted([left, right], key=lambda inter: (inter[0], inter[1][0], inter[1][1]))
        combinations.append([[sortd[0][0], sortd[1][0]],
                             [sortd[0][1][0], sortd[0][1][1], sortd[1][1][0], sortd[1][1][1]]])

    return combinations


def interactive(context_jumps=[1, 2, 3], walls=False):
    field = initialize_game(HEIGHT, WIDTH, N_GOOD_GUYS, N_BAD_GUYS, walls)

    score = 0
    render(field, score)
    q = initialize_q()

    jumps_and_contexts = [(jumps, make_context(jumps)) for jumps in context_jumps]

    while 1:
        inp = input("Use WASD keys to move: ")

        if (inp not in ACTIONS):
            break;

        score += move(field, inp)
        render(field, score)

        print("Move: {}".format(MOV[inp]))
        print("")
        for jumps, context in jumps_and_contexts:
            print(display_interactions(spot_interactions(field, context, walls), CON(jumps)))
        print("")
        # time_step(field, q, CONTEXT3)


def percentage(positives, negatives):
    if (positives == 0 and negatives == 0):
        return 1
    else:
        return positives / float(negatives + positives)


def accuracy(positives, negatives):
    return percentage(positives, negatives)


def step_by_step(context_jumps=2, higher_order=True):
    field = initialize_game(HEIGHT, WIDTH, N_GOOD_GUYS, N_BAD_GUYS)

    context = make_context(context_jumps)

    score = 0
    negatives = 0
    positives = 0
    steps = 0

    q = initialize_q()

    clear()
    print("Step {}".format(steps))
    print(display(field, score, percentage(positives, negatives)))

    while 1:
        inp = input("Press q to quit, n for new field, or any key to continue: ")
        if (inp == 'q'):
            break;

        if (inp == 'n'):
            score = 0
            negatives = 0
            positives = 0
            steps = 0
            field = initialize_game(HEIGHT, WIDTH, N_GOOD_GUYS, N_BAD_GUYS)
            clear()
            print("Step {}".format(steps))
            print(display(field, score, percentage(positives, negatives)))
            continue

        (reward, given_string) = time_step(field, q, context, give_string=True, higher_order=higher_order)

        if (reward == -1):
            negatives += 1

        if (reward == 1):
            positives += 1

        score += reward
        steps += 1

        clear()
        print("Step {}".format(steps))
        print(display(field, score, percentage(positives, negatives)))
        print(given_string)


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


def output_to_file(progress_string, steps_per_epoch, amount_of_epochs, context, mode, score_per_epoch, rate_per_epoch):
    tim = time.time()
    file_name_values = 'values_{}.txt'.format(tim)
    file_name_plot = 'plot_{}.png'.format(tim)

    with open(file_name_values, "w") as text_file:
        text_file.write("EXPERIMENT ON {}".format(tim))
        text_file.write(os.linesep)

        text_file.write("Amount of epochs: {}".format(amount_of_epochs))
        text_file.write(os.linesep)

        text_file.write("Steps per epoch: {}".format(steps_per_epoch))
        text_file.write(os.linesep)

        text_file.write("Context: {}".format(CON(context)))
        text_file.write(os.linesep)

        text_file.write("Room height: {}".format(HEIGHT))
        text_file.write(os.linesep)

        text_file.write("Room width: {}".format(WIDTH))
        text_file.write(os.linesep)

        text_file.write("Bad guys: {} ({})".format(N_BAD_GUYS, SCO[BAD_GUY]))
        text_file.write(os.linesep)

        text_file.write("Good guys: {} ({})".format(N_GOOD_GUYS, SCO[GOOD_GUY]))
        text_file.write(os.linesep)

        text_file.write("Learning rate: {}".format(LEARNING_RATE))
        text_file.write(os.linesep)

        text_file.write("Explore rate: {}".format(EXPLORE_RATE))
        text_file.write(os.linesep)
        text_file.write("Gamma: {}".format(GAMMA))
        text_file.write(os.linesep)
        text_file.write("Mode: {}".format(mode))
        text_file.write(os.linesep)
        text_file.write("RESULTS")
        text_file.write(os.linesep)
        text_file.write(progress_string)
        text_file.write(os.linesep)
        text_file.write("FULL DATA")
        text_file.write(os.linesep)
        text_file.write("All scores: {}".format(score_per_epoch))
        text_file.write(os.linesep)
        text_file.write("All rates: {}".format(rate_per_epoch))

    x = np.arange(0, len(score_per_epoch))

    # fig2 = matplotlib.pyplot.figure(figsize=(8.0, 5.0))
    fig, ax1 = plt.subplots(figsize=(15, 15))

    ax2 = ax1.twinx()
    # ax1.plot(x, score_per_epoch, 'g-')
    # ax2.plot(x, rate_per_epoch, 'b-')

    xnew = np.linspace(0, len(score_per_epoch), 50)  # 300 represents number of points to make between T.min and T.max
    score_smooth = spline(x, score_per_epoch, xnew)
    rate_smooth = spline(x, rate_per_epoch, xnew)

    ax1.plot(xnew, score_smooth, 'g-')
    ax2.plot(xnew, rate_smooth, 'b-')

    ax1.set_ylim(0, 20)
    ax2.set_ylim(0, 1.2)

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Scores', color='g')
    ax2.set_ylabel('Rates', color='b')

    plt.suptitle('Mode: {}. Context: {}. Learning rate: {}. Explore rate: {}. Gamma: {}. Room: {}x{}. Good: {}. Bad:{}.'
                 .format(mode, context, LEARNING_RATE, EXPLORE_RATE, GAMMA, HEIGHT, WIDTH, N_GOOD_GUYS, N_BAD_GUYS), fontsize=16)

    plt.savefig(file_name_plot)
    # plt.show()


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


def unpy(numpy_array):
    return [int(long) for long in numpy_array.tolist()]


def play_game(q, context, steps, size, n_guys, learn, gamma, higher_order):
    field = initialize_game(size, size, n_guys, n_guys)

    negatives = 0
    positives = 0

    for step in range(steps):
        reward = time_step(field, q, context, learn=learn, gamma=gamma, higher_order=higher_order)
        if (reward == -1):
            negatives += 1
        elif (reward == 1):
            positives += 1

    return (positives - negatives, accuracy(positives, negatives))


def avg(list):
    return sum(list) / float(len(list))


def time_step(field, q, context, verbose=False, give_string=False, walls=False, learn=True, higher_order=False, gamma=GAMMA):
    interactions = spot_interactions(field, context, walls)
    combinations = make_combinations(interactions) if higher_order else []

    do_text = verbose or give_string
    given_string = ""

    if (chance(EXPLORE_RATE) or not interactions):
        action = random.choice(ACTIONS)
        if (do_text):
            str = "Random move: {}".format(MOV[action]) + os.linesep
            if (verbose):
                print(str)
            if (give_string):
                given_string += str

    else:
        action_ratings = {}

        for candidate_action in ACTIONS:
            q_expectations = [q[interaction[0]][tuple(interaction[1] + [candidate_action])] for interaction in interactions]
            q_higher_order_expectations = [q[tuple(combination[0])][tuple(combination[1] + [candidate_action])] for combination in combinations]
            action_ratings[candidate_action] = sum(q_expectations + q_higher_order_expectations)

        action = max(action_ratings.items(), key=operator.itemgetter(1))[0]

        if (do_text):
            str = "Specific move: {}".format(MOV[action]) + os.linesep + os.linesep
            str += "{}: {}, {}: {}, {}: {}, {}: {}".format(
                MOV[LEFT], action_ratings[LEFT],
                MOV[RIGHT], action_ratings[RIGHT],
                MOV[UP], action_ratings[UP],
                MOV[DOWN], action_ratings[DOWN]
            )
            str += os.linesep
            if (verbose):
                print(str)
            if (give_string):
                given_string += str + os.linesep

    reward = move(field, action)

    if not learn:
        return reward

    interactions_strings = []

    for interaction in interactions:
        max_future_reward = 0
        new_interaction_offset = [interaction[1][0] - DIR[action][0], interaction[1][1] - DIR[action][1]]

        if (new_interaction_offset in context):
            max_future_reward = max([q[interaction[0]][tuple(new_interaction_offset + [candidate_action])] for candidate_action in ACTIONS])

        current_value = q[interaction[0]][tuple(interaction[1] + [action])]
        new_value = (1 - LEARNING_RATE) * current_value + LEARNING_RATE * (reward + gamma * max_future_reward)
        q[interaction[0]][tuple(interaction[1] + [action])] = new_value

        if (do_text):
            str = "Q[{}|{}|({:2d},{:2d})]:         {:+.0f} (now {:.0f})".format(AVA[interaction[0]], MOV[action], interaction[1][0], interaction[1][1], (new_value - current_value) * 100, new_value * 100)
            if (verbose):
                print(str)
            if (give_string):
                interactions_strings.append(str)

    for combination in combinations:
        max_future_reward = 0
        new_combination_offsets = [combination[1][0] - DIR[action][0],
                                   combination[1][1] - DIR[action][1],
                                   combination[1][2] - DIR[action][0],
                                   combination[1][3] - DIR[action][1]]

        if ([new_combination_offsets[0], new_combination_offsets[1]] in context
                and [new_combination_offsets[2], new_combination_offsets[3]] in context):
            max_future_reward = max([q[tuple(combination[0])][tuple(new_combination_offsets + [candidate_action])] for candidate_action in ACTIONS])

        current_value = q[tuple(combination[0])][tuple(combination[1] + [action])]
        new_value = (1 - LEARNING_RATE) * current_value + LEARNING_RATE * (reward + gamma * max_future_reward)
        q[tuple(combination[0])][tuple(combination[1] + [action])] = new_value

        if (do_text):
            str = "Q[{}{}|{}|({:2d},{:2d})({:2d},{:2d})]: {:+.0f} (now {:.0f})".format(
                AVA[combination[0][0]],
                AVA[combination[0][1]],
                MOV[action],
                combination[1][0], combination[1][1],
                combination[1][2], combination[1][3],
                (new_value - current_value) * 100, new_value * 100)
            if (verbose):
                print(str)
            if (give_string):
                interactions_strings.append(str)

    COLUMNS = 3
    interactions_strings_columns = [interactions_strings[i:i + COLUMNS] for i in range(0, len(interactions_strings), COLUMNS)]

    for row in interactions_strings_columns:
        given_string += "   ".join(row) + os.linesep

    if (give_string):
        return (reward, given_string)
    else:
        return reward


def experiment(context_jumps, size, n_guys, n_agents, n_epochs, n_train_steps, test_interval, n_test_games, n_test_steps,

               gamma=GAMMA, higher_order=False,

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
    text_file_name = '{}.txt'.format(experiment_name)

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

    plt.savefig("{}.png".format(experiment_name))
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

    plt.savefig("{}.png".format(experiment_name))
    if (show): plt.show()


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
    # compare_higher_order()

    redo_plots()

    # test_plot_compare()

    # step_by_step(4)

    # experiment(context_jumps=2,
    #            size=8,
    #            n_guys=7,
    #            n_agents=1,
    #            n_epochs=500,
    #            n_train_steps=100,
    #            test_interval=10,
    #            n_test_games=10,
    #            n_test_steps=200,
    #            progress=0.1,
    #            higher_order=True)
