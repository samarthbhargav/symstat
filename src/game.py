# -*- coding: utf-8 -*-
from __future__ import print_function as print_future
from __future__ import unicode_literals

import itertools
import math
import operator
import os
import random

import numpy as np

from constants import *
from display   import *


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


def move(field, direction, walls=False):

    if walls:
        # FOR USE WITH WALLS
        player_position = find_player(field)
        new_position = wrap(field, player_position + DIR[direction])
        encountered_object = field[tuple(new_position)]

        if (encountered_object != WALL):
            field[tuple(player_position)] = NOTHING
            field[tuple(new_position)] = PLAYER

        return sco(encountered_object)

    shadow_position = np.argwhere(field == SHADOW)

    if (shadow_position.size):
        field[tuple(shadow_position[0])] = NOTHING

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

    interactions_strings_columns = [interactions_strings[i:i + COLUMNS] for i in range(0, len(interactions_strings), COLUMNS)]

    for row in interactions_strings_columns:
        given_string += "   ".join(row) + os.linesep

    if (give_string):
        return (reward, given_string)
    else:
        return reward


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
