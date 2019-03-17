from utils import *


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


def render(field, score=None, rate=None):
    clear()
    print(display(field, score, rate))


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


def output_to_file(progress_string, steps_per_epoch, amount_of_epochs, context, mode, score_per_epoch, rate_per_epoch):
    tim = time.time()
    file_name_values = '../results/values_{}.txt'.format(tim)
    file_name_plot = '../results/plot_{}.png'.format(tim)

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
