

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
