

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
