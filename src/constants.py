NOTHING  = 0
WALL     = 1
PLAYER   = 2
BAD_GUY  = 3
GOOD_GUY = 4
TEST_GUY = 5
SHADOW   = 6

AVA = {
    NOTHING : ' ',
    WALL    : '#',
    PLAYER  : 'M',
    BAD_GUY : '~',
    GOOD_GUY: 'o',
    TEST_GUY: '?',
    SHADOW  : '·'
}

SCO = {
    BAD_GUY : -1,
    GOOD_GUY: 1
}

LEFT  = 'a'
RIGHT = 'd'
UP    = 'w'
DOWN  = 's'

ACTIONS = [LEFT, RIGHT, UP, DOWN]

DIR = {
    LEFT : [0, -1],
    RIGHT: [0, 1],
    UP   : [-1, 0],
    DOWN : [1, 0]
}

MOV = {
    LEFT : "le",
    RIGHT: "ri",
    UP   : "up",
    DOWN : "do"
}

NARROW = "narrow"
MEDIUM = "medium"
WIDE   = "wide"

CONTEXT1 = [[0, -1], [0, 1], [-1, 0], [1, 0]]
CONTEXT2 = CONTEXT1 + [[-1, -1], [1, -1], [-1, 1], [1, 1]]
CONTEXT3 = CONTEXT2 + [[0, -2], [0, 2], [-2, 0], [2, 0]]

CONTEXT = {
    "narrow": CONTEXT1,
    "medium": CONTEXT2,
    "wide"  : CONTEXT3
}


HEIGHT      = 8
WIDTH       = 8
N_BAD_GUYS  = 7
N_GOOD_GUYS = 7

LEARNING_RATE = 1.0
GAMMA         = 0.9
EXPLORE_RATE  = 0.1
HASTE         = 0

COLUMNS = 3
