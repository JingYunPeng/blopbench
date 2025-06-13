import os
import random

import numpy as np

__DEBUG_MODE = False
__CUR_SEED = 42


def debug_print(*args, **kwargs):
    if __DEBUG_MODE:
        print(*args, **kwargs)


def is_debug_mode():
    return __DEBUG_MODE


def set_debug_mode(debug):
    global __DEBUG_MODE
    __DEBUG_MODE = debug


assert_seeds = [864, 137, 978, 243, 241, 637, 812, 331, 232, 474, 585, 463, 485, 265, 109, 988, 370, 534, 185, 693, 927,
                168, 981, 947, 729, 386, 765, 664, 115, 561, 552, 12, 79, 584, 541, 561, 336, 698, 654, 214, 469, 390,
                654, 39, 418, 278, 909, 360, 561, 68, 509, 249, 274, 631, 935, 92, 989, 43, 594, 229, 315, 506, 950,
                455, 487, 424, 72, 76, 759, 700, 931, 331, 75, 286, 880, 461, 379, 818, 834, 148, 278, 518, 148, 508,
                749, 204, 810, 149, 407, 83, 208, 85, 431, 934, 559, 778, 379, 199, 364, 413]


def seed_every_where(seed):
    global __CUR_SEED
    __CUR_SEED = seed

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if seed < 100:
        assert random.randint(0, 1000) == assert_seeds[seed]


def cur_seed():
    return __CUR_SEED
