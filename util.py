import time
from matplotlib import colors
import matplotlib.pyplot as plt
import os
import contextlib
import numpy as np


PROJECT_CWD = "C:\\Users\\OWNER\\PycharmProjects\\GloopNetEvolution"
TIME_FUNCTIONS = True

DEVICE = "cuda:0"
DEBUG = False
STABLE_T=(1000, 2000) if not DEBUG else (10, 20)
RELAXATION_TIME = 4000 if not DEBUG else 40
DEFAULT_GAMMA = 1.5

# T_WS = [20, 40, 160, 640, 900, 1280, 2100, 3000]
T_WS = [20, 40, 160, 640, 900, 1280]
# T_WS = [20, 40]

cdict = {'red':   ((0.0,  0.22, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.89, 1.0)),

         'green': ((0.0,  0.49, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.12, 1.0)),

         'blue':  ((0.0,  0.72, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.11, 1.0))}
cmap = colors.LinearSegmentedColormap('custom', cdict)

def time_func(f):
    def wrap(*args, **qwargs):
        time1 = time.time()
        ret = f(*args, **qwargs)
        time2 = time.time()
        print('function {:s} timed at {:.3f} s'.format(f.__name__, (time2 - time1)))
        return ret

    def no_wrap(*args, **qwargs):
        return f(*args, **qwargs)

    return wrap if TIME_FUNCTIONS else no_wrap

def print_red(*args, **qwargs):
    # See https://en.wikipedia.org/wiki/ANSI_escape_code#3-bit_and_4-bit for a full list of colors
    print('\033[31m',  *args,  '\033[0m', **qwargs)  # 91 is bright red

def save_fig(name, format="png", dpi=300, add_transparent=False, facecolor="w", figure_dump=True):
    name = os.path.join(PROJECT_CWD, "figure_dump", name) if figure_dump else name
    folder = "".join(os.path.split(name)[:-1])
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    plt.savefig(f"{name}.{format}", dpi=dpi, transparent=False, bbox_inches="tight", facecolor=facecolor)
    if format == "png" and add_transparent:
        plt.savefig(f"{name}_transparent.{format}", dpi=dpi, transparent=True, bbox_inches="tight")
    print(f"fig saved to {name}.{format}")
    plt.close()




@contextlib.contextmanager
def np_temp_seed(temp_seed):
    if temp_seed is None:
        yield
    else:
        original_state = np.random.get_state()
        np.random.seed(temp_seed)
        try:
            yield
        finally:
            np.random.set_state(original_state)
