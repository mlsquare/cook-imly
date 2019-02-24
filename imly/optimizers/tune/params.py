from ray import tune
from hyperopt import hp
import numpy as np

'''
This file is just a prototyping effort. It can be ignored.
This is an attempt in representing params as a python dict instead of
JSON. It is never imported or used in any other files of IMLY.
Two types of representations are included here,
1) A dict for basic grid search
2) A scan dict for hyperopt
'''


glm_1 = {
    "lr": 2,
    "units": 1,
    "batch_size": 10,
    "epochs": tune.grid_search([100, 200]),
    "optimizer": tune.grid_search(["adam", "nadam"]),
    "losses": "mse",
    "activation": "linear"
}

# Space is specifically for special search algorithms(hyperopt, nevergrid etc)
space = {
    "lr": hp.uniform("lr", 0.001, 0.1),
    "momentum": hp.uniform("momentum", 0.1, 0.9),
    "hidden": hp.choice("hidden", np.arange(16, 256, dtype=int)),
}

# TODO
# New structure
# 1) Pass foundational params from arch/params.json(as JSON)
# 2) The Optimizer would not have any params
# 3) If the user wants to optimize, they can pass
#    the config/scan and other optimization details while
#    fitting.