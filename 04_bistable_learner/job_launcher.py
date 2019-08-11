import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.util import load_logs
from bayes_opt.event import Events
#from Experiment import Experiment


class job():
    def __init__(self, *args, **kwargs):
        self.local_id = None
        self.nemo_id = None
        self.pars = dict()
        self.status = None
    
    



from bayes_opt import UtilityFunction




# Let's start by definying our function, bounds, and instanciating an optimization object.
def black_box_function(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1

optimizer = BayesianOptimization(
    f=None,
    pbounds={'x': (-2, 2), 'y': (-3, 3)},
    verbose=2,
    random_state=1,
)

utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

for i in range(10):
    next_point_to_probe = optimizer.suggest(utility)
    print("Next point to probe is:", next_point_to_probe)

for _ in range(2):
    next_point = optimizer.suggest(utility)
    target = black_box_function(**next_point)
    optimizer.register(params=next_point, target=target)
    print(target, next_point)

print('max', optimizer.max)


for i in range(10):
    next_point_to_probe = optimizer.suggest(utility)
    print("Next point to probe is:", next_point_to_probe)
