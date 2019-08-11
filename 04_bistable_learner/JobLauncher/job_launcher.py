import numpy as np
import 
from typing import Dict
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.util import load_logs
from bayes_opt.event import Events
from bayes_opt import UtilityFunction
#from Experiment import Experiment


class Job():
    def __init__(self, local_id:int, pars:Dict[str, float]):
        self.local_id = local_id
        self.pars = pars
        self.nemo_id = None
        self.status = None



class JobLauncher():
    def __init__(self, *args, **kwargs):
        self.jobs = list()
        self.max_running_jobs = 4
        self.optimizer = BayesianOptimization(
            f=None,
            pbounds={'x': (-2, 2), 'y': (-3, 3)},
            verbose=2,
            random_state=1)
        self.utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    @property
    def n_jobs(self):
        return len(self.jobs)

    def launch_new_job(self):
        new_job = Job(
            local_id=self.n_jobs,
            pars=self.optimizer.suggest(self.utility)
        )

    def create_batchjob(self, job):
        with fo=open()





    
    
    







# Let's start by definying our function, bounds, and instanciating an optimization object.
def black_box_function(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1






for _ in range(2):
    next_point = optimizer.suggest(utility)
    target = black_box_function(**next_point)
    optimizer.register(params=next_point, target=target)
    print(target, next_point)

print('max', optimizer.max)


for i in range(10):
    next_point_to_probe = optimizer.suggest(utility)
    print("Next point to probe is:", next_point_to_probe)
