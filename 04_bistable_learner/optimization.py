import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.util import load_logs
from bayes_opt.event import Events
from Experiment import Experiment

ASEED = 43

def run_experiment(aplus_mult, aminus_mult, aversion):
    exp = Experiment(seed=ASEED)
    exp.brain.vta.DA_pars['A_plus'] = aplus_mult * exp.brain.vta.DA_pars['weight']
    exp.brain.vta.DA_pars['A_minus'] = aminus_mult * exp.brain.vta.DA_pars['weight']
    av = True if aversion > .5 else False
    success_history = exp.train_brain(n_trials=150, aversion=av, full_io=False)
    return np.sum(success_history[-100:])

# Bounded region of parameter space
pbounds = {
    'aplus_mult': (.005, .1), 
    'aminus_mult': (0, .2),
    'aversion' : (0., 1.)}

optimizer = BayesianOptimization(
    f=run_experiment,
    pbounds=pbounds,
    verbose=2,
    random_state=ASEED+1,
)

load_logs(optimizer, logs=["../../results/optimization/home3.json"]);
load_logs(optimizer, logs=["../../results/optimization/bcf2.json"]);
logger = JSONLogger(path="../../results/optimization/home3.json")
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

"""optimizer.probe(
    params={
        "aminus_mult": 0.13321804261960515, 
        "aplus_mult": 0.05641041016692336, 
        "aversion": 0.0290138244243604},
    lazy=True,
)"""

optimizer.maximize(
    init_points=1,
    n_iter=12,
)






