import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from Experiment import Experiment

ASEED = 42

def run_experiment(aplus_mult, aminus_mult, aversion):
    exp = Experiment(seed=ASEED)
    exp.brain.vta.DA_pars['A_plus'] = aplus_mult * exp.brain.vta.DA_pars['weight']
    exp.brain.vta.DA_pars['A_minus'] = aminus_mult * exp.brain.vta.DA_pars['weight']
    av = True if aversion > .5 else False
    success_history = exp.train_brain(n_trials=300, aversion=av)
    return np.sum(success_history[200:])

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

logger = JSONLogger(path="../../results/optimization/home.json")
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

# configuration used for local_01
optimizer.probe(
    params={
        'aplus_mult': .01, 
        'aminus_mult': .015,
        'aversion' : 1.},
    lazy=True,
)

optimizer.maximize(
    init_points=3,
    n_iter=12,
)






