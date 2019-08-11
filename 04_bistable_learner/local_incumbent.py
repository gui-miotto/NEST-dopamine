from Experiment import Experiment
import numpy as np


exp = Experiment()
exp.brain.vta.DA_pars['A_plus'] = .1 * exp.brain.vta.DA_pars['weight']
exp.brain.vta.DA_pars['A_minus'] = .15 * exp.brain.vta.DA_pars['weight']
save_dir = '../../results/local_incumbent'
success_history = exp.train_brain(n_trials=150, aversion=False, save_dir=save_dir)
success_history = exp.train_brain(n_trials=100, baseline_only=True, save_dir=save_dir)
success_history = exp.train_brain(n_trials=150, aversion=False, rev_learn=True, save_dir=save_dir)
success_history = exp.train_brain(n_trials=100, rev_learn=True, baseline_only=True, save_dir=save_dir)


