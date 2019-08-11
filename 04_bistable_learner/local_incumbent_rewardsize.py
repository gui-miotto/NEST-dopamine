from Experiment import Experiment
import numpy as np


exp = Experiment()
exp.brain.vta.DA_pars['A_plus'] = .1 * exp.brain.vta.DA_pars['weight']
exp.brain.vta.DA_pars['A_minus'] = .15 * exp.brain.vta.DA_pars['weight']
save_dir = '../../results/local_incumbent_rewardsize'
exp.train_brain(n_trials=400, aversion=False, save_dir=save_dir)


