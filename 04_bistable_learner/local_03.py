from Experiment import Experiment
import numpy as np

exp = Experiment()
exp.brain.vta.DA_pars['A_minus'] = 0.
exp.train_brain(400, aversion=True, save_dir='../../results/local_03')


