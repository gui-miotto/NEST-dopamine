from Experiment import Experiment
import numpy as np

exp = Experiment()
# I just the wait_time multiplicator from 50. to 100.

exp.train_brain(400, aversion=False, save_dir='../../results/local_02')


