from SpikingBGRL import Experiment
import numpy as np



exp = Experiment(seed=0, debug_mode=False)
#exp.train_brain(400, save_dir='../../results/trash')
#exp.train_brain(n_trials=1, save_dir='../../results/blendernet')
exp.train_brain(n_trials=1)




#exp.train_brain(n_trials=2, baseline_only=True)
#exp.train_brain(n_trials=3, baseline_only=False, rev_learn=True)

