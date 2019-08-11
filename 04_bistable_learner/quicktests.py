from Conditioning import Experiment
import numpy as np



exp = Experiment(seed=0)
#exp.train_brain(400, save_dir='../../results/trash')
exp.train_brain(n_trials=2)
exp.train_brain(n_trials=2, baseline_only=True)
exp.train_brain(n_trials=3, baseline_only=False, rev_learn=True)


"""ratios = []
for seed in range(10):
    seed += 18
    print(seed)
    brain = Brain(master_seed=seed)
    #print('Building network')
    brain.build_local_network()
    #print('Simulating')
    ratios.append(brain.Simulate())
print(np.mean(ratios), np.std(ratios))"""