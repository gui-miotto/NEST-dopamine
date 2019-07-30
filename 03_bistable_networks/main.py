from Brain import Brain
import numpy as np

ratios = []

for seed in range(10):
    seed += 18
    print(seed)
    brain = Brain(master_seed=seed)
    #print('Building network')
    brain.build_local_network()
    #print('Simulating')
    ratios.append(brain.Simulate())
    


print(np.mean(ratios), np.std(ratios))