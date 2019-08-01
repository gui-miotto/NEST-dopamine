import pickle, os
import numpy as np
import pandas as pd
from glob import glob
from functools import reduce

class Reader():
    @property
    def num_of_trials(self):
        return len(self.cue)

    def __init__(self):
        self.cue = []
        self.lminusr = []
        self.success = []
        self.events = []
        self.weights_mean = []

    def read(self, exp_dir):

        weights_count, weight_count_total = self._read_methods(exp_dir)


        #loop through trials
        trials_dir = os.path.join(exp_dir, 'trial-*/')
        for trial_dir in sorted(glob(trials_dir)):
            print('Reading', trial_dir)
            
            # Some information are the same for every MPI process. So we read just from rank 0.
            r0_file_path = os.path.join(trial_dir, 'rank-000.data')
            ER0 = pickle.load(open(r0_file_path, 'rb'))
            self.cue.append(ER0.cue)
            self.lminusr.append(ER0.lminusr)
            self.success.append(ER0.success)
            events = {key : {'senders':np.array([]), 'times':np.array([])} for key in ER0.events}
            weights_mean = weights_count[0] * 0

            # And then we loop through the different MPI processes outputs
            for proc_file in os.listdir(trial_dir):
                ER_file_path = os.path.join(trial_dir, proc_file)
                ER = pickle.load(open(ER_file_path, 'rb'))

                # Merge events (for raster plot)
                for key in events:
                    events[key]['senders'] = \
                        np.concatenate((events[key]['senders'], ER.events[key]['senders']))
                    events[key]['times'] = \
                        np.concatenate((events[key]['times'], ER.events[key]['times']))
                
                # Merge weights data
                weights_mean += weights_count[ER.rank] * ER.weights_mean



            
            
            self.events.append(events)
            self.weights_mean.append(weights_mean / weight_count_total)



        return self

    def _read_methods(self, exp_dir):
        # Read the methods
        # Some information are the same for every MPI process. So we read just from rank 0.
        m0_file_path =  os.path.join(exp_dir, 'methods-rank-000.data')
        EM0 = pickle.load(open(m0_file_path, 'rb'))
        self.neurons = EM0.neurons
        self.eval_time_window = EM0.eval_time_window

        #And then we loop through the different MPI processes outputs
        methods_paths = os.path.join(exp_dir, 'methods-rank-*.data')
        weights_count = list()
        for mp in sorted(glob(methods_paths)):
            EM = pickle.load(open(mp, 'rb'))
            weights_count.append(EM.weights_count)
        weight_count_total = reduce(lambda x, y: x.add(y, fill_value=0), weights_count)

        return weights_count, weight_count_total
    


    


