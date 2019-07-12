import pickle
import os.path

class ExperimentMethods():

    def __init__(self, net):
        self.N = net.N
        self.C = net.C
        self.nodes = net.nodes
        self.DA_pars = net.DA_pars
        self.n_trials = net.n_trials
        self.trial_duration = net.trial_duration
        self.eval_time_window = net.eval_time_window
        self.n_procs = net.n_procs

    def _pickle(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        file_path = os.path.join(save_dir, 'methods.data')
        pickle.dump(self, open(file_path, 'wb'))
    
    


