import pickle, time, os.path


class ExperimentResults(object):

    def __init__(self, net):
        self.rank = net.rank
        self.trial = net.trial
        self.trial_begin = net.trial_begin
        self.decision_spikes = net.decision_spikes
        self.events = net.events
        self.rescal_factor = net.rescal_factor
        self.EE_w_hist = net.EE_w_hist
        self.EE_fr_hist = net.EE_fr_hist
        self.S_to_pop_weight = net.S_to_pop_weight
        self.cnn_matrix = net.cnn_matrix

    def _pickle(self, save_dir):
        save_dir = os.path.join(save_dir, 'trial-'+str(self.trial).rjust(3, '0'))
        while not os.path.exists(save_dir):
            if self.rank == 0:
                os.mkdir(save_dir)
            else:
                time.sleep(.1)
        file_path = os.path.join(save_dir, 'rank-'+str(self.rank).rjust(3, '0')+'.data')
        pickle.dump(self, open(file_path, 'wb'))