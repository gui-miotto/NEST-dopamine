import pickle
import os.path

class ExperimentMethods():

    def __init__(self, exp):
        self.rank = exp.brain.mpi_rank
        self.weights_count = exp.brain.weights_count
        self.eval_time_window = exp.eval_time_window
        self.neurons = {
            'E_rec' : exp.brain.cortex.neurons['E_rec'],
            'low' : exp.brain.cortex.neurons['low'],
            'high' : exp.brain.cortex.neurons['high'],
            'left' : exp.brain.striatum.neurons['left'],
            'right' : exp.brain.striatum.neurons['right'],
        }

    def write(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        file_path = os.path.join(save_dir, 'methods-rank-'+str(self.rank).rjust(3, '0')+'.data')
        pickle.dump(self, open(file_path, 'wb'))
    
    


