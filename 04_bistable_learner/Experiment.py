import nest
import numpy as np
import BrainStructures as BS
import DataIO as DIO

class Experiment():
    """Class representing the instrumental conditioning of a brain. A experiment is sequence of 
    trials. At each trial, a cue is presented to the brain and an action is taken by the brain. 
    Class members whose names are followed by a trailing _ (e.g. self.success_) are updated at every
    trial, the others are constant throughout the whole experiment.
    """
    def __init__(self, seed=42):
        """Constructor
        
        Parameters
        ----------
        seed : int, optional
            Master seed for EVERYTHING. Runs with the same seed and number of virtual processes
            should yeld the same results. By default 42
        """
        self.debug = False
        
        # Experiment parameters
        #TODO: warmup
        #self.warmup_duration = (1. if self.debug else 25.) * self.tau_n # warmup (no sensory input or decision making) before the trials begin
        trial_duration = 1100. if self.debug else 6000.  # Trial duration
        self.eval_time_window = 50. # Time window to check response via spike count
        self.tail_of_trial = trial_duration - self.eval_time_window
        self.min_DA_wait_time = 100. # Minimum waiting time to reward
        self.max_DA_wait_time = 1000. # Maximum waiting time to reward

        # A random number generator (used to determine the sequence of cues)
        self.rng = np.random.RandomState(seed)

        # The brain to be trained
        scale = .2 if self.debug else 1.
        self.brain = BS.Brain(master_seed=seed, scale=scale)

    
    def train_brain(self, n_trials=400, save_dir='temp'):
        """ Creates a brain and trains it for a specific number of trials.
        
        Parameters
        ----------
        n_trials : int, optional
            Number of trials to perform, by default 400
        save_dir : str, optional
            Directory where the outputs will be saved. Existing files will be overwritten. By 
            default 'temp'
        """
        # Create the whole neural network
        self.brain.build_local_network()
        
        # Write to file the experiment properties which are trial-independent
        DIO.ExperimentMethods(self).write(save_dir)

        for self.trial_ in range(1, n_trials+1):
            print('Trial', self.trial_, 'of', n_trials)
            self._simulate_one_trial()
            print('Correct action?', self.success_)

            # Store experiment results on file(s):
            self.brain.read_reset_spike_detectors()
            self.brain.read_synaptic_weights()
            DIO.ExperimentResults(self).write(save_dir)



    def _simulate_one_trial(self):
        # Decide randomly what will be the next cue and do the corresponding stimulation
        self.cue_ = ['low', 'high'][self.rng.randint(2)]
        self.brain.cortex.stimulate_subpopulation(spop=self.cue_, delay=self.brain.dt)
        
        # Simulate evaluation window and count the resulting decision spikes
        self.brain.vta.set_drive(length=self.eval_time_window, drive_type='baseline')
        nest.Simulate(self.eval_time_window)
        decision_spikes = self.brain.striatum.count_decision_spikes()

        # According to the selected action, deliver the appropriate DA response
        self.lminusr_ = decision_spikes['left'] - decision_spikes['right']
        self.success_ = (self.cue_ == 'low' and self.lminusr_ > 0) or \
                        (self.cue_ == 'high' and self.lminusr_ < 0)
        
        if self.lminusr_ == 0:  # just keep the baseline  #TODO I think this wont almost never happen anymore. Change the criterion?
            self.brain.vta.set_drive(length=self.tail_of_trial, drive_type='baseline')
        else:
            wait_time = self.max_DA_wait_time - (abs(self.lminusr_) - 1) * 10.  #TODO: calibrate this
            wait_time = round(np.clip(wait_time, self.min_DA_wait_time, self.max_DA_wait_time))
            drive_type = 'rewarding' if self.success_ else 'aversive'
            self.brain.vta.set_drive(length=self.tail_of_trial, drive_type=drive_type, delay=wait_time)

        # Simulate rest of the trial
        nest.Simulate(self.tail_of_trial)





        








    





