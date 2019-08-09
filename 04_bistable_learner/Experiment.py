import nest
import numpy as np
import BrainStructures as BS
import DataIO as DIO
from mpi4py import MPI
from time import time
from datetime import timedelta

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
        self.trial_duration = 1100. if self.debug else 6000.  # Trial duration
        self.eval_time_window = 20. # Time window to check response via spike count
        self.tail_of_trial = self.trial_duration - self.eval_time_window
        self.min_DA_wait_time = 100. # Minimum waiting time to reward
        self.max_DA_wait_time = 1000. # Maximum waiting time to reward
        self.warmup_magnitude = 1. if self.debug else 25. # The duration of the warmup period is        
                                                          # given by warmup_magnitude * vta.tau_n

        # A random number generator (used to determine the sequence of cues)
        self.rng = np.random.RandomState(seed)

        # The brain to be trained
        scale = .2 if self.debug else 1.
        self.brain = BS.Brain(master_seed=seed, scale=scale)

        #MPI rank (here used basically just to avoid multiple printing)
        self.mpi_rank = MPI.COMM_WORLD.Get_rank()

    
    def train_brain(self, n_trials=400, syn_scaling=True, aversion=True, save_dir='/tmp/learner'):
        """ Creates a brain and trains it for a specific number of trials.
        
        Parameters
        ----------
        n_trials : int, optional
            Number of trials to perform, by default 400
        save_dir : str, optional
            Directory where the outputs will be saved. Existing files will be overwritten. By 
            default 'temp'
        """
        # Some handy variables
        rank0 = self.mpi_rank == 0
        color = {'red' : '\033[91m', 'green' : '\033[92m', 'none' : '\033[0m'}

        # Create the whole neural network
        if rank0:
            print('\nBuilding network')
        build_start = time()
        n_nodes = self.brain.build_local_network()
        build_elapsed_time = time() - build_start

        # Write to file the experiment properties which are trial-independent
        DIO.ExperimentMethods(self).write(save_dir)

        # Simulate warmup
        warmup_duration = self.warmup_magnitude * self.brain.vta.tau_n
        
        if rank0:
            print(f'Building completed in {build_elapsed_time:.1f} seconds')
            print('Number of nodes:', n_nodes)
            print(f'Initial total plastic weight: {self.brain.initial_total_weight:,}')
            print(f'Simulating warmup for {warmup_duration} ms')
        warmup_start = time()
        syn_change = self.simulate_rest_state(duration=warmup_duration, reset_weights=True)
        warmup_elapsed_time = time() - warmup_start
        if rank0:
            print(f'Warmup simulated in {warmup_elapsed_time:.1f} seconds')
            print(f'Synaptic change during warmup: {syn_change:.5f}\n')

        # Simulate trials
        trials_wall_clock_time, successes = list(), list()
        for self.trial_ in range(1, n_trials+1):
            if rank0:
                print(f'Simulating trial {self.trial_} of {n_trials}:')
            
            # Simulate one trial and measure time taken to do it
            trial_start = time()
            self._simulate_one_trial(aversion)
            wall_clock_time = time() - trial_start
            trials_wall_clock_time.append(wall_clock_time)
            successes.append(self.success_)

            # Synaptic scaling
            if syn_scaling:
                self.brain.homeostatic_scaling()
            
            # Store experiment results on file(s):
            self.brain.read_reset_spike_detectors()
            self.brain.read_synaptic_weights()
            DIO.ExperimentResults(self).write(save_dir)

            # Print some useful monitoring information
            n_succ = np.sum(successes)
            if rank0:
                print(f'Trial simulation concluded in {wall_clock_time:.1f} seconds')
                print(f'End-of-trial weight change: {self.brain.syn_change_factor_:.5f}')
                if self.success_:
                    print(f'{color["green"]}Correct action{color["none"]}')
                else:
                    print(f'{color["red"]}Wrong action{color["none"]}')
                print(f'{n_succ} correct actions so far ({n_succ * 100. / self.trial_:.2f}%)')
                mean_wct = np.mean(trials_wall_clock_time)
                print(f'Average elapsed time per trial: {mean_wct:.1f} seconds')
                remaining_wct = round(mean_wct * (n_trials - self.trial_))
                print(f'Expected remaining time: {timedelta(seconds=remaining_wct)}\n')
        
        return successes


    def _simulate_one_trial(self, aversion):
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
            wait_time = self.max_DA_wait_time - (abs(self.lminusr_) - 1) * 100.  #TODO: calibrate this
            wait_time = round(np.clip(wait_time, self.min_DA_wait_time, self.max_DA_wait_time))
            drive_type = 'rewarding' if self.success_ else 'aversive' if aversion else 'baseline'
            self.brain.vta.set_drive(length=self.tail_of_trial, drive_type=drive_type, delay=wait_time)

        # Simulate rest of the trial
        nest.Simulate(self.tail_of_trial)

    
    def simulate_rest_state(self, duration=100., reset_weights=True):
        """Simulates the network in its resting state, i.e.: no stimulus and under dopamine baseline
        levels. This function is used to simulate the warmup period and is a great debuging tool.
        
        Parameters
        ----------
        duration : float, optional
            Simulation duration, by default 100.
        reset_weights : bool, optional
            If true corticostriatal synapses will be set to it initial value after the simulation, 
            by default True
        """        
        self.brain.vta.set_drive(length=duration, drive_type='baseline')
        nest.Simulate(duration)
        syn_change_factor = self.brain.get_total_weight_change()
        self.brain.read_reset_spike_detectors()
        if reset_weights:
            self.brain.reset_corticostriatal_synapses()

        return syn_change_factor



        








    





