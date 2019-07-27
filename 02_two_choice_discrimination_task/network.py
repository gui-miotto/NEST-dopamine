import nest, os, itertools
import numpy as np
from scipy import sparse
from mpi4py import MPI
from time import time
from datetime import timedelta
from experiment_methods import ExperimentMethods
from experiment_results import ExperimentResults


class Network():
    def __init__(self, master_seed=400, A_plus_mult=2., A_minus_mult=1.5, Wmax_mult=4.):

        self.debug = False

        # Kernel parameters
        self.mseed = master_seed  # master seed
        self.dt = .1
        self.verbosity = 20
        self.n_threads = 10
        self.kernel_pars = {
            'print_time' : False,
            'resolution' : self.dt,
            'local_num_threads' : self.n_threads,
            }
        
        # Groups of subpopulations of neurons
        self.sp_group = dict()
        self.sp_group['sensory'] = ['S1', 'S2']
        self.sp_group['action'] = ['A1', 'A2']
        self.sp_group['decision'] = self.sp_group['sensory'] + self.sp_group['action']
        self.sp_group['excitatory'] = self.sp_group['decision'] + ['exc']
        self.sp_group['all_sps'] = self.sp_group['excitatory'] +  ['inh']

        # Number of neurons
        self.N = dict()  # number of neurons
        self.N['I'] = 100 if self.debug else 250 # number of inhibitory neurons 
        self.N['E'] = 4 * self.N['I']  # number of excitatory neurons
        self.N['ALL'] = self.N['I'] + self.N['E'] # total number of neurons
        for sp in self.sp_group['all_sps']: # subpopulations of neurons (involved in decision making or just for control)
            self.N[sp] = 50
        self.N['E_wout_S'] = self.N['E'] - self.N['S1'] - self.N['S2'] # All excitatory neurons, except the sensory ones
        self.N['cnn_mat'] = sum([self.N[sp] for sp in self.sp_group['excitatory']]) # neurons that will appear in the connectivity matrix
        self.N['DA'] = 1 # number of DA neurons (extra population)

        # Neuron (iaf_psc_delta) parameters
        self.neu_pars = {
            'tau_m' : 20.,
            'E_L': 0.,
            'V_reset' : 0.,
            'V_m' : 0.,
            'V_th' : 20.,
            }

        # Synapse parameters
        g = 6. #5. # ratio inhibitory weight/excitatory weight
        self.J = dict()  # amplitude of the postsynaptic potentials
        self.J['E'] = .1  # excitatory weight
        self.J['I'] = -g * self.J['E'] # inhibitory weight
        self.delay = 1.5  # synaptic delay in ms
        epsilon = .1  # connection probability
        self.C = {pop : int(epsilon * self.N[pop]) for pop in ['E', 'I', 'S1', 'S2', 'E_wout_S']} # num of syns per neuron
        self.initial_total_plastic_w = self.J['E'] * self.N['E'] * self.C['E']
     
        # Dopamine modulation parameters
        self.tau_n = 200.
        A_plus = A_plus_mult * self.J['E']
        A_minus = A_minus_mult * A_plus
        self.DA_pars = {
            'weight' : self.J['E'],  # Default 1.
            'delay': self.delay, # Default 1.; Synaptic delay
            'tau_n' : self.tau_n, # Default 200.; Time constant of dopaminergic trace in ms
            'b' : 1. / self.dt, # Default 0.; Dopaminergic baseline concentration
            'n' : 1. / self.dt + 2.5 / self.tau_n, # Default 0.; Initial dopamine concentration
            'A_plus' : A_plus,  # Default 1.; Amplitude of weight change for facilitation
            'A_minus' : A_minus,  # Default 1.5; Amplitude of weight change for depression
            'Wmax' : self.J['E'] * Wmax_mult, # Maximal synaptic weight  
            #'tau_c' : 1000., # Default 1000.,  # Time constant of eligibility trace in ms
            #'tau_plus' : 20.0, #  Default 20.; STDP time constant for facilitation in ms
            #'Wmin' : 0., # Default 0. # Minimal synaptic weight
            #'vt' : volt_DA[0], # Volume transmitter will be assigned later on
            }
        self.default_data_dir = f'Aplus={A_plus_mult}_Aminus={A_minus_mult}_Wmax={Wmax_mult}'
        
        # External noise parameters
        eta = .87 # 2. # external rate relative to threshold rate
        nu_th = self.neu_pars['V_th'] / (self.J['E'] * self.C['E'] * self.neu_pars['tau_m'])
        nu_ex = eta * nu_th
        self.p_rate = 1000. * nu_ex * self.C['E']

        # Experiment parameters
        self.warmup_duration = (1. if self.debug else 25.) * self.tau_n # warmup (no sensory input or decision making) before the trials begin
        self.trial_duration = 1100. if self.debug else 6000.  # Trial duration
        self.cue_duration = 3. # Cue stimulus duration
        self.cue_stim = 315. #300 # Stimulus delivered to cue neurons in pF
        self.eval_time_window = 20. # Time window to check response via spike count
        self.min_DA_wait_time = 100. # Minimum waiting time to reward
        self.max_DA_wait_time = 1000. # Maximum waiting time to reward
        

    def _configure_kernel(self):
        nest.ResetKernel()
        nest.set_verbosity(self.verbosity)
        nest.SetKernelStatus(self.kernel_pars)

        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_procs = self.mpi_comm.Get_size()
        v_procs = self.n_threads * self.mpi_procs

        # Internal random number generator (RNG) for NEST (i.e. used by the kernel)
        mid_seed = self.mseed + 1 + v_procs
        nest.SetKernelStatus({
                'grng_seed' : self.mseed,
                'rng_seeds' : range(self.mseed + 1, mid_seed),
                })
        # RNGs for the user (i.e used by this script)        
        self.py_rngs = [np.random.RandomState(seed) for seed in range(mid_seed, mid_seed + v_procs)]


    def build_network(self):
        self._configure_kernel()

        # Create all neurons
        nest.SetDefaults('iaf_psc_delta', self.neu_pars)
        self.nodes = {pop : nest.Create("iaf_psc_delta", self.N[pop]) for pop in ['E', 'I']}

        # Some usefull aliases:
        self.nodes['ALL'] = self.nodes['E'] + self.nodes['I']
        self.nodes['inh'] = self.nodes['I'][:self.N['inh']]
        cut = 0
        for pop in self.sp_group['excitatory']: 
            self.nodes[pop] = self.nodes['E'][cut : cut+self.N[pop]]
            cut += self.N[pop]
        self.nodes['E_wout_S'] = tuple(set(self.nodes['E']) - set(self.nodes['S1']) - set(self.nodes['S2']) )

        # Initiate membrane potentials randomly
        v_min, v_max = self.neu_pars['V_reset'], self.neu_pars['V_th']
        node_info = nest.GetStatus(self.nodes['ALL'])
        local_nodes = [(ni['global_id'], ni['vp']) for ni in node_info if ni['local']]
        for gid, proc in local_nodes:
            nest.SetStatus([gid], {'V_m': self.py_rngs[proc].uniform(v_min, v_max)})

        # Create and connect volume transmiter
        self.nodes['DA'] = nest.Create('spike_generator')
        workaround_parrot = nest.Create('parrot_neuron') # because conns between spike_gens and vol_trans are currently not allowed
        volt_DA = nest.Create("volume_transmitter")
        nest.Connect(self.nodes['DA'], workaround_parrot, syn_spec={'delay' : self.dt})
        nest.Connect(workaround_parrot, volt_DA, syn_spec={'delay' : self.dt})
        self.DA_pars['vt'] = volt_DA[0]

        # Main population connections:
        nest.CopyModel('static_synapse', 'static_E', {'weight': self.J['E'], 'delay': self.delay })
        nest.CopyModel('static_E', 'static_I', {'weight': self.J['I']})
        nest.CopyModel('stdp_dopamine_synapse', 'plastic_E', self.DA_pars)
        for pop in self.sp_group['sensory']:
            nest.Connect(self.nodes[pop], self.nodes['E'], \
                {'rule': 'fixed_indegree', 'indegree': self.C[pop]}, 'plastic_E')
        nest.Connect(self.nodes['E_wout_S'], self.nodes['E'], \
            {'rule': 'fixed_indegree', 'indegree': self.C['E_wout_S']}, 'plastic_E')
        nest.Connect(self.nodes['E'], self.nodes['I'], \
            {'rule': 'fixed_indegree', 'indegree': self.C['E']}, 'static_E')
        nest.Connect(self.nodes['I'], self.nodes['ALL'], \
            {'rule': 'fixed_indegree', 'indegree': self.C['I']}, 'static_I')
        self.plastic_syns = nest.GetConnections(self.nodes['E'], self.nodes['E'])
        self.S_to_A_syns = {'S1' : {}, 'S2': {}}
        for S, A in itertools.product(self.sp_group['sensory'], self.sp_group['action']):
            self.S_to_A_syns[S][A] = nest.GetConnections(self.nodes[S], self.nodes[A])

        # Create and background noise
        noise = nest.Create('poisson_generator', params={'rate': self.p_rate}) # Background noise
        nest.Connect(noise, self.nodes['ALL'], syn_spec='static_E')

        # Create and connect sensory stimulus
        self.sensory_drive = dict()
        for pop in self.sp_group['sensory']:
            self.sensory_drive[pop] = nest.Create('step_current_generator')
            nest.Connect(self.sensory_drive[pop], self.nodes[pop], syn_spec={'delay' : self.dt})

        # Create and connect spike detectors:
        self.spkdet = dict()
        for pop in self.sp_group['all_sps'] + ['E']:
            self.spkdet[pop] = nest.Create('spike_detector')
            nest.Connect(self.nodes[pop], self.spkdet[pop])

    def simulate(self, data_dir=None, n_trials=400, syn_scaling=True, aversion=True):
        # (warning: nest.ResetNetwork wont reboot synaptic weights, 
        # we would have to do it manually or just rebuild the network)
        self.build_network()

        if data_dir is None:
            data_dir = f'Aversion={aversion}_' + self.default_data_dir
            data_dir = os.path.join('results', data_dir)
        
        self.n_trials = n_trials
        if self.mpi_rank == 0:
            ExperimentMethods(self)._pickle(data_dir)
            correct, wrong, draw = 0, 0, 0
            print(f'Initial total plastic weight: {self.initial_total_plastic_w}\n')
            print(f'Simulating warmup for {self.warmup_duration} ms')
            warmup_start = time()
        self.simulate_rest_state(duration=self.warmup_duration, reset_weights=True)
        if self.mpi_rank == 0:
            print(f'Warmup simulated in {time() - warmup_start : .3f} seconds\n')
            trials_wall_clock_time = list()

        self.rescal_factor = 1.
        for self.trial in range(1, n_trials+1):
            if self.mpi_rank == 0:
                print(f'Simulating trial {self.trial} of {n_trials}:')
                trial_start = time()
            self.trial_cue, A1_minus_A2, was_correct = self._simulate_one_trial(aversion=aversion)
            if self.mpi_rank == 0:
                wall_clock_time = time() - trial_start
                trials_wall_clock_time.append(wall_clock_time)
                
            # TODO: delelte this, once debuged
            #A_minus_B, n_spikes_DA = self._simulate_one_trial(aversion=aversion)
            #print('DA spikes', n_spikes_DA, self.rank)

            if self.mpi_rank == 0:
                print('Stimulus', self.trial_cue)
                for pop, dspk in self.decision_spikes.items():
                    print(f'Spikes {pop}: {dspk}')
                if A1_minus_A2 == 0:
                    print('Draw. :-|')
                    draw += 1
                elif was_correct:
                    print('right response! :-)')
                    correct += 1    
                else:
                    print('wrong response. :-(')
                    wrong += 1

            # Weight rescaling:
            if(syn_scaling):
                old_total_weight, new_weights, self.rescal_factor = self._synaptic_rescaling()
                print_r0('End-of-trial total weight:', old_total_weight)
                print_r0('scaling it by a factor of', self.rescal_factor)
            else:
                new_weights = nest.GetStatus(self.plastic_syns, 'weight')
            self.EE_w_hist = np.histogram(new_weights, bins=20, range=(0., self.DA_pars['Wmax']))
            
            # Weights between sensory and motor populations
            self.S_to_A_weight = {'S1' : {}, 'S2': {}}
            for S, A in itertools.product(self.sp_group['sensory'], self.sp_group['action']):
                self.S_to_A_weight[S][A] = nest.GetStatus(self.S_to_A_syns[S][A], 'weight')

            # Connectivity matrix
            self.cnn_matrix = self._E_pop_connectivity_matrix()

            if self.mpi_rank == 0:
                print(f'Parcial results (out of {self.trial} trials):')
                print(f'{correct} correct selections ({(correct*100./self.trial):.2f}%)')
                print(f'{wrong} wrong selections ({(wrong*100./self.trial):.2f}%)')
                print(f'{draw} draws ({(draw*100./self.trial):.2f}%)')
                print(f'Elapsed real time: {wall_clock_time:.1f} seconds')
                mean_wct = np.mean(trials_wall_clock_time)
                print(f'Average elapsed time per trial: {mean_wct:.1f} seconds')
                remaining_wct = round(mean_wct * (n_trials - self.trial))
                print(f'Expected remaining time: {timedelta(seconds=remaining_wct)}\n')
            
            ExperimentResults(self)._pickle(data_dir)

        return correct / n_trials


    def _simulate_one_trial(self, aversion=True):
        # Decide randomly what will be the next cue
        next_cue = None
        if self.mpi_rank == 0:
            cue_index = self.py_rngs[0].randint(len(self.sp_group['sensory']))
            next_cue = self.sp_group['sensory'][cue_index]
        next_cue = self.mpi_comm.bcast(next_cue, root=0)

        # Program stimulus
        self.trial_begin = nest.GetKernelStatus()['time']
        cue_onset = self.trial_begin + self.dt
        nest.SetStatus(self.sensory_drive[next_cue], params={
            'amplitude_times' : [cue_onset, cue_onset + self.cue_duration],
            'amplitude_values' : [self.cue_stim, 0.],
        })

        # Program dopamine baseline
        DA_spike_times = np.arange(
            self.trial_begin + self.dt, 
            self.trial_begin + self.dt + self.eval_time_window,
            self.dt)
        DA_spike_times = np.round(DA_spike_times, decimals=1)
        nest.SetStatus(self.nodes['DA'], params={'spike_times' : DA_spike_times})

        # Simulate the evaluation windown
        nest.Simulate(self.eval_time_window)
        curr_time = nest.GetKernelStatus()['time']

        # Read response spike detectors
        dec_spk = [nest.GetStatus(self.spkdet[pop], 'n_events')[0] for pop in self.sp_group['action']]
        dec_spk = np.array(dec_spk, dtype='i')
        recvbuf = np.empty([self.mpi_procs, 2], dtype='i') if self.mpi_rank == 0 else None
        self.mpi_comm.Gather(dec_spk, recvbuf, root=0)
        if self.mpi_rank == 0:
            recvbuf = np.sum(recvbuf, axis=0)
            self.decision_spikes = {pop : recvbuf[it] for it, pop in enumerate(['A1', 'A2'])}
        else:
            self.decision_spikes = dict()
        self.decision_spikes = self.mpi_comm.bcast(self.decision_spikes, root=0)

        # According to the selected action, deliver the appropriate DA response
        A1_minus_A2 = self.decision_spikes['A1'] - self.decision_spikes['A2']
        correct_action = (next_cue == 'S1' and A1_minus_A2 > 0) or (next_cue == 'S2' and A1_minus_A2 < 0)
        end_of_trial = curr_time + self.trial_duration - self.eval_time_window + self.dt
        if A1_minus_A2 == 0 or (not correct_action and not aversion):  # just keep the baseline
            DA_spike_times = np.arange(curr_time, end_of_trial, self.dt)
        else:
            wait_time = self.max_DA_wait_time - (abs(A1_minus_A2) - 1) * 100.
            wait_time = np.clip(wait_time, self.min_DA_wait_time, self.max_DA_wait_time)
            delivery_time = round(curr_time + wait_time)
            if correct_action:  # Deliver reward (an extra DA spike)
                DA_spike_times = np.concatenate((
                    np.arange(curr_time, delivery_time, self.dt),
                    np.arange(delivery_time - self.dt, end_of_trial, self.dt)
                ))
            else:  # Deliver punishment (a missing DA spike)
                DA_spike_times = np.concatenate((
                    np.arange(curr_time, delivery_time - 1.5 * self.dt, self.dt),  # 1.5 multiplication for numerical stability
                    np.arange(delivery_time, end_of_trial, self.dt)
                ))
        DA_spike_times = np.round(DA_spike_times, decimals=1)
        nest.SetStatus(self.nodes['DA'], params={'spike_times' : DA_spike_times})

        # Simulate the rest of the trial
        nest.Simulate(self.trial_duration - self.eval_time_window)

        # Read events
        self.events = dict()
        for pop in self.sp_group['all_sps']:
            self.events[pop] = nest.GetStatus(self.spkdet[pop], 'events')[0]
            nest.SetStatus(self.spkdet[pop], {'n_events' : 0 })
        self.EE_fr_hist = self._EE_fr_histogram(nest.GetStatus(self.spkdet['E'], 'events')[0])
        nest.SetStatus(self.spkdet['E'], {'n_events' : 0 })

        # TODO: delete DA spike detector once we know that this is bug free        
        #n_events_DA = nest.GetStatus(self.spkdet['DA'], 'n_events')[0]
        #nest.SetStatus(self.spkdet['DA'], {'n_events' : 0 })

        return next_cue, A1_minus_A2, correct_action #, n_events_DA


    def simulate_rest_state(self, duration=100., reset_weights=True):
        # DA baseline and no stimulus
        curr_time = nest.GetKernelStatus()['time']
        DA_spike_times = np.arange(
            curr_time + self.dt, 
            curr_time + self.dt + duration, 
            self.dt)
        DA_spike_times = np.round(DA_spike_times, decimals=1)
        nest.SetStatus(self.nodes['DA'], params={'spike_times' : DA_spike_times})
        nest.Simulate(duration)
        syn_rescal_factor, _, _, _ = self._get_weight_scaling_factor()
        #print(nest.GetStatus(self.spkdet['DA'], 'n_events')) # for debug
        #for pop in ['S', 'A', 'B', 'exc', 'inh', 'E', 'DA']:
        for pop in self.sp_group['all_sps']:
            nest.SetStatus(self.spkdet[pop], {'n_events' : 0 })
        if reset_weights:
            nest.SetStatus(self.plastic_syns, params='weight', val=self.J['E'])
        return syn_rescal_factor

    def _synaptic_rescaling(self):
        syn_rescal_factor, _, old_total_weight, new_weights = self._get_weight_scaling_factor()
        nest.SetStatus(self.plastic_syns, params='weight', val=new_weights)
        return old_total_weight, new_weights, syn_rescal_factor
    
    def _get_weight_scaling_factor(self):
        weights = nest.GetStatus(self.plastic_syns, 'weight')
        total_weight = np.sum(weights, dtype='f')
        recvbuf = np.empty(self.mpi_procs, dtype='f') if self.mpi_rank == 0 else None
        self.mpi_comm.Gather(total_weight, recvbuf, root=0)
        if self.mpi_rank == 0:
            total_weight = np.sum(recvbuf)
            syn_rescal_factor = self.initial_total_plastic_w / total_weight
        else:
            syn_rescal_factor = None
        syn_rescal_factor = self.mpi_comm.bcast(syn_rescal_factor, root=0)
        scaled_weights = np.array(weights) * syn_rescal_factor
        return syn_rescal_factor, weights, total_weight, scaled_weights

    def _EE_fr_histogram(self, events):
        snds = events['senders']
        times = np.array(events['times'])
        frs = []
        for snd in np.unique(snds):
            snd_times = times[snds==snd]
            fr = len(snd_times) * 1000. / self.trial_duration
            frs.append(fr)
        return np.histogram(frs, bins=20, range=(0, 10.))

    def _E_pop_connectivity_matrix(self):
        # First create a map from global ids to matrices index
        prev_max_gid = 0
        gids, inds = list(), dict()
        for pop in self.sp_group['excitatory']:
            gids += self.nodes[pop]
            gid = np.array(self.nodes[pop])
            ind = gid - np.min(gid) + prev_max_gid + 1
            for g, i in zip(gid, ind):
                inds[g] = i
            prev_max_gid = np.max(gid)
        # Then calculate the matrix itself
        cnn_matrix = np.zeros((len(gids), len(gids)))
        for cnn in nest.GetConnections(gids, gids):
            pre = nest.GetStatus([cnn], 'source')[0]
            post = nest.GetStatus([cnn], 'target')[0]
            weights = nest.GetStatus([cnn], 'weight')
            cnn_matrix[inds[pre]-1, inds[post]-1] += np.sum(weights)
        return sparse.coo_matrix(cnn_matrix)

def print_r0(*args):
    if nest.Rank() == 0:
        print(*args)