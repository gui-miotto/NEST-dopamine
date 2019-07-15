import nest, os
import numpy as np
from scipy import sparse
from mpi4py import MPI
from time import time
from datetime import timedelta
from experiment_methods import ExperimentMethods
from experiment_results import ExperimentResults


class Network():
    def __init__(self, master_seed=400, A_plus_mult=2., A_minus_mult=1.5, Wmax_mult=4.):

        self.debug = True

        # Kernel parameters
        self.mseed = master_seed  # master seed
        self.dt = .1
        self.verbosity = 20
        self.kernel_pars = {
            'print_time' : False,
            'resolution' : self.dt,
            }

        # Number of neurons
        self.N = dict()  # number of neurons
        self.N['I'] = 50 if self.debug else 250 # number of inhibitory neurons 
        self.N['E'] = 4 * self.N['I']  # number of excitatory neurons
        self.N['ALL'] = self.N['I'] + self.N['E'] # total number of neurons
        self.N['S'] = self.N['A'] = self.N['B'] = 50 # Subpopulations involved in the decision making
        self.N['exc'] = self.N['inh'] = 50 # Subpopulations samples to be recorded
        self.N['cnn_mat'] = self.N['S'] + self.N['A'] + self.N['B'] + self.N['exc']
        self.N['E_wout_S'] = self.N['E'] - self.N['S']
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
        self.C = {pop : int(epsilon * self.N[pop]) for pop in ['E', 'I', 'S', 'E_wout_S']} # num of syns per neuron
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
        self.trial_duration = 1100. if self.debug else 6000.  # Trial duration
        self.cue_duration = 3. # Cue stimulus duration
        self.cue_stim = 315. #300 # Stimulus delivered to cue neurons in pF
        self.eval_time_window = 20. # Time window to check response via spike count
        self.max_DA_wait_time = 1000. # Maximum waiting time to reward
        

    def _configure_kernel(self):
        nest.ResetKernel()
        nest.set_verbosity(self.verbosity)
        nest.SetKernelStatus(self.kernel_pars)

        self.mpi_comm = MPI.COMM_WORLD
        self.rank = self.mpi_comm.Get_rank()
        self.n_procs = self.mpi_comm.Get_size()
        if self.n_procs != nest.GetKernelStatus('total_num_virtual_procs'):
            raise Exception('Dont use multithreading. As of NEST 2.18.0 it is buggy with stdp_dopamine_synapse')

        # Create random number generators
        # (2 * num_of_virtual_processes + 1)
        # (i.e. : One global NEST RNG + num_of_VPs RNGs for python + num_of_VPs RNGs for NEST)
        msd_range1 = range(self.mseed, self.mseed + self.n_procs)
        msd_range2 = range(self.mseed + 1 + self.n_procs, self.mseed + 1 + 2 * self.n_procs)
        self.py_rngs = [np.random.RandomState(seed) for seed in msd_range1]
        nest.SetKernelStatus({
                'grng_seed' : self.mseed + self.n_procs,
                'rng_seeds' : msd_range2,
                })
    
    def build_network(self):
        self._configure_kernel()

        # Create all neurons
        nest.SetDefaults('iaf_psc_delta', self.neu_pars)
        self.nodes = {pop : nest.Create("iaf_psc_delta", self.N[pop]) for pop in ['E', 'I']}

        # Some usefull aliases:
        self.nodes['ALL'] = self.nodes['E'] + self.nodes['I']
        self.nodes['inh'] = self.nodes['I'][:self.N['inh']]
        cut = 0
        for pop in ['S', 'A', 'B', 'exc']: 
            self.nodes[pop] = self.nodes['E'][cut : cut+self.N[pop]]
            cut += self.N[pop]
        self.nodes['E_wout_S'] = tuple(set(self.nodes['E']) - set(self.nodes['S']))

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
        nest.Connect(self.nodes['S'], self.nodes['E'], \
            {'rule': 'fixed_indegree', 'indegree': self.C['S']}, 'plastic_E')
        nest.Connect(self.nodes['E_wout_S'], self.nodes['E'], \
            {'rule': 'fixed_indegree', 'indegree': self.C['E_wout_S']}, 'plastic_E')
        nest.Connect(self.nodes['E'], self.nodes['I'], \
            {'rule': 'fixed_indegree', 'indegree': self.C['E']}, 'static_E')
        nest.Connect(self.nodes['I'], self.nodes['ALL'], \
            {'rule': 'fixed_indegree', 'indegree': self.C['I']}, 'static_I')
        self.plastic_syns = nest.GetConnections(self.nodes['E'], self.nodes['E'])
        self.S_to_pop_syns = {pop : nest.GetConnections(self.nodes['S'], self.nodes[pop]) for pop in ['A', 'B']}

        # Create and background noise
        noise = nest.Create('poisson_generator', params={'rate': self.p_rate}) # Background noise
        nest.Connect(noise, self.nodes['ALL'], syn_spec='static_E')

        # Create and connect sensory stimulus
        self.sensory_drive = nest.Create('step_current_generator')
        nest.Connect(self.sensory_drive, self.nodes['S'], syn_spec={'delay' : self.dt})

        # Create and connect spike detectors:
        self.spkdet = {}
        for pop in ['S', 'A', 'B', 'exc', 'inh', 'E']:
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
        warmup_duration = (1. if self.debug else 25.) * self.tau_n
        if self.rank == 0:
            ExperimentMethods(self)._pickle(data_dir)
            A_sel, B_sel, draw = 0, 0, 0
            print(f'Initial total plastic weight: {self.initial_total_plastic_w}\n')
            print(f'Simulating warmup for {warmup_duration} ms')
            warmup_start = time()
        self.simulate_rest_state(duration=warmup_duration, reset_weights=True)
        if self.rank == 0:
            print(f'Warmup simulated in {time() - warmup_start : .3f} seconds\n')
            trials_wall_clock_time = list()

        self.rescal_factor = 1.
        for self.trial in range(1, n_trials+1):

            if self.rank == 0:
                print(f'Simulating trial {self.trial} of {n_trials}:')
                trial_start = time()
            A_minus_B = self._simulate_one_trial(aversion=aversion)
            if self.rank == 0:
                wall_clock_time = time() - trial_start
                trials_wall_clock_time.append(wall_clock_time)
                

            # TODO: delelte this, once debuged
            #A_minus_B, n_spikes_DA = self._simulate_one_trial(aversion=aversion)
            #print('DA spikes', n_spikes_DA, self.rank)

            if self.rank == 0:
                for pop, dspk in self.decision_spikes.items():
                    print(f'Spikes {pop}: {dspk}')
                if A_minus_B > 0:
                    print('right response! :-)')
                    A_sel += 1
                elif A_minus_B < 0:
                    print('wrong response. :-(')
                    B_sel += 1
                else:
                    print('Draw. :-|')
                    draw += 1

            # Weight rescaling:
            if(syn_scaling):
                old_total_weight, new_weights, self.rescal_factor = self._synaptic_rescaling()
                print_r0('End-of-trial total weight:', old_total_weight)
                print_r0('scaling it by a factor of', self.rescal_factor)
            else:
                new_weights = nest.GetStatus(self.plastic_syns, 'weight')
            self.EE_w_hist = np.histogram(new_weights, bins=20, range=(0., self.DA_pars['Wmax']))
            
            # Mean weight between population S and population A and B
            self.S_to_pop_weight = {pop : nest.GetStatus(self.S_to_pop_syns[pop], 'weight') for pop in ['A', 'B']}

            # Connectivity matrix
            self.cnn_matrix = self._E_pop_connectivity_matrix()

            if self.rank == 0:
                print(f'Parcial results (out of {self.trial} trials):')
                print(f'{A_sel} correct selections ({(A_sel*100./self.trial):.2f}%)')
                print(f'{B_sel} wrong selections ({(B_sel*100./self.trial):.2f}%)')
                print(f'{draw} draws ({(draw*100./self.trial):.2f}%)')
                print(f'Elapsed real time: {wall_clock_time:.1f} seconds')
                mean_wct = np.mean(trials_wall_clock_time)
                print(f'Average elapsed time per trial: {mean_wct:.1f} seconds')
                remaining_wct = round(mean_wct * (n_trials - self.trial))
                print(f'Expected remaining time: {timedelta(seconds=remaining_wct)}\n')
                                
            
            ExperimentResults(self)._pickle(data_dir)

        return A_sel / n_trials


    def _simulate_one_trial(self, aversion=True):
        # Program stimulus
        self.trial_begin = nest.GetKernelStatus()['time']
        cue_onset = self.trial_begin + self.dt
        nest.SetStatus(self.sensory_drive, params={
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
        dec_spk = [nest.GetStatus(self.spkdet[pop], 'n_events')[0] for pop in ['A', 'B']]
        dec_spk = np.array(dec_spk, dtype='i')
        recvbuf = np.empty([self.n_procs, 2], dtype='i') if self.rank == 0 else None
        self.mpi_comm.Gather(dec_spk, recvbuf, root=0)
        if self.rank == 0:
            recvbuf = np.sum(recvbuf, axis=0)
            self.decision_spikes = {pop : recvbuf[it] for it, pop in enumerate(['A', 'B'])}
        else:
            self.decision_spikes = dict()
        self.decision_spikes = self.mpi_comm.bcast(self.decision_spikes, root=0)

        # According to the selected action, deliver the appropriate DA response
        A_minus_B = self.decision_spikes['A'] - self.decision_spikes['B']
        end_of_trial = curr_time + self.trial_duration - self.eval_time_window + self.dt
        if A_minus_B == 0 or (A_minus_B < 0 and aversion == False):  # just keep the baseline
            DA_spike_times = np.arange(curr_time, end_of_trial, self.dt)
        else:
            wait_time = self.max_DA_wait_time - (abs(A_minus_B) - 1) * 100.
            wait_time = np.clip(wait_time, 100., self.max_DA_wait_time)
            delivery_time = round(curr_time + wait_time)
            if A_minus_B > 0:  # Deliver reward (an extra DA spike)
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
        for pop in ['S', 'A', 'B', 'exc', 'inh']:
            self.events[pop] = nest.GetStatus(self.spkdet[pop], 'events')[0]
            nest.SetStatus(self.spkdet[pop], {'n_events' : 0 })
        self.EE_fr_hist = self._EE_fr_histogram(nest.GetStatus(self.spkdet['E'], 'events')[0])
        nest.SetStatus(self.spkdet['E'], {'n_events' : 0 })

        # TODO: delete DA spike detector once we know that this is bug free        
        #n_events_DA = nest.GetStatus(self.spkdet['DA'], 'n_events')[0]
        #nest.SetStatus(self.spkdet['DA'], {'n_events' : 0 })

        return A_minus_B #, n_events_DA

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
        for pop in ['S', 'A', 'B', 'exc', 'inh', 'E']:
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
        recvbuf = np.empty(self.n_procs, dtype='f') if self.rank == 0 else None
        self.mpi_comm.Gather(total_weight, recvbuf, root=0)
        if self.rank == 0:
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
        beg = time()
        cnnmat = self._E_pop_connectivity_matrix_old()
        print(time() - beg)
        quit()
        return cnnmat
    
    def _E_pop_connectivity_matrix(self):
        # First create a map from global ids to matrices index
        prev_max_gid = 0
        gids, inds = list(), dict()
        for pop in ['S', 'A', 'B', 'exc']:
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
    
    def _E_pop_connectivity_matrix_old(self):
        cnn_matrix = np.zeros((self.N['cnn_mat'], self.N['cnn_mat']))
        n_pre_ind = -1
        for pop_pre in ['S', 'A', 'B', 'exc']:
            for n_pre in self.nodes[pop_pre]:
                n_post_ind = -1
                n_pre_ind += 1
                for pop_post in ['S', 'A', 'B', 'exc']:
                    for n_post in self.nodes[pop_post]:
                        n_post_ind += 1
                        cnns = nest.GetConnections([n_pre], [n_post])
                        weights = nest.GetStatus(cnns, 'weight')
                        cnn_matrix[n_pre_ind, n_post_ind] = np.sum(weights)
        return sparse.coo_matrix(cnn_matrix)
    



def print_r0(*args):
    if nest.Rank() == 0:
        print(*args)