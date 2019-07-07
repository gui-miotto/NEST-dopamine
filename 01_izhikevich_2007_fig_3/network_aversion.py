import nest
import numpy as np
import matplotlib, os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from numpy.polynomial.polynomial import polyfit
from math import exp as exp

class Network():
    def __init__(self, master_seed=42, A_plus_mult=2., A_minus_mult=1.5, Wmax_mult=5.):
        """[summary]
        Model parameters
        No nest function is called here
        pops = ['S', 'A', 'B', 'exc', 'inh', 'DA', 'E', 'I', 'ALL']
        Returns:
            [type] -- [description]
        """

        self.debug = True

        # Kernel parameters
        self.mseed = master_seed  # master seed
        self.dt = .1
        self.verbosity = 20
        self.kernel_pars = {
            'print_time' : False,
            'overwrite_files' : True,
            'local_num_threads' : 1,
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
            'n' : 1. / self.dt + 1.5 / self.tau_n, # Default 0.; Initial dopamine concentration
            'A_plus' : A_plus,  # Default 1.; Amplitude of weight change for facilitation
            'A_minus' : A_minus,  # Default 1.5; Amplitude of weight change for depression
            'Wmax' : self.J['E'] * Wmax_mult, # Maximal synaptic weight  
            #'tau_c' : 1000., # Default 1000.,  # Time constant of eligibility trace in ms
            #'tau_plus' : 20.0, #  Default 20.; STDP time constant for facilitation in ms
            #'Wmin' : 0., # Default 0. # Minimal synaptic weight
            #'vt' : volt_DA[0], # Volume transmitter will be assigned later on
            }
        
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

        # Create random number generators
        # (2 * num_of_virtual_processes + 1)
        # (i.e. : One global NEST RNG + num_of_VPs RNGs for python + num_of_VPs RNGs for NEST)
        n_vp = nest.GetKernelStatus('total_num_virtual_procs')
        msd_range1 = range(self.mseed, self.mseed + n_vp)
        msd_range2 = range(self.mseed + 1 + n_vp, self.mseed + 1 + 2 * n_vp)
        self.py_rngs = [np.random.RandomState(seed) for seed in msd_range1]
        nest.SetKernelStatus({
                'grng_seed' : self.mseed + n_vp,
                'rng_seeds' : msd_range2,
                })
    
    def _build_network(self):
        self._configure_kernel()

        # Create all neurons
        nest.SetDefaults('iaf_psc_delta', self.neu_pars)
        self.nodes = {pop : nest.Create("iaf_psc_delta", self.N[pop]) for pop in ['E', 'I']}

        # Some usefull aliases:
        self.nodes['ALL'] = self.nodes['E'] + self.nodes['I']
        self.nodes['inh'] = self.nodes['I'][:self.N['inh']]
        cut = 0
        for pop in ['S', 'A', 'B', 'exc']:  # line bellow assumes that S is the first cut (better not to change this order)
            self.nodes[pop] = self.nodes['E'][cut : cut+self.N[pop]]
            cut += self.N[pop]
        self.nodes['E_wout_S'] = self.nodes['E'][self.N['S']:]

        # Initiate membrane potentials randomly
        v_min, v_max = self.neu_pars['V_reset'], self.neu_pars['V_th']
        node_info = nest.GetStatus(self.nodes['ALL'])
        local_nodes = [(ni['global_id'], ni['vp']) for ni in node_info if ni['local']]
        for gid, vp in local_nodes:
            nest.SetStatus([gid], {'V_m': self.py_rngs[vp].uniform(v_min, v_max)})

        # Create and connect volume transmiter
        self.nodes['DA'] = nest.Create('spike_generator')
        volt_DA = nest.Create("volume_transmitter")
        nest.Connect(self.nodes['DA'], volt_DA, syn_spec={'delay' : self.dt})
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
        for pop in ['S', 'A', 'B', 'exc', 'inh', 'DA', 'E', 'I']:
            self.spkdet[pop] = nest.Create('spike_detector')
            nest.Connect(self.nodes[pop], self.spkdet[pop])

    
    def static_test(self):
        # (warning: nest.ResetNetwork wont reboot synaptic weights, 
        # we would have to do it manually or just rebuild the network)      
        self._build_network()
        
        nest.Simulate(self.trial_duration)
        print(f'Trial duration: {self.trial_duration / 1000.} (s)', )

        # Check firing rates
        n_events, fr = dict(), dict()
        for pop in ['exc', 'inh']:
            n_events[pop] = nest.GetStatus(self.spkdet[pop], 'n_events')[0]
            fr[pop] = n_events[pop] * 1000.0 / self.trial_duration / self.N[pop]
            print(f'Population {pop} mean firing rate: {fr[pop]} (hz)')
        
        return fr['exc']
        

    def simulate(self, n_trials=400, syn_scaling=True, aversion=True, plot=False, save_plot_dir=None):
        # (warning: nest.ResetNetwork wont reboot synaptic weights, 
        # we would have to do it manually or just rebuild the network)
        self._build_network()
        if save_plot_dir is not None and not os.path.exists(save_plot_dir):
            os.mkdir(save_plot_dir)
        
        A_sel, B_sel, draw = 0, 0, 0
        all_selections, scale_factors = [], []
        S_to_pop_mean_w = {'A' : [], 'B' : []}
        
        print(f'Initial total plastic weight: {self.initial_total_plastic_w}\n')
        warmup_duration = (1. if self.debug else 25.) * self.tau_n
        print(f'Simulating warmup for {warmup_duration} ms')
        self._run_warmup(warmup_duration)

        for trial in range(1, n_trials+1):
            print(f'Simulating trial {trial} of {n_trials}:')
            trial_begin, A_minus_B, decision_spikes, events, n_events = self._simulate_one_trial(aversion=aversion)
            print(f'Spikes S: {decision_spikes["S"]}')
            print(f'Spikes A: {decision_spikes["A"]}, spikes B: {decision_spikes["B"]}')

            if self.debug:            
                print('DA spikes:', n_events['DA'])
            
            if A_minus_B > 0:
                print('right response! :-)')
                A_sel += 1
                all_selections.append('A')
            elif A_minus_B < 0:
                print('wrong response. :-(')
                B_sel += 1
                all_selections.append('B')
            else:
                print('Draw. :-|')
                draw += 1
                all_selections.append('none')

            # Print firing rates            
            for pop in ['S', 'A', 'B', 'exc', 'inh', 'DA']:
                fr = n_events[pop] * 1000.0 / self.trial_duration / self.N[pop]
                print(f'{pop} neuron mean firing rate: {fr} (hz)')

            # Weight rescaling:
            old_weights, new_weights, rescal_factor = self._synaptic_rescaling(syn_scaling)
            total_plastic_ws = np.sum(old_weights)
            scale_factors.append(rescal_factor)
            print('End-of-trial total weight:', total_plastic_ws)
            if syn_scaling:
                print('scaling it by a fator of', rescal_factor)
            
            # Mean weight between population S and population A and B
            S_to_pop_w = dict()
            for pop in ['A', 'B']:
                S_to_pop_w[pop] = nest.GetStatus(self.S_to_pop_syns[pop], 'weight')
                mean_w = np.mean(S_to_pop_w[pop])
                print(f'Mean weight S to {pop}: {mean_w}')
                S_to_pop_mean_w[pop].append(mean_w)

            # Connectivity matrix
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

            # Show raster plots, histograms, etc...
            if plot:
                self._plot(trial, trial_begin, A_minus_B, events, n_events, new_weights, \
                    all_selections, S_to_pop_mean_w, S_to_pop_w, cnn_matrix, \
                    scale_factors, save_plot_dir)

            print(f'Parcial results (out of {trial} trials):')
            print(f'{A_sel} correct selections ({A_sel*100./trial}%)')
            print(f'{B_sel} wrong selections ({B_sel*100./trial}%)')
            print(f'{draw} draws ({draw*100./trial}%)\n')

        return A_sel / n_trials


    def _simulate_one_trial(self, aversion=True):
        # Program stimulus
        trial_begin = nest.GetKernelStatus()['time']
        cue_onset = trial_begin + self.dt
        nest.SetStatus(self.sensory_drive, params={
            'amplitude_times' : [cue_onset, cue_onset + self.cue_duration],
            'amplitude_values' : [self.cue_stim, 0.],
        })

        #DEBUG
        #ts, ns = [], []
        #ts.append(trial_begin)
        #ns.append(nest.GetStatus(self.plastic_syns[:1], 'n')[0])


        # Program dopamine baseline
        DA_spike_times = np.arange(
            trial_begin + self.dt, 
            trial_begin + self.dt + self.eval_time_window,
            self.dt)
        DA_spike_times = np.round(DA_spike_times, decimals=1)
        nest.SetStatus(self.nodes['DA'], params={'spike_times' : DA_spike_times})

        # Simulate the evaluation windown
        nest.Simulate(self.eval_time_window)
        curr_time = nest.GetKernelStatus()['time']

        # Read response spike detectors
        decision_spikes = {pop : nest.GetStatus(self.spkdet[pop], 'n_events')[0] for pop in ['S', 'A', 'B']}
        
        # According to the selected action, deliver the appropriate DA response
        A_minus_B = decision_spikes['A'] - decision_spikes['B']
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
        #ttt = np.arange(curr_time, curr_time + self.trial_duration - self.eval_time_window, 20.)
        #for tt in ttt:
        #    nest.Simulate(20.)
        #    ts.append(tt)
        #    ns.append(nest.GetStatus(self.plastic_syns[:1], 'n')[0])
        #ns = np.array(ns) - nest.GetDefaults('plastic_E', 'b')
        #plt.plot(ts[1:], ns[1:])
        #plt.grid()
        #plt.show()

        # Read events
        events, n_events = dict(), dict()
        for pop in ['S', 'A', 'B', 'exc', 'inh', 'DA', 'E', 'I']:
            n_events[pop] = nest.GetStatus(self.spkdet[pop], 'n_events')[0]
            events[pop] = nest.GetStatus(self.spkdet[pop], 'events')[0]
            nest.SetStatus(self.spkdet[pop], {'n_events' : 0 })

        return trial_begin, A_minus_B, decision_spikes, events, n_events

    def _synaptic_rescaling(self, rescal=True):
        
        old_weights = nest.GetStatus(self.plastic_syns, 'weight')
        old_total_weight = np.sum(old_weights)
        if not rescal:
            return old_weights, old_weights, 1.
        syn_rescal_factor = self.initial_total_plastic_w / old_total_weight
        new_weights = np.array(old_weights) * syn_rescal_factor
        nest.SetStatus(self.plastic_syns, params='weight', val=new_weights)
        return old_weights, new_weights, syn_rescal_factor

    def _run_warmup(self, warmup_duration=100.):
        DA_spike_times = np.arange(self.dt, warmup_duration + self.dt, self.dt)
        DA_spike_times = np.round(DA_spike_times, decimals=1)
        nest.SetStatus(self.nodes['DA'], params={'spike_times' : DA_spike_times})
        nest.Simulate(warmup_duration)
        initial_weights = [self.J['E']] * (self.N['E'] * self.C['E'])
        nest.SetStatus(self.plastic_syns, params='weight', val=initial_weights)
        for pop in ['S', 'A', 'B', 'exc', 'inh', 'DA', 'E', 'I']:
            nest.SetStatus(self.spkdet[pop], {'n_events' : 0 })


    def _plot(self, trial_num, trial_begin, A_minus_B, events, n_events, curr_plastic_w,\
         selections, S_to_pop_syn_w, S_to_pop_w, cnn_matrix, scale_factors, figdir=None):

        if figdir is None:
            plt.ion()
            plt.close()
        
        plt.style.use('ggplot')
        #plt.style.use('seaborn')
        fig = plt.figure(figsize=(15., 9.))
        plt.subplots_adjust(wspace = 0.25, hspace = 0.4, left=0.1, right=0.95, bottom=.07)
        trial_num_str = str(trial_num).rjust(3, '0')
        sel_act = 'action A' if A_minus_B > 0 else 'action B' if A_minus_B < 0 else 'no action'
        plt.suptitle(f'Trial {trial_num_str}\n{sel_act}', size=16., weight='normal')

        # decision period raster plot 
        plt.subplot(3,4,1)
        plt.title('Decision period raster plot')
        prev_max_snd, times, shifted_snds = 0, {}, {}
        for pop in ['S', 'A', 'B', 'exc', 'inh']:
            if n_events[pop] == 0:
                continue
            snds = np.array(events[pop]['senders'])
            times[pop] = events[pop]['times'] - trial_begin
            shifted_snds[pop] = snds - np.min(snds) + prev_max_snd + 1
            plt.scatter(times[pop], shifted_snds[pop], marker='o', s=5., label=pop)
            prev_max_snd = np.max(shifted_snds[pop])
        t_min = -.1 * self.eval_time_window
        t_max = 1.1 * self.eval_time_window
        plt.xlim(t_min, t_max)
        plt.xlabel('time (ms)')

        # full trial raster plot
        plt.subplot(3,4,(2,3))
        plt.title('Full trial raster plot')
        prev_max_snd = 0
        for pop in times.keys():
            plt.scatter(times[pop] / 1000., shifted_snds[pop], marker='.', s=5., label=pop)
        plt.legend(loc='upper right')
        plt.xlabel('time (s)')

        # firing rates histogram
        plt.subplot(3, 4, 4)
        plt.title('Excitatory firing rate')
        snds = events['E']['senders']
        times = np.array(events['E']['times'])
        frs = []
        for snd in np.unique(snds):
            snd_times = times[snds==snd]
            fr = len(snd_times) * 1000. / self.trial_duration
            frs.append(fr)
        plt.hist(frs, bins=20)
        plt.xlabel('firing rate (hz)')

        # E-E weights histogram
        plt.subplot(3, 4, 5)
        plt.title('E-E weights')
        weight_bins = np.linspace(0, self.DA_pars['Wmax'], 20)
        plt.hist(curr_plastic_w, bins=weight_bins, log=True, bottom=0.)
        plt.ylim(1., self.N['E'] * self.C['E'] * 10)
        plt.xlabel('weight (mV)')

        # S-pop weights histogram
        plt.subplot(3, 4, 8)
        plt.title('S-A and S-B weights')
        plt.hist([S_to_pop_w['A'], S_to_pop_w['B']], label=['S to A', 'S to B'], \
            bins=weight_bins, log=True)
        plt.ylim(1., self.N['A'] * self.C['S'] * 10)
        plt.xlabel('weight (mV)')
        plt.legend(loc='upper center')

        # connectivity matrix
        plt.subplot(3, 4, 12)
        plt.title('Connectivity matrix')
        plt.imshow(cnn_matrix, cmap=plt.get_cmap('hot'), origin='lower', \
            vmin= 0., vmax=self.DA_pars['Wmax'])
        plt.xticks([50, 100, 150, 200], ('S    ', 'A    ', 'B    ', 'exc  '), ha='right')
        plt.yticks([50, 100, 150, 200], ('\nS', '\nA', '\nB', '\nexc'), va='top')
        plt.xlabel('post-synaptic')
        plt.ylabel('pre-synaptic')

        # synaptic scaling factors
        plt.subplot(3, 4, 9)
        plt.title('Synaptic scaling factor')
        plt.plot(scale_factors)
        if len(scale_factors) > 1:
            x = np.arange(len(scale_factors))
            reg_line = np.poly1d(np.polyfit(x, scale_factors, 1))
            plt.plot(x, reg_line(x), label='lin reg')
            plt.legend()
        plt.ylim(.8, 1.2)
        plt.xlabel('trial')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # Mean weight between S and A or B
        plt.subplot(3, 4, (6, 7))
        plt.title('Mean weight on direct connections between S and A or B')
        for pop in ['A', 'B']:
            plt.plot(S_to_pop_syn_w[pop], label='S to ' + pop)
        plt.legend(loc='upper left')
        plt.xlabel('trial')
        plt.ylabel('weight (mV)')
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Probability of action selection plot
        plt.subplot(3, 4, (10, 11))
        plt.title('Probability of action selection')
        bin_size = 25
        plot_data = {'T' : [], 'A' : [], 'B' : [], 'none' : []}
        for trial_n in range(0, len(selections), bin_size):
            sels = np.array(selections[trial_n : trial_n+bin_size])
            plot_data['T'].append(trial_n)
            for res in ['A', 'B', 'none']:
                ind = np.where(sels==res)[0]
                plot_data[res].append(len(ind) / len(sels))
        for res in ['A', 'B', 'none']:
            plt.plot(plot_data['T'], plot_data[res], marker='.', label=res)
        plt.legend(loc='lower left')
        plt.xlabel('trial')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        
        if figdir is None:
            ##fig.canvas.draw_idle()
            ##fig.canvas.start_event_loop(0.001)
            ##fig.canvas.start_event_loop(0.001)
            plt.pause(0.001)
            plt.pause(0.001) # Needed the second pause (strange behavior)
            #plt.show()
        else:
            fig_file = 'trial_' + trial_num_str + '.png'
            plt.savefig(os.path.join(figdir, fig_file), transparent=False)
            plt.close(fig)

