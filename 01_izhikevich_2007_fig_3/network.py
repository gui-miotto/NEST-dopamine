import nest
import numpy as np
import matplotlib, os
import matplotlib.pyplot as plt

class Network():
    def __init__(self, ext_thr_ratio=.87, master_seed=42, A_plus=2., A_minus=1.5, Wmax=5.):
        """[summary]
        Model parameters
        No nest function is called here
        pops = ['S', 'A', 'B', 'exc', 'inh', 'DA', 'E', 'I', 'ALL']
        Returns:
            [type] -- [description]
        """

        # Number of neurons
        self.N = dict()  # number of neurons
        self.N['I'] = 250  # number of inhibitory neurons 
        self.N['E'] = 4 * self.N['I']  # number of excitatory neurons
        self.N['ALL'] = self.N['I'] + self.N['E'] # total number of neurons
        self.N['S'] = 50
        self.N['A'] = self.N['B'] = 50 # Subpopulations to be recorded
        self.N['exc'] = self.N['inh'] = 50 # Subpopulations to be recorded
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
     
        # Dopamine modulation parameters
        self.DA_pars = {
            'weight' : self.J['E'],  # Default 1.
            'b' : 0., # Default 0.; Dopaminergic baseline concentration
            'A_plus' : A_plus * self.J['E'],  # Default 1.; Amplitude of weight change for facilitation
            'tau_c' : 1000., # Default 1000.,  # Time constant of eligibility trace in ms
            'tau_n' : 200., # Default 200.; Time constant of dopaminergic trace in ms
            'tau_plus' : 20.0, #  Default 20.; STDP time constant for facilitation in ms
            'Wmin' : 0., # Default 0. # Minimal synaptic weight
            'Wmax' : self.J['E'] * Wmax, # Maximal synaptic weight  
            'delay': self.delay, # Default 1.; Synaptic delay
            #'vt' : volt_DA[0], # Volume transmitter will be assigned later on
            }
        self.DA_pars['A_minus'] = A_minus * self.DA_pars['A_plus'] # Default 1.5; Amplitude of weight change for depression
        
        # External noise parameters
        eta = ext_thr_ratio # 2. # external rate relative to threshold rate
        nu_th = self.neu_pars['V_th'] / (self.J['E'] * self.C['E'] * self.neu_pars['tau_m'])
        nu_ex = eta * nu_th
        self.p_rate = 1000.0 * nu_ex * self.C['E']  

        # Kernel parameters
        self.mseed = master_seed  # master seed
        self.dt = .1
        self.verbosity = 20
        self.kernel_pars = {
            'print_time' : False,
            'overwrite_files' : True,
            'local_num_threads' : 12,
            'resolution' : self.dt,
            }

        # Experiment parameters
        self.trial_duration = 6000. #10000.0  # Trial duration
        self.cue_duration = 3. # Cue stimulus duration
        self.cue_stim = 315. #300 # Stimulus delivered to cue neurons in pF
        self.eval_time_window = 20. # Time window to check response via spike count
        self.max_DA_wait_time = 1000. # Maximum waiting time to reward
        

    def _configure_kernel(self):
        nest.ResetKernel()
        nest.set_verbosity(20)
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
        nest.Connect(self.nodes['DA'], volt_DA)
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

        # Create and background noise
        noise = nest.Create("poisson_generator", params={"rate": self.p_rate}) # Background noise
        nest.Connect(noise, self.nodes['ALL'], syn_spec="static_E")

        # Create and connect sensory stimulus
        self.sensory_drive = nest.Create("step_current_generator")
        nest.Connect(self.sensory_drive, self.nodes['S'])

        # Create and connect spike detectors:
        self.spkdet = {}
        for pop in ['S', 'A', 'B', 'exc', 'inh', 'DA', 'E', 'I']:
            self.spkdet[pop] = nest.Create("spike_detector")
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
        

    def simulate(self, n_trials=400, plot=False, save_plot_dir=None, syn_scaling=True):
        # (warning: nest.ResetNetwork wont reboot synaptic weights, 
        # we would have to do it manually or just rebuild the network)
        self._build_network()
        if save_plot_dir is not None and not os.path.exists(save_plot_dir):
            os.mkdir(save_plot_dir)
        
        A_sel, B_sel, draw = 0, 0, 0
        all_selections = []
        S_to_pop_mean_w = {'A' : [], 'B' : []}
        S_to_pop_syn_ids = {pop : nest.GetConnections(self.nodes['S'], self.nodes[pop]) for pop in ['A', 'B']}
        plastic_syn_ids = nest.GetConnections(self.nodes['E'], self.nodes['E'])
        initial_total_plastic_w = self.J['E'] * self.N['E'] * self.C['E']
        print('simulation is about to start')
        print(f'Initial total plastic weight: {initial_total_plastic_w}\n')

        for trial in range(1, n_trials+1):
            print(f'Simulating trial {trial} of {n_trials}:')
            A_minus_B, decision_spikes, events, n_events = self._simulate_one_trial()
            print(f'Spikes S: {decision_spikes["S"]}')
            print(f'Spikes A: {decision_spikes["A"]}, spikes B: {decision_spikes["B"]}')
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

            # Synapses weights and weight rescaling:
            old_weights, new_weights, rescal_factor = self._synaptic_rescaling(plastic_syn_ids, initial_total_plastic_w, syn_scaling)
            total_plastic_ws = np.sum(old_weights)
            print('Current total weight:', total_plastic_ws)
            if syn_scaling:
                print('scaling it by a fator of', rescal_factor)
            S_to_pop_w = dict()
            for pop in ['A', 'B']:
                S_to_pop_w[pop] = nest.GetStatus(S_to_pop_syn_ids[pop], 'weight')
                mean_w = np.mean(S_to_pop_w[pop])
                print(f'Mean weight S to {pop}: {mean_w}')
                S_to_pop_mean_w[pop].append(mean_w)
            
            # Show raster plots, histograms, etc...
            if plot:
                self._plot(trial, A_minus_B, events, n_events, new_weights, \
                     all_selections, S_to_pop_mean_w, S_to_pop_w, save_plot_dir)

            print(f'Parcial results (out of {trial} trials):')
            print(f'{A_sel} correct selections ({A_sel*100./trial}%)')
            print(f'{B_sel} wrong selections ({B_sel*100./trial}%)')
            print(f'{draw} draws ({draw*100./trial}%)\n')

        return A_sel / n_trials


    def _simulate_one_trial(self):
        # Program stimulus
        curr_time = nest.GetKernelStatus()['time']
        cue_onset = curr_time + self.dt
        nest.SetStatus(self.sensory_drive, params={
            'amplitude_times' : [cue_onset, cue_onset + self.cue_duration],
            'amplitude_values' : [self.cue_stim, 0.],
        })

        # Simulate the evaluation windown
        nest.Simulate(self.eval_time_window)
        curr_time = nest.GetKernelStatus()['time']

        # Read response spike detectors
        decision_spikes = {pop : nest.GetStatus(self.spkdet[pop], 'n_events')[0] for pop in ['S', 'A', 'B']}
        
        # If the response is correct, then program DA response
        A_minus_B = decision_spikes['A'] - decision_spikes['B']  # TODO use the walrus operator with python 3.8
        if A_minus_B > 0:
            wait_time = self.max_DA_wait_time - (A_minus_B - 1) * 100.
            wait_time = wait_time if wait_time > 100. else 100.
            wait_time = wait_time if wait_time < self.max_DA_wait_time else self.max_DA_wait_time
            delivery_time = curr_time + wait_time 
            nest.SetStatus(self.nodes['DA'], params={'spike_times' : [delivery_time]})

        # Simulate the rest of the trial
        nest.Simulate(self.trial_duration - self.eval_time_window)

        # Read events
        events, n_events = dict(), dict()
        for pop in ['S', 'A', 'B', 'exc', 'inh', 'DA', 'E', 'I']:
            n_events[pop] = nest.GetStatus(self.spkdet[pop], 'n_events')[0]
            events[pop] = nest.GetStatus(self.spkdet[pop], 'events')[0]
            nest.SetStatus(self.spkdet[pop], {'n_events' : 0 })

        return A_minus_B, decision_spikes, events, n_events

    def _synaptic_rescaling(self, all_plastic_syn, initial_total_plastic_w, rescal=True):
        old_weights = nest.GetStatus(all_plastic_syn, 'weight')
        old_total_weight = np.sum(old_weights)
        if not rescal:
            return old_weights, old_weights, 1.
        syn_rescal_factor = initial_total_plastic_w / old_total_weight
        new_weights = np.array(old_weights) * syn_rescal_factor
        nest.SetStatus(all_plastic_syn, params='weight', val=new_weights)
        return old_weights, new_weights, syn_rescal_factor


    def _plot(self, trial_num, A_minus_B, events, n_events, curr_plastic_w,\
         selections, S_to_pop_syn_w, S_to_pop_w, figdir=None):

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

        # raster plot decision period
        plt.subplot(3,4,1)
        plt.title('Decision period raster plot')
        prev_max_snd, times, shifted_snds = 0, {}, {}
        for pop in ['S', 'A', 'B', 'exc', 'inh', 'DA']:
            if n_events[pop] == 0:
                continue
            snds = np.array(events[pop]['senders'])
            times[pop] = events[pop]['times']
            shifted_snds[pop] = snds - np.min(snds) + prev_max_snd + 1
            plt.scatter(times[pop], shifted_snds[pop], marker='o', s=5., label=pop)
            prev_max_snd = np.max(shifted_snds[pop])
        t_min = times['S'][0] - 0.1 * self.eval_time_window
        t_max = times['S'][0] + 1.1 * self.eval_time_window
        plt.xlim(t_min, t_max)
        plt.xlabel('time (ms)')

        # full trial raster plot
        plt.subplot(3,4,(2,3))
        plt.title('Full trial raster plot')
        prev_max_snd = 0
        for pop in times.keys():
            s = 16. if pop == 'DA' else 5.
            mkr = '*' if pop == 'DA' else '.'
            plt.scatter(times[pop], shifted_snds[pop], marker=mkr, s=s, label=pop)
        plt.legend(loc='upper right')
        plt.xlabel('time (ms)')

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

        # Probability of action selection plot
        plt.subplot(3, 4, (6,7))
        plt.title('Probability of action selection')
        bin_size = 25
        plot_data = {'T' : [], 'A' : [], 'B' : [], 'none' : []}
        for trial_n in range(0, len(selections), bin_size):
            sels = np.array(selections[trial_n : trial_n+bin_size])
            plot_data['T'].append(trial_n)
            for res in ['A', 'B', 'none']:
                ind = np.where(sels==res)[0]
                plot_data[res].append(len(ind))
        for res in ['A', 'B', 'none']:
            y_percentage = np.array(plot_data[res]) / bin_size
            plt.plot(plot_data['T'], y_percentage, marker='.', label=res)
        plt.legend(loc='lower left')
        plt.xlabel('trial')

        # E-E weights histogram
        plt.subplot(3, 4, 8)
        plt.title('E-E weights')
        weight_bins = np.linspace(0, self.DA_pars['Wmax'], 20)
        plt.hist(curr_plastic_w, bins=weight_bins, log=True, bottom=0.)
        plt.ylim(1., self.N['E'] * self.C['E'] * 10)
        plt.xlabel('weight (mV)')

        # Mean weight between S and A or B
        plt.subplot(3, 4, (10,11))
        plt.title('Mean weight on direct connections between S and A or B')
        for pop in ['A', 'B']:
            plt.plot(S_to_pop_syn_w[pop], label='S to ' + pop)
        plt.legend(loc='upper left')
        plt.xlabel('trial')
        plt.ylabel('weight (mV)')

        # S-pop weights histogram
        plt.subplot(3, 4, 12)
        plt.title('S-A and S-B weights')
        for pop in ['A', 'B']:
            plt.hist(S_to_pop_w[pop], label='S to ' + pop, bins=weight_bins, log=True, alpha=.65)
        #plt.hist([S_to_pop_w['A'], S_to_pop_w['B']], label='S to ' + pop,\
            # bins=weight_bins, log=True)
        plt.ylim(1., self.N['A'] * self.C['S'] * 10)
        plt.xlabel('weight (mV)')
        plt.legend(loc='upper center')
        
        if figdir is None:
            #fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.001)
            fig.canvas.start_event_loop(0.001)
            #plt.pause(0.001)
            #plt.pause(0.001) # Needed the second pause (strange behavior)
            #plt.show()
        else:
            fig_file = 'trial_' + trial_num_str + '.png'
            plt.savefig(os.path.join(figdir, fig_file), transparent=False)
            plt.close(fig)









