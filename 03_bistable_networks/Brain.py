import nest, utils
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt


from BaseBrainStructure import BaseBrainStructure
from Cortex import Cortex
from Striatum import Striatum

class Brain(BaseBrainStructure):
    def __init__(self, master_seed, **args):
        super().__init__(scale=1, **args)

        # Default neuron parameters
        tauSyn = 0.5  # synaptic time constant in ms
        self.neuron_params = {
                "C_m": 250.,  # capacitance of membrane in in pF
                "tau_m": 20.,  # time constant of membrane potential in ms
                "tau_syn_ex": tauSyn,
                "tau_syn_in": tauSyn,
                "E_L": 0.0,
                "V_reset": 0.0,
                "V_m": 0.0,
                "V_th": 20.,  # membrane threshold potential in mV
                }
        
        # Synapse parameters
        self.syn_delay = 1.5  # synaptic delay in ms
        self.ctx_str_indegree = 1000 # cortical boutons per MSN
        
        # Kernel parameters
        self.dt = .1
        self.verbosity = 20
        self.kernel_pars = {
            'print_time' : False,
            'resolution' : self.dt,
            'local_num_threads' : 12,
            'grng_seed' : master_seed,
            }
        
        # Define structures in the Brain
        self.cortex = Cortex(neu_params=self.neuron_params, scale=self.scale)
        self.striatum = Striatum(scale=self.scale)
        self.structures = [self.cortex, self.striatum]


    def build_local_network(self):
        # Configure kernel and threads
        self._configure_kernel()

        # Create default neuron and synapse (will be used by the structures bellow)
        nest.CopyModel('iaf_psc_alpha', 'default_neuron', self.neuron_params)
        nest.CopyModel('static_synapse', 'default_synapse', {'delay': self.syn_delay})

        # Create neurons from structures of the brain
        for struct in self.structures:
            struct.build_local_network()
        
        # Connect cortex to striatum
        nest.Connect(
            self.cortex.neurons['E'], 
            self.striatum.neurons['ALL'],
            {'rule': 'fixed_indegree', 'indegree': self.ctx_str_indegree},
            'cortex_E_synapse'
            )
        
    def _configure_kernel(self):
        # Threads and MPI processes
        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_procs = self.mpi_comm.Get_size()
        v_procs = self.mpi_procs * self.kernel_pars['local_num_threads']

        # Internal random number generator (RNG) for NEST (i.e. used by the kernel)
        mid_seed = self.kernel_pars['grng_seed'] + 1 + v_procs
        self.kernel_pars['rng_seeds'] = range(self.kernel_pars['grng_seed'] + 1, mid_seed)

        # RNGs for the user (i.e used by this script)        
        BaseBrainStructure.py_rngs = \
            [np.random.RandomState(seed) for seed in range(mid_seed, mid_seed + v_procs)]

        # Configure kernel
        nest.ResetKernel()
        nest.set_verbosity(self.verbosity)
        nest.SetKernelStatus(self.kernel_pars)
    
    def TEST_reinforce(self, sp_stim, sp_act):
        source = self.cortex.neurons[sp_stim]
        target = self.striatum.neurons[sp_act]
        cnns = nest.GetConnections(source, target)
        nest.SetStatus(cnns, params='weight', val=40.)

    def Simulate(self):
        self.TEST_reinforce('L', 'A')
        self.TEST_reinforce('H', 'B')
        self.cortex.stimulate_subpopulation('L', 10000.)
        
        nest.Simulate(20000.)
        cortex_events = nest.GetStatus(self.cortex.spkdets['L'], 'events')[0]
        cortex_events_2 = nest.GetStatus(self.cortex.spkdets['E_rec'], 'events')[0]
        str_A_events = nest.GetStatus(self.striatum.spkdets['A'], 'events')[0]
        str_B_events = nest.GetStatus(self.striatum.spkdets['B'], 'events')[0]
        
        plt.style.use('ggplot')
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(14, 14))
        fig.suptitle('w = ' + str(self.striatum.w), size=16., weight='bold')
        
        axes[0].set_title('Cortex L')
        axes[0].scatter(cortex_events['times']/1000., cortex_events['senders'], marker='.', s=3)

        axes[1].set_title('Cortex control')
        axes[1].scatter(cortex_events_2['times']/1000., cortex_events_2['senders'], marker='.', s=3)
        
        str_A_sample = self.striatum.neurons['A'][:500]
        strA_snd, strA_t = utils.get_raster_data(str_A_events, str_A_sample)
        axes[2].set_title('Striatum (sub-network A)')
        axes[2].scatter(strA_t/1000., strA_snd - np.min(strA_snd), marker='.', s=3)
        
        str_B_sample = self.striatum.neurons['B'][:500]
        strB_snd, strB_t = utils.get_raster_data(str_B_events, str_B_sample)
        axes[3].set_title('Striatum (sub-network B)')
        axes[3].scatter(strB_t/1000., strB_snd - np.min(strB_snd), marker='.', s=3)
        axes[3].set_xlabel('time (s)')
        
        plt.show()


        #print('\n', cortex_events, cortex_events/self.structures['cortex'].N['E_rec'] / t)
        #print(str_A_events, str_A_events / self.structures['striatum'].N['A'] / t)
        #print(str_B_events, str_B_events / self.structures['striatum'].N['B'] / t)
        #ratio =  str_A_events/str_B_events
        #print('ratio', ratio)