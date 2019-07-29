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
        self.ctx_str_cnn = {'rule': 'fixed_indegree', 'indegree': 1000}  # cortical axons per MSN
        
        # Kernel parameters
        self.dt = .1
        self.verbosity = 20
        self.kernel_pars = {
            'print_time' : False,
            'resolution' : self.dt,
            'local_num_threads' : 4,
            'grng_seed' : master_seed,
            }
        
        # Define structures in the Brain
        self.structures = {
            'cortex'    : Cortex(neu_params=self.neuron_params, scale=self.scale),
            'striatum'  : Striatum(scale=self.scale),
            }

    def build_local_network(self):
        # Configure kernel and threads
        self._configure_kernel()

        # Create default neuron and synapse (will be used by the structures bellow)
        nest.CopyModel('iaf_psc_alpha', 'default_neuron', self.neuron_params)
        nest.CopyModel('static_synapse', 'default_synapse', {'delay': self.syn_delay})

        # Create neurons from cortex and striatum
        for _, struct in self.structures.items():
            struct.build_local_network()
        
        # Connect cortex to striatum
        nest.CopyModel('default_synapse', 'cortex_E_syn_A', {'weight': 20.})
        nest.CopyModel('default_synapse', 'cortex_E_syn_B', {'weight': 20.})
        for pop in ['A', 'B']:
            nest.Connect(
                self.structures['cortex'].neurons['E'],
                self.structures['striatum'].neurons[pop],
                self.ctx_str_cnn,
                'cortex_E_syn_'+pop,
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
    
    def Simulate(self):
        nest.Simulate(30000.)
        cortex_events = nest.GetStatus(self.structures['cortex'].spkdets['E_rec'], 'events')[0]
        str_A_events = nest.GetStatus(self.structures['striatum'].spkdets['A'], 'events')[0]
        str_B_events = nest.GetStatus(self.structures['striatum'].spkdets['B'], 'events')[0]

        #print('\n', cortex_events, cortex_events/self.structures['cortex'].N['E_rec'] / t)
        #print(str_A_events, str_A_events / self.structures['striatum'].N['A'] / t)
        #print(str_B_events, str_B_events / self.structures['striatum'].N['B'] / t)
        #ratio =  str_A_events/str_B_events
        #print('ratio', ratio)
        
        plt.style.use('ggplot')
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(18, 8))
        fig.suptitle('w = ' + str(self.structures['striatum'].w), size=16., weight='bold')
        
        axes[0].set_title('Cortex')
        axes[0].scatter(cortex_events['times']/1000., cortex_events['senders'], marker='.', s=3)
        
        str_A_sample = self.structures['striatum'].neurons['A'][:500]
        strA_snd, strA_t = utils.get_raster_data(str_A_events, str_A_sample)
        axes[1].set_title('Striatum (sub-network A)')
        axes[1].scatter(strA_t/1000., strA_snd - np.min(strA_snd), marker='.', s=3)
        
        str_B_sample = self.structures['striatum'].neurons['B'][:500]
        strB_snd, strB_t = utils.get_raster_data(str_B_events, str_B_sample)
        axes[2].set_title('Striatum (sub-network B)')
        axes[2].scatter(strB_t/1000., strB_snd - np.min(strB_snd), marker='.', s=3)
        axes[2].set_xlabel('time (s)')
        
        plt.show()
