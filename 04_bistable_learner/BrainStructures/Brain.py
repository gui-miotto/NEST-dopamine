import nest, multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import BrainStructures as BS
from itertools import product
from mpi4py import MPI
from copy import deepcopy


class Brain(BS.BaseBrainStructure):
    """Abstraction of a trainable brain. A brain is made of a cortex, a striatum and a VTA. Synapses
    between those areas are handled by this class. Synapses between the cortex and striataum are 
    excitatory, plastic and modulated by the dopamine of the VTA.
    """
    def __init__(self, master_seed, **args):
        super().__init__(**args)

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
        
        # Default synapse parameters
        self.J = 20.  # amplitude of excitatory postsynaptic current 
        self.syn_delay = 1.5  # synaptic delay in ms
        
        # Kernel parameters
        self.dt = .1
        self.verbosity = 20
        self.kernel_pars = {
            'print_time' : False,
            'resolution' : self.dt,
            #'local_num_threads' : multiprocessing.cpu_count(),
            'local_num_threads' : 4,
            'grng_seed' : master_seed,
            }

        # Define structures in the Brain
        self.cortex = BS.Cortex(neu_params=self.neuron_params, J_E=self.J, scale=self.scale)
        self.striatum = BS.Striatum(C_E=self.cortex.C['E'], J_I=self.cortex.J['I'], scale=self.scale)
        self.vta = BS.VTA(dt=self.dt, J_E=self.J, syn_delay=self.syn_delay, scale=self.scale)
        self.structures = [self.cortex, self.striatum, self.vta]

        # MPI communication
        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_procs = self.mpi_comm.Get_size()

    
    def _configure_kernel(self):
        # Internal random number generator (RNG) for NEST (i.e. used by the kernel)
        v_procs = self.mpi_procs * self.kernel_pars['local_num_threads']
        mid_seed = self.kernel_pars['grng_seed'] + 1 + v_procs
        self.kernel_pars['rng_seeds'] = range(self.kernel_pars['grng_seed'] + 1, mid_seed)

        # RNGs for the user (i.e used by these scripts)
        BS.BaseBrainStructure.py_rngs = \
            [np.random.RandomState(seed) for seed in range(mid_seed, mid_seed + v_procs)]

        # Configure kernel
        nest.ResetKernel()
        nest.set_verbosity(self.verbosity)
        nest.SetKernelStatus(self.kernel_pars)

    
    def build_local_network(self):
        # Configure kernel and threads
        self._configure_kernel()

        # Create default neuron and synapse (will be used by the structures bellow)
        nest.CopyModel('iaf_psc_alpha', 'default_neuron', self.neuron_params)
        nest.CopyModel('static_synapse', 'default_synapse', {
            'delay' : self.syn_delay,
            'weight' : self.J,
            })

        # Create neurons from structures of the brain
        for struct in self.structures:
            struct.build_local_network()
            self.spkdets.update(struct.spkdets)
            #TODO: there shouldnt be repeated keys here. assure that
        
        # Connect cortex to striatum in a balanced way (We wouldnt need to be so careful if the 
        # network was larger, because the chance of having big percentual differences in connecivity
        # between suppopulations would be smaller)
        for source, target in product(['low', 'high', 'E_no_S'], ['left', 'right']):
            nest.Connect(
                self.cortex.neurons[source], 
                self.striatum.neurons[target],
                {'rule': 'fixed_indegree', 'indegree': self.cortex.C[source]},
                'corticostriatal_synapse'
                )
        
        # Get connections for later weight monitoring
        self.w_ind = ['low', 'high', 'E_rec', 'E']
        self.w_col = ['left', 'right', 'ALL']
        self.synapses = pd.DataFrame(index=self.w_ind, columns=self.w_col)
        self.weights_count = deepcopy(self.synapses)
        self.weights_mean_ = deepcopy(self.synapses) 
        self.weights_hist_ = deepcopy(self.synapses)
        for source, target in product(self.w_ind, self.w_col):
            cnns = nest.GetConnections(self.cortex.neurons[source], self.striatum.neurons[target])
            self.synapses.loc[source, target] = cnns
            self.weights_count.loc[source, target] = len(cnns)
        self.initial_total_weight = self.striatum.N['ALL'] * self.cortex.C['E'] * self.J


    def read_synaptic_weights(self):
        for source, target in product(self.w_ind, self.w_col):
            weights = nest.GetStatus(self.synapses.loc[source, target], 'weight')
            self.weights_mean_.loc[source, target] = np.mean(weights)
            self.weights_hist_.loc[source, target] = np.histogram(
                weights, bins=21, range=(0., self.vta.DA_pars['Wmax']))[0]
    

    def reset_corticostriatal_synapses(self):
        nest.SetStatus(self.synapses.loc['E', 'ALL'], params='weight', val=self.J)


    def rescale_corticostriatal_synapses(self):
        self.syn_rescal_factor_, old_total_weight, new_weights = self.get_weight_scaling_factor()
        nest.SetStatus(self.synapses.loc['E', 'ALL'], params='weight', val=new_weights)
        return old_total_weight


    def get_weight_scaling_factor(self):
        weights = nest.GetStatus(self.synapses.loc['E', 'ALL'], 'weight')
        total_weight = np.sum(weights, dtype='f')
        recvbuf = np.empty(self.mpi_procs, dtype='f') if self.mpi_rank == 0 else None
        self.mpi_comm.Gather(total_weight, recvbuf, root=0)
        if self.mpi_rank == 0:
            total_weight = np.sum(recvbuf)
            syn_rescal_factor = self.initial_total_weight / total_weight
        else:
            syn_rescal_factor = None
        syn_rescal_factor = self.mpi_comm.bcast(syn_rescal_factor, root=0)
        scaled_weights = np.array(weights) * syn_rescal_factor
        return syn_rescal_factor, total_weight, scaled_weights

