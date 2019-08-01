import nest
import numpy as np
from itertools import product
from mpi4py import MPI
import BrainStructures as BS

class Striatum(BS.BaseBrainStructure):
    def __init__(self, C_E, **args):
        super().__init__(**args)
    
        # Number of neurons
        n = int(1.25 * C_E)  # neurons per subpopulation
        self.N['left'] = self.N['right'] = n
        self.N['ALL'] = self.N['left'] + self.N['right']

        # Connectivity
        epsilon = .1  # connection probability
        self.conn_params = {'rule': 'fixed_indegree', 'indegree': int(epsilon * n)} 

        # synapse parameters
        self.w = .25 # ratio between strength of inter-subpopulation synapses and intra-subpopulation ones
        self.J_inter = -160.
        self.J_intra = self.w * self.J_inter

        # Background activity
        self.bg_rate = 7500.

        # MPI communication
        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_procs = self.mpi_comm.Get_size()
        
    def build_local_network(self):
        # Create neurons and connect them to spike detectors
        for pop in ['left', 'right']:
            self.neurons[pop] = nest.Create('default_neuron', self.N[pop])
            self.spkdets[pop] = nest.Create('spike_detector')
            nest.Connect(self.neurons[pop], self.spkdets[pop])
        self.neurons['ALL'] = self.neurons['left'] + self.neurons['right']

        # Connect neurons to each other
        nest.CopyModel('default_synapse', 'striatum_intra_syn', {"weight": self.J_intra})
        nest.CopyModel('default_synapse', 'striatum_inter_syn', {"weight": self.J_inter})
        for origin, target in product(['left', 'right'], ['left', 'right']):
            syn_model = 'striatum_intra_syn' if origin == target else 'striatum_inter_syn'
            nest.Connect(self.neurons[origin], self.neurons[target], self.conn_params, syn_model)

        # Create and connect background activity
        background_activity = nest.Create('poisson_generator', params={"rate": self.bg_rate})
        nest.Connect(background_activity, self.neurons['ALL'], syn_spec='cortex_E_synapse')

        # initiate membrane potentials
        self.initiate_membrane_potentials_randomly()
    
    def count_decision_spikes(self):
        dec_spk = [nest.GetStatus(self.spkdets[pop], 'n_events')[0] for pop in ['left', 'right']]
        dec_spk = np.array(dec_spk, dtype='i')
        recvbuf = np.empty([self.mpi_procs, 2], dtype='i') if self.mpi_rank == 0 else None
        self.mpi_comm.Gather(dec_spk, recvbuf, root=0)
        if self.mpi_rank == 0:
            recvbuf = np.sum(recvbuf, axis=0)
            decision_spikes = {pop : recvbuf[it] for it, pop in enumerate(['left', 'right'])}
        else:
            decision_spikes = dict()
        decision_spikes = self.mpi_comm.bcast(decision_spikes, root=0)
        return decision_spikes