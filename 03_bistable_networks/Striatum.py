import nest
from itertools import product
from BaseBrainStructure import BaseBrainStructure


class Striatum(BaseBrainStructure):
    def __init__(self, **args):
        super().__init__(**args)
    
        # Number of neurons
        n = int(1250 * self.scale)  # neurons per subpopulation
        self.N['A'] = self.N['B'] = n
        self.N['ALL'] = self.N['A'] + self.N['B']

        # Connectivity
        epsilon = .1  # connection probability
        self.conn_params = {'rule': 'fixed_indegree', 'indegree': int(epsilon * n)} 

        # synapse parameters
        self.w = .2 # ratio between strength of inter-subpopulation synapses and intra-subpopulation ones
        self.J_inter = -160.
        self.J_intra = self.w * self.J_inter

        # Background activity
        self.bg_rate = 7500.
        
    def build_local_network(self):
        # Create neurons and connect them to spike detectors
        for pop in ['A', 'B']:
            self.neurons[pop] = nest.Create('default_neuron', self.N[pop])
            self.spkdets[pop] = nest.Create('spike_detector')
            nest.Connect(self.neurons[pop], self.spkdets[pop])
        self.neurons['ALL'] = self.neurons['A'] + self.neurons['B']

        # Connect neurons to each other
        nest.CopyModel('default_synapse', 'striatum_intra_syn', {"weight": self.J_intra})
        nest.CopyModel('default_synapse', 'striatum_inter_syn', {"weight": self.J_inter})
        for origin, target in product(['A', 'B'], ['A', 'B']):
            syn_model = 'striatum_intra_syn' if origin == target else 'striatum_inter_syn'
            nest.Connect(self.neurons[origin], self.neurons[target], self.conn_params, syn_model)

        # Create and connect background activity
        background_activity = nest.Create('poisson_generator', params={"rate": self.bg_rate})
        nest.Connect(background_activity, self.neurons['ALL'], syn_spec='cortex_E_synapse')

        # initiate membrane potentials
        self.initiate_membrane_potentials_randomly()
