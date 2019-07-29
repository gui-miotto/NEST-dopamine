import math, nest
from BaseBrainStructure import BaseBrainStructure


class Cortex(BaseBrainStructure):
    
    def __init__(self, neu_params, **args):
        super().__init__(**args)

        # Number of neurons
        self.N['I'] = int(2500 * self.scale)  # number of inhibitory neurons
        self.N['E'] = 4 * self.N['I']  # number of excitatory neurons
        self.N['E_rec'] = self.N['I_rec'] = 500  # number of neurons to record from

        # Connectivity
        epsilon = 0.1  # connection probability
        self.C = {pop : int(epsilon * n) for pop, n in self.N.items()} # num synapses per neuron

        # synapse parameters
        g = 8.  # ratio inhibitory weight/excitatory weight
        self.J = {'E' : 20.} # amplitude of excitatory postsynaptic current
        self.J['I'] = -g * self.J['E']  # amplitude of inhibitory postsynaptic current

        # Background firing rate
        eta = .9  # external rate relative to threshold rate
        nu_th = neu_params['V_th'] * neu_params['C_m']
        nu_th /= self.J['E'] * self.C['E'] * math.e * neu_params['tau_m'] * neu_params['tau_syn_ex']
        nu_ex = eta * nu_th
        self.bg_rate = 1000.0 * nu_ex * self.C['E']

    def build_local_network(self):
        # Create neurons and connect them to spike detectors
        for pop in ['E', 'I']:
            sd_id = pop + '_rec'
            self.neurons[pop] = nest.Create('default_neuron', self.N[pop])
            self.spkdets[sd_id] = nest.Create('spike_detector')
            nest.Connect(self.neurons[pop][:self.N[sd_id]], self.spkdets[sd_id])
        self.neurons['ALL'] = self.neurons['E'] + self.neurons['I']

        # Connect neurons with each other
        for pop in ['E', 'I']:
            syn_model_name = f'cortex_{pop}_synapse'
            nest.CopyModel('default_synapse', syn_model_name, {"weight": self.J[pop]})
            conn_params = {'rule': 'fixed_indegree', 'indegree': self.C[pop]}
            nest.Connect(self.neurons[pop], self.neurons['ALL'], conn_params, syn_model_name)

        # Create and connect background activity
        background_activity = nest.Create('poisson_generator', params={"rate": self.bg_rate})
        nest.Connect(background_activity, self.neurons['ALL'], syn_spec='cortex_E_synapse')

        # initiate membrane potentials
        self.initiate_membrane_potentials_randomly()
