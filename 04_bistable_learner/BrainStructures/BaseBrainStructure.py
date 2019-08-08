import nest
import numpy as np

class BaseBrainStructure(object):
    # static numpy random number generators
    _py_rngs = None
    @property
    def py_rngs(self):
        return type(self)._py_rngs
    

    def __init__(self, scale=1):
        self.scale = scale  #TODO: make it static?
        self.N = dict()  # Number of neurons in each subpopulation
        self.neurons = dict()  # Neuron handles for each subpopulation
        self.spkdets = dict()  # Spike detectors
        self.events_ = dict()  # Events registered by the spike detectors
        self.plastic_synapses = dict()  # A dictionary of plastic synapses - one entry per neuron
        self.plastic_weight_setpoint = None # Total plastic weight per neuron - homeostatic setpoint


    def build_local_network(self):
        raise NotImplementedError('All brain scructures must implement build_local_network()')


    def initiate_membrane_potentials_randomly(self, v_min=None, v_max=None, pops=['ALL']):
        if v_min == None and v_max == None:
            neu_pars = nest.GetDefaults('default_neuron')
            v_min, v_max = neu_pars['V_reset'], neu_pars['V_th']

        for pop in pops:
            node_info = nest.GetStatus(self.neurons[pop])
            local_nodes = [(ni['global_id'], ni['vp']) for ni in node_info if ni['local']]
            for gid, proc in local_nodes:
                nest.SetStatus([gid], {'V_m': self.py_rngs[proc].uniform(v_min, v_max)})
    

    def read_reset_spike_detectors(self):
        for pop, spkdet in self.spkdets.items():
            self.events_[pop] = nest.GetStatus(spkdet, 'events')[0]
            nest.SetStatus(spkdet, {'n_events' : 0 })


    def store_plastic_synapses(self, neurons, inputs, syn_model):
        # TODO: check locals here?
        self.plastic_synapses = {n : nest.GetConnections(n, inputs, synapse_model=syn_model)}
    

    def homeostatic_scaling(self)
        for n, syns in self.plastic_synapses.items():
            current_weights = np.array(nest.GetStatus(syns, 'weight'))
            scaling_factor = self.plastic_weight_setpoint / np.sum(current_weights)
            new_weights = scaling_factor * current_weights
            nest.SetStatus(syns, params='weight', val=new_weights)



