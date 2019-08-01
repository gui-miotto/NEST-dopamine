import nest
import numpy as np
import BrainStructures as BS


class VTA(BS.BaseBrainStructure):
    def __init__(self, dt, J_E, syn_delay, **args):
        super().__init__(**args)
        
        self.dt = dt  # simulation timestep
        self.N['ALL'] = 1  # Number of neurons. No need to scale here

        # Dopamine modulation parameters
        A_plus_mult = .01  #TODO: clean up this mess
        Wmax_mult = 5.
        A_minus_mult = 1.5

        self.J = J_E
        self.tau_n = 200.
        A_plus = A_plus_mult * self.J
        A_minus = A_minus_mult * A_plus
        self.DA_pars = {
            'weight' : self.J,  # Default 1.
            'delay': syn_delay, # Default 1.; Synaptic delay
            'tau_n' : self.tau_n, # Default 200.; Time constant of dopaminergic trace in ms
            'b' : 1. / self.dt,  # Default 0.; Dopaminergic baseline concentration
            'n' : 1. / self.dt + 2.5 / self.tau_n, # Default 0.; Initial dopamine concentration
            'A_plus' : A_plus,  # Default 1.; Amplitude of weight change for facilitation
            'A_minus' : A_minus,  # Default 1.5; Amplitude of weight change for depression
            'Wmax' : self.J * Wmax_mult, # Maximal synaptic weight  
            #'tau_c' : 1000., # Default 1000.,  # Time constant of eligibility trace in ms
            #'tau_plus' : 20.0, #  Default 20.; STDP time constant for facilitation in ms
            #'Wmin' : 0., # Default 0. # Minimal synaptic weight
            #'vt' : volt_DA[0], # Volume transmitter will be assigned later on
            }
        self.salience = 10  # integer greater than 0. Number of spikes added or subtracted from the 
                            # baseline in the face of rewarding or aversive events (respectively)
        
    def build_local_network(self):
        # Create nodes
        self.drive = nest.Create('spike_generator')  # Spike generator to drive VTA activity
        self.neurons['ALL'] = nest.Create('parrot_neuron', self.N['ALL']) #A middleman parrot neuron
        self.vt = nest.Create('volume_transmitter')  # volume transmitter

        # Connect nodes in a chain 
        # We can't connect the spike generator directly to the volume transmitter due to a NEST bug)
        nest.Connect(self.drive, self.neurons['ALL'], syn_spec={'delay' : self.dt})
        nest.Connect(self.neurons['ALL'], self.vt, syn_spec={'delay' : self.dt})
        self.DA_pars['vt'] = self.vt[0]

        # Create synapse that will be used by cortico-striatal neurons
        nest.CopyModel('stdp_dopamine_synapse', 'corticostriatal_synapse', self.DA_pars)


    def set_drive(self, length, drive_type='baseline', delay=None):
        drive_types = ['baseline', 'rewarding', 'aversive']
        if drive_type not in drive_types:
            raise ValueError('drive_type must one of those:', drive_types)

        begin = nest.GetKernelStatus()['time'] + self.dt
        end = begin + length - .5 * self.dt  # subtract .5 dt for numerical stability

        if drive_type == 'baseline':
            spike_times = np.arange(begin, end, self.dt)
        else:
            if delay is None:
                raise ValueError('It is necessary to specify the delay for reward or aversion')
            delivery = begin + delay
            if drive_type == 'rewarding':  # i.e the baseline with some extra spikes
                spike_times = np.sort(np.concatenate((
                    np.arange(begin, end, self.dt), 
                    np.arange(delivery, delivery + (self.salience - .5) * self.dt, self.dt)
                )))
            elif drive_type == 'aversive':  # i.e. the baseline with some missing spikes
                spike_times = np.concatenate((
                    np.arange(begin, delivery - .5 * self.dt, self.dt),  
                    np.arange(delivery + (self.dt * self.salience), end, self.dt)
                ))
        
        spike_times = np.round(spike_times, decimals=1)
        nest.SetStatus(self.drive, params={'spike_times' : spike_times})
