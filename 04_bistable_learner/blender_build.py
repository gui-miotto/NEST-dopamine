import multiprocessing, pickle, itertools, os
import nest
import numpy as np
from SpikingBGRL.DataIO.Reader import Reader
from blender_neuron import Neuron


class BlenderVisualization():
    def __init__(self):
        self.neurons = dict()
        self.anim_tstep = 100.  # animation time stepe in miliseconds


    def get_ids_and_events(self, pickle_path):
        nest.SetKernelStatus({
            'print_time' : False,
            'resolution' : .1,
            'local_num_threads' : multiprocessing.cpu_count(),
            #'grng_seed' : master_seed,
            })

        orig_net = Reader().read('../../results/blendernet')

        orig_ids = orig_net.neurons['E'] + orig_net.neurons['I'] + \
            orig_net.neurons['left'] + orig_net.neurons['right']

        nest.SetDefaults('iaf_psc_alpha', {
            "C_m": 250.,  # capacitance of membrane in in pF
            "tau_m": 20.,  # time constant of membrane potential in ms
            "tau_syn_ex": .5,
            "tau_syn_in": .5,
            "E_L": 0.0,
            "V_reset": 0.0,
            "V_m": 0.0,
            "V_th": 20.,  # membrane threshold potential in mV
            })

        clone_ids = nest.Create('iaf_psc_alpha', len(orig_ids))
        for gid in clone_ids:
            nest.SetStatus([gid], {'V_m': np.random.uniform(0., 20.)})

        spk_det = nest.Create('spike_detector')
        nest.Connect(clone_ids, spk_det)

        orig_to_clone = {orig_id : clone_ids[n] for n, orig_id in enumerate(orig_ids)}
        self.ids = {
            'E' : [orig_to_clone[oid] for oid in orig_net.neurons['E']],
            'I' : [orig_to_clone[oid] for oid in orig_net.neurons['I']],
            'low' : [orig_to_clone[oid] for oid in orig_net.neurons['low']],
            'high' : [orig_to_clone[oid] for oid in orig_net.neurons['high']],
            'left' : [orig_to_clone[oid] for oid in orig_net.neurons['left']],
            'right' : [orig_to_clone[oid] for oid in orig_net.neurons['right']]}
        self.ids['str'] = self.ids['left'] + self.ids['right']
        self.ids['ctx'] = self.ids['E'] + self.ids['I']
        self.ids['E_no_S'] = list(set(self.ids['E']) - set(self.ids['low']) - set(self.ids['high']))
        pickle.dump(self.ids, open(os.path.join(pickle_path, 'ids.data'), 'wb'))

        print('Connecting nodes')
        # neurons to each other
        for cnn in orig_net.snapshot:
            source = orig_to_clone[cnn['source']]
            target = orig_to_clone[cnn['target']]
            syn_spec = {'weight': cnn['weight'], 'delay' : cnn['delay']}
            nest.Connect([source], [target], {'rule': 'one_to_one'}, syn_spec)
        # noise generator for cortex
        nest.Connect(
            nest.Create('poisson_generator', params={"rate": 8093.347705771733}),  # TODO: read this frequency from reader
            self.ids['ctx'],
            {'rule': 'all_to_all'},
            {'weight': 20., 'delay' : 1.5})  
        # noise generator for striatum
        nest.Connect(
            nest.Create('poisson_generator', params={"rate": 7950.}),  # TODO: read this frequency from reader
            self.ids['str'],
            {'rule': 'all_to_all'},
            {'weight': 20., 'delay' : 1.5})


        print('Simulating')
        nest.Simulate(10000.)

        self.events = nest.GetStatus(spk_det, 'events')[0]
        pickle.dump(self.events, open(os.path.join(pickle_path, 'events.data'), 'wb'))

    def load_ids_and_events(self, pickle_path):
        self.ids = pickle.load(open(os.path.join(pickle_path, 'ids.data'), 'rb'))
        self.events = pickle.load(open(os.path.join(pickle_path, 'events.data'), 'rb'))


    def create_neurons(self):
        #group 1 - random cortex neurons
        ids = self.ids['E_no_S'][:200]
        x0 = -9.5 * Neuron.c_to_c_dist
        y0 = -4.5 * Neuron.c_to_c_dist
        z0 = 0.
        self.create_box_of_neurons(x0, y0, z0, 4, 10, 5, ids)

        #group 2 - cortex neurons of group low
        ids = self.ids['low']
        x0 = -5.5 * Neuron.c_to_c_dist
        y0 = -4.5 * Neuron.c_to_c_dist
        z0 = 0.
        color = (0.151736, 0.0997155, 1, 1)
        self.create_box_of_neurons(x0, y0, z0, 2, 10, 5, ids, color=color)

        #group 3 - more random cortex neurons
        ids = self.ids['E_no_S'][200:600]
        x0 = -3.5 * Neuron.c_to_c_dist
        y0 = -4.5 * Neuron.c_to_c_dist
        z0 = 0.
        self.create_box_of_neurons(x0, y0, z0, 8, 10, 5, ids)

        #group 4 - cortex neurons of group high
        ids = self.ids['high']
        x0 = 4.5 * Neuron.c_to_c_dist
        y0 = -4.5 * Neuron.c_to_c_dist
        z0 = 0.
        color = (0.55, 1., .23, 1)
        self.create_box_of_neurons(x0, y0, z0, 2, 10, 5, ids, color=color)

        #group 5 - last bit of random cortex neurons
        ids = self.ids['E_no_S'][-200:]
        x0 = 6.5 * Neuron.c_to_c_dist
        y0 = -4.5 * Neuron.c_to_c_dist
        z0 = 0.
        self.create_box_of_neurons(x0, y0, z0, 4, 10, 5, ids)

        #group 6 - left striatum
        ids = self.ids['left']
        x0 = -3.5 * Neuron.c_to_c_dist
        y0 = -2. * Neuron.c_to_c_dist
        z0 = 8. * Neuron.c_to_c_dist
        color = (0.151736, 0.0997155, 1, 1)
        self.create_box_of_neurons(x0, y0, z0, 4, 5, 5, ids, color=color)

        #group 7 - right striatum
        ids = self.ids['right']
        x0 = 3.5 * Neuron.c_to_c_dist
        y0 = -2. * Neuron.c_to_c_dist
        z0 = 8. * Neuron.c_to_c_dist
        color = (0.55, 1., .23, 1)
        self.create_box_of_neurons(x0, y0, z0, 4, 5, 5, ids, color=color)


    def create_box_of_neurons(self, x0, y0, z0, I, J, K, ids, color=Neuron.default_color):
        assert I * J * K == len(ids)
        for n, (i, j, k) in enumerate(itertools.product(range(I), range(J), range(K))):
            x = x0 + i * Neuron.c_to_c_dist
            y = y0 + j * Neuron.c_to_c_dist
            z = z0 + k * Neuron.c_to_c_dist
            self.neurons[ids[n]] = Neuron(position=(x, y, z), color=color)


    def create_animation(self, pickle_path):
        animation = list()
        end_frame = self.anim_tstep
        for time, n_id in zip(self.events['times'], self.events['senders']):
            # check if its time to end this frame
            if time > end_frame:
                # apply strength decay
                for n_key in self.neurons.keys():
                    self.neurons[n_key].strength *= Neuron.strength_decay
                animation.append(self.neurons)
                end_frame += self.anim_tstep
            # for now lets ignore inhibitory neurons from the cortex
            if n_id in self.ids['I']:
                continue
            # if the neuron spiked we increase its strength
            self.neurons[n_id].strength += Neuron.strength_increment
        
        # Save
        pickle.dump(animation, open(os.path.join(pickle_path, 'animation.data'), 'wb'))






if __name__ == "__main__":
    files_path = '../../results/blendernet'
    BV = BlenderVisualization()
    #BV.get_ids_and_events(files_path)
    BV.load_ids_and_events(files_path)
    BV.create_neurons()
    BV.create_animation(files_path)



    pass
    #BV.load_net()
