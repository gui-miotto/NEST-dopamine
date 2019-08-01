import nest
import BrainStructures as BS
import numpy as np
import matplotlib.pyplot as plt


# Kernel parameters
dt = .1
verbosity = 20
kernel_pars = {
    'print_time' : False,
    'resolution' : dt,
    'local_num_threads' : 4,
    'grng_seed' : 0,
    }

# Configure kernel
nest.ResetKernel()
nest.set_verbosity(verbosity)
nest.SetKernelStatus(kernel_pars)


sim_time=5000.

vta = BS.VTA(dt, 20., 1.5)
vta.build_local_network()
b = nest.GetDefaults('corticostriatal_synapse', 'b') + 1. / 400.

neuron_spk = nest.Create('spike_generator', params={'spike_times' : [500., 1000., 2000.]})
neurons = nest.Create('parrot_neuron', 2)
nest.Connect(neuron_spk, neurons[:1])
nest.Connect(neurons[:1], neurons[1:], syn_spec='corticostriatal_synapse')
cnn = nest.GetConnections(neurons[:1], neurons[1:])

for _ in range(3):
    vta.set_drive(sim_time, 'baseline', 3000.)
    n, w = [], []
    beg = nest.GetKernelStatus()['time']
    times = np.arange(beg + dt, beg+sim_time, dt)
    for _ in times:
        n.append(nest.GetStatus(cnn, 'n'))
        w.append(nest.GetStatus(cnn, 'weight'))
        nest.Simulate(dt)
    fig, axes = plt.subplots(2,1,True)
    
    axes[0].plot(times[:], np.array(n[:]) - b)
    axes[1].plot(times[:], w)
    plt.show()
    
    


