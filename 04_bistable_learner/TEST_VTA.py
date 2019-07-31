import nest
from VTA import VTA
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

vta = VTA(dt, 20., 1.5)
vta.build_local_network()
b = nest.GetDefaults('corticostriatal_synapse', 'b') + 1. / 400.

neurons = nest.Create('parrot_neuron', 2)
nest.Connect(neurons[:1], neurons[1:], syn_spec='corticostriatal_synapse')
cnn = nest.GetConnections(neurons[:1], neurons[1:])

for _ in range(3):
    vta.program_drive(sim_time, 'rewarding', 3000.)
    n = []
    beg = nest.GetKernelStatus()['time']
    times = np.arange(beg + dt, beg+sim_time, dt)
    for _ in times:
        n.append(nest.GetStatus(cnn, 'n'))
        nest.Simulate(dt)
    plt.plot(times[:], np.array(n[:]) - b)
    plt.show()


