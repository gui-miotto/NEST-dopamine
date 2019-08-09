import nest
nest.SetKernelStatus({'local_num_threads' : 2})

rank = nest.Rank()
neurons = nest.Create('parrot_neuron', 10)
nA, nB = [neurons[7]], [neurons[8]]
nest.SetDefaults('stdp_dopamine_synapse', {'vt' : nest.Create('volume_transmitter')[0]})

#nest.Connect(nA, nB)
#nest.Connect(nA, nB, syn_spec='stdp_synapse')
nest.Connect(nA, nB, syn_spec='stdp_dopamine_synapse')


cnn = nest.GetConnections(nA, nB)
cnn_status = nest.GetStatus(cnn)

print('rank', rank, 'is local',  nest.GetStatus(nA, 'local'), nest.GetStatus(nB, 'local'))
print('rank', rank, 'status', cnn_status)


# BOTTOMLINE: synapse information is stored by the proecess of the postsynaptic neuron (at least on 
# the three tested models)



ks = nest.GetKernelStatus()
if nest.Rank() == 0:
    print('np', nest.NumProcesses())
    print('lp', ks['local_num_threads'])
    print('tp', ks['total_num_virtual_procs'])