from network_aversion_mpi import Network

#net = Network(master_seed=400, A_plus_mult=2., A_minus_mult=1.5, Wmax_mult=4.)
net = Network(A_plus_mult=1., A_minus_mult=2., Wmax_mult=3.)

net.build_network()
for n in range(30):
    sf = net.simulate_rest_state(1000.)
    if net.rank == 0:
        print(sf)

#accuracy = net.simulate(n_trials=300, syn_scaling=True, aversion=False, data_dir='test2')