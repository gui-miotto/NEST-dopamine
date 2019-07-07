from network_aversion import Network

net = Network(master_seed=400, A_plus_mult=2., A_minus_mult=1.5, Wmax_mult=5.)
accuracy = net.simulate(n_trials=300, plot=True, save_plot_dir='test', syn_scaling=True, aversion=True)