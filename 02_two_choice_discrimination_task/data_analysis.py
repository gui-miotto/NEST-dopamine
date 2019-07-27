import pickle, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from experiment_methods import ExperimentMethods
from glob import glob

exp_dir = 'test2/'
figs_dir = methods_file_path = os.path.join(exp_dir, 'plots')
if not os.path.exists(figs_dir):
    os.mkdir(figs_dir)

methods_file_path = os.path.join(exp_dir, 'methods.data')
EM = pickle.load(open(methods_file_path, 'rb'))

scale_factors, actions = [], []
mean_S_to_pop_w = {'A' : [], 'B' : []}

#pdf = PdfPages(os.path.join(figs_dir, 'asdf.pdf'))

#loop through trials
for trial_dir in sorted(glob(exp_dir + 'trial-*/')):
    print('Reading', trial_dir)
    r0_file_path = os.path.join(trial_dir, 'rank-000.data')
    ER0 = pickle.load(open(r0_file_path, 'rb'))
    
    scale_factors.append(ER0.rescal_factor)
    A1_minus_A2 = ER0.decision_spikes['A1'] - ER0.decision_spikes['A2']
    sel_act = 'action A1' if A1_minus_A2 > 0 else 'action A2' if A1_minus_A2 < 0 else 'no action'
    actions.append(sel_act)
    cnn_matrix = np.zeros_like(ER0.cnn_matrix.toarray())
    events = {pop : [[], []] for pop in EM.sp_group['all_sps']}
    EE_fr_hist = [np.zeros_like(ER0.EE_fr_hist[0]), ER0.EE_fr_hist[1]]
    EE_w_hist = [np.zeros_like(ER0.EE_w_hist[0]), ER0.EE_w_hist[1]]
    S_to_pop_weight = {'A1' : [], 'A2' : []}

    #loop through the different MPI processes outputs
    for proc_file in os.listdir(trial_dir):
        ER_file_path = os.path.join(trial_dir, proc_file)
        ER = pickle.load(open(ER_file_path, 'rb'))

        # read events (for raster plots)
        for pop in EM.sp_group['all_sps']:
            events[pop][0] += list(ER.events[pop]['times'])
            events[pop][1] += list(ER.events[pop]['senders'])

        # read histogram of excitatory firing rates
        EE_fr_hist[0] += ER.EE_fr_hist[0]

        # read histogram of excitatory weights
        EE_w_hist[0] += ER.EE_w_hist[0]

        # read S to pop weights
        for pop in ['A1', 'A2']:
            S_to_pop_weight[pop] += ER.S_to_pop_weight[pop]
        
        # read connectivity matrix
        cnn_matrix += ER.cnn_matrix.toarray()

    # Create figure
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(15., 9.))
    plt.subplots_adjust(wspace = 0.25, hspace = 0.4, left=0.1, right=0.95, bottom=.07)
    trial_num_str = str(ER0.trial).rjust(3, '0')
    plt.suptitle(f'Trial {trial_num_str}\n{sel_act}', size=16., weight='normal')

    # decision period raster plot 
    plt.subplot(3,4,1)
    plt.title('Decision period raster plot')
    prev_max_snd, times, shifted_snds = 0, {}, {}
    for pop in ['S', 'A', 'B', 'exc', 'inh']:
        if len(events[pop][0]) == 0:
            continue
        snds = np.array(events[pop][1])
        times[pop] = np.array(events[pop][0]) - ER0.trial_begin
        shifted_snds[pop] = snds - np.min(snds) + prev_max_snd + 1
        plt.scatter(times[pop], shifted_snds[pop], marker='o', s=5., label=pop)
        prev_max_snd = np.max(shifted_snds[pop])
    t_min = -.1 * EM.eval_time_window
    t_max = 1.1 * EM.eval_time_window
    plt.xlim(t_min, t_max)
    plt.xlabel('time (ms)')

    # full trial raster plot
    plt.subplot(3,4,(2,3))
    plt.title('Full trial raster plot')
    prev_max_snd = 0
    for pop in times.keys():
        plt.scatter(times[pop] / 1000., shifted_snds[pop], marker='.', s=5., label=pop)
    plt.legend(loc='upper right')
    plt.xlabel('time (s)')

    # firing rates histogram
    plt.subplot(3, 4, 4)
    plt.title('Excitatory firing rate')
    plt.bar(EE_fr_hist[1][:-1], EE_fr_hist[0], width=.5, align='edge')
    plt.xlabel('firing rate (hz)')

    # E-E weights histogram
    plt.subplot(3, 4, 5)
    plt.title('E-E weights')
    plt.bar(EE_w_hist[1][:-1], EE_w_hist[0], width=.025, log=True, align='edge')
    plt.xlabel('weight (mV)')

    # Mean weight between S and A or B
    plt.subplot(3, 4, (6, 7))
    plt.title('Mean weight on direct connections between S and A or B')
    x_trials = np.arange(1, ER0.trial+1)
    for pop in ['A', 'B']:
        mean_S_to_pop_w[pop].append(np.mean(S_to_pop_weight[pop]))
        plt.plot(x_trials, mean_S_to_pop_w[pop], label='S to ' + pop)
    plt.legend(loc='upper left')
    plt.xlabel('trial')
    plt.ylabel('weight (mV)')
    #trial_xlim = (int(-.05*EM.n_trials), int(1.05*EM.n_trials))
    #plt.xlim(trial_xlim)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # S-pop weights histogram
    plt.subplot(3, 4, 8)
    plt.title('S-A and S-B weights')
    plt.hist([S_to_pop_weight['A'], S_to_pop_weight['B']], label=['S to A', 'S to B'], \
        bins=EE_w_hist[1], log=True)
    plt.ylim(1., EM.N['A'] * EM.C['S'] * 10)
    plt.xlabel('weight (mV)')
    plt.legend(loc='upper center')
    
    # synaptic scaling factors
    plt.subplot(3, 4, 9)
    plt.title('Synaptic scaling factor')
    plt.plot(scale_factors)
    if len(scale_factors) > 1:
        x = np.arange(len(scale_factors))
        reg_line = np.poly1d(np.polyfit(x, scale_factors, 1))
        plt.plot(x, reg_line(x), label='lin reg')
        plt.legend()
    plt.ylim(.8, 1.2)
    plt.xlabel('trial')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Probability of action selection plot
    plt.subplot(3, 4, (10, 11))
    plt.title('Probability of action selection')
    bin_size = 25
    plot_data = {'T' : [], 'action A' : [], 'action B' : [], 'no action' : []}
    for trial_n in range(0, len(actions), bin_size):
        sels = np.array(actions[trial_n : trial_n+bin_size])
        plot_data['T'].append(trial_n + bin_size)
        for res in ['action A', 'action B', 'no action']:
            ind = np.where(sels==res)[0]
            plot_data[res].append(len(ind) / len(sels))
    for res in ['action A', 'action B', 'no action']:
        plt.plot(plot_data['T'], plot_data[res], marker='.', label=res)
    plt.legend(loc='lower left')
    plt.xlabel('trial')
    #plt.xlim(trial_xlim)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))


    # connectivity matrix
    plt.subplot(3, 4, 12)
    plt.title('Connectivity matrix')
    plt.imshow(cnn_matrix, cmap=plt.get_cmap('hot'), origin='lower', \
        vmin= 0., vmax=EM.DA_pars['Wmax'])
    plt.xticks([50, 100, 150, 200], ('S    ', 'A    ', 'B    ', 'exc  '), ha='right')
    plt.yticks([50, 100, 150, 200], ('\nS', '\nA', '\nB', '\nexc'), va='top')
    plt.xlabel('post-synaptic')
    plt.ylabel('pre-synaptic')
    
    


    # Save figure
    fig_file = 'trial_' + trial_num_str + '.png'
    #pdf.savefig()
    plt.savefig(os.path.join(figs_dir, fig_file), transparent=False)
    plt.close(fig)

#pdf.close()