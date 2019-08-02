import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import DataIO as DIO
import numpy as np

def get_raster_data(events, gids=None, shift_senders=False, shift_times=False, tmin=None, tmax=None):
    senders = events['senders']
    times = events['times']
    if gids is not None:
        matches = np.isin(senders, gids)
        senders = senders[matches]
        times = times[matches]
    if tmin is not None:
        matches = np.where(times >= tmin)
        senders = senders[matches]
        times = times[matches]
    if tmax is not None:
        matches = np.where(times <= tmax)
        senders = senders[matches]
        times = times[matches]
    senders = senders - np.min(senders) if shift_senders else senders
    times = times - np.min(times) if shift_times else times
    return senders, times

def build_trial_plots(figs_dir, data):
    for trial in range(data.num_of_trials):
        print('Plotting trial', trial)
        
        # Create figure
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(15., 9.))
        plt.subplots_adjust(wspace = 0.25, hspace = 0.4, left=0.1, right=0.95, bottom=.07)
        trial_num_str = str(trial + 1).rjust(3, '0')
        lminusr = data.lminusr[trial]
        cue = 'high tone' if data.cue[trial]=='high' else 'low tone'
        sel_act = 'went left' if lminusr > 0 else 'went right' if lminusr < 0 else 'no action'
        outcome = 'success' if data.success[trial] else 'fail'
        suptitle = f'Trial {trial_num_str}\n{cue} + {sel_act} = {outcome}'
        plt.suptitle(suptitle, size=16., weight='normal')
        
        # Format raster plot data
        raster_data, nsamp = dict(), 50 
        for i, (pop, gids) in enumerate(data.neurons.items()):
            senders, times = get_raster_data(
                data.events[trial][pop], 
                gids=gids[:nsamp], 
                shift_senders=True, 
                shift_times=True)
            raster_data[pop] = {'senders' : senders + i * nsamp, 'times' : times}
        
        # decision period raster plot
        plt.subplot(3,4,1)
        plt.title('Decision period raster plot')
        for pop, events in raster_data.items():
            plt.scatter(events['times'], events['senders'], marker='o', s=5., label=pop)
        t_min = -.1 * data.eval_time_window
        t_max = 1.1 * data.eval_time_window
        plt.xlim(t_min, t_max)
        plt.xlabel('time (ms)')

        # full trial raster plot
        plt.subplot(3,4,(2,3))
        plt.title('Full trial raster plot')
        for pop, events in raster_data.items():
            plt.scatter(events['times'] / 1000., events['senders'], marker='.', s=5., label=pop)
        plt.legend(loc='upper right')
        plt.xlabel('time (s)')
        
        # Save figure
        fig_file = 'trial_' + trial_num_str + '.png'
        #pdf.savefig()
        plt.savefig(os.path.join(figs_dir, fig_file), transparent=False)
        plt.close(fig)

def build_experiment_plot(figs_dir, data):
    print('Plotting experiment overview')

    # Create figure
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(15., 9.))
    plt.subplots_adjust(wspace = 0.25, hspace = 0.4, left=0.1, right=0.95, bottom=.07)
    plt.suptitle('Experiment overview', size=16., weight='normal')
    trials = range(1, data.num_of_trials+1)

    # Difference on decision spikes counts
    plt.subplot(3,4,(1, 2))
    plt.title('Decision spikes difference')
    plt.plot(trials, np.abs(data.lminusr))
    plt.xlabel('trials')

    # Synaptic scaling factors
    plt.subplot(3,4,(5, 6))
    plt.title('Synaptic scaling factor')
    plt.plot(trials, data.syn_rescal_factor)
    reg_line = np.poly1d(np.polyfit(trials, data.syn_rescal_factor, 1))
    plt.plot(trials, reg_line(trials), label='lin reg')
    plt.legend()
    plt.ylim(.8, 1.2)
    plt.xlabel('trials')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))


    # Mean weight between stimulus and actions
    def pop_to_pop_weight_plot(target, position):
        plt.subplot(3, 4, position)
        plt.title(f'Mean synaptic weight arriving at the {target} striatal subnetwork')
        for source in ['low', 'high', 'E_rec']:
            weights_mean = [wm.loc[source, target] for wm in data.weights_mean]
            plt.plot(trials, weights_mean, label=source)
        plt.xlabel('trials')
        plt.legend()
    pop_to_pop_weight_plot('left', (3,4))
    pop_to_pop_weight_plot('right', (7,8))

    # Probability of action selection
    plt.subplot(3, 4, (11, 12))
    plt.title('Probability of correct action selection')
    prob_sucess, bin_size = [], 25
    trials_coarse = range(0, data.num_of_trials, bin_size)
    for begin in trials_coarse:
        cut = data.success[begin:begin+bin_size]
        prob_sucess.append(np.sum(cut) / len(cut))
    plt.plot(trials_coarse, prob_sucess)
    plt.xlabel('trial')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))


    # Save figure
    fig_file = 'experiment_overview.png'
    #pdf.savefig()
    plt.savefig(os.path.join(figs_dir, fig_file), transparent=False)
    plt.close(fig)



data_dir = 'temp2'
figs_dir = os.path.join(data_dir, 'plots')
if not os.path.exists(figs_dir):
    os.mkdir(figs_dir)
data = DIO.Reader().read(data_dir)

build_trial_plots(figs_dir, data)
build_experiment_plot(figs_dir, data)






