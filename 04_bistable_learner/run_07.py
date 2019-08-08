import sys, argparse
from Experiment import Experiment

if __name__ == '__main__':
    # Parse command line input parameters
    parser = argparse.ArgumentParser(description='Run one experiment once')
    parser.add_argument('outdir', type=str, help='Output folder for results')
    parser.add_argument('--n_trials', default=400, type=int,
        help='Number of trials in the experiment (default 400)')
    args = parser.parse_args()
    
    # Build experiment
    exp = Experiment()
    
    # Tweeks go here:
    aversion=True
    exp.brain.striatum.w = .25
    exp.brain.striatum.J_intra = exp.brain.striatum.w * exp.brain.striatum.J_inter
    exp.brain.vta.DA_pars['A_minus'] = 0.


    # Run experiment
    exp.train_brain(n_trials=args.n_trials, aversion=aversion, save_dir=args.outdir)


    
