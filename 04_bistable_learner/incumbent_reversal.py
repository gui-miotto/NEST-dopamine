import sys, os, argparse
import numpy as np
from SpikingBGRL import Experiment

if __name__ == '__main__':
    # Parse command line input parameters
    parser = argparse.ArgumentParser(description='Run one experiment once')
    parser.add_argument('outdir', type=str, help='Output folder for result')
    args = parser.parse_args()
    
    # Build experiment
    exp = Experiment()

    # Run normal conditioning
    success_history = exp.train_brain(n_trials=150, aversion=False, save_dir=args.outdir)
    result = np.sum(success_history[-100:])
    # Run reversal learning
    success_history = exp.train_brain(n_trials=150, aversion=False, rev_learn=True, save_dir=args.outdir)
    result += np.sum(success_history[-100:])
