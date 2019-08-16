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
    
    # Tweeks go here:
    exp.brain.vta.DA_pars['A_minus'] = 0.100392926122981 * exp.brain.vta.DA_pars['weight']
    exp.brain.vta.DA_pars['A_plus'] = 0.150908614634723 * exp.brain.vta.DA_pars['weight']
    exp.brain.vta.degree = 1.00944414708664
    exp.brain.vta.memory = 43
    exp.brain.vta.DA_pars['Wmax'] = 2.93502652704248 * exp.brain.vta.DA_pars['weight']
    

    # Run normal conditioning
    success_history = exp.train_brain(n_trials=150, save_dir=args.outdir)
    # Run reversal learning
    success_history = exp.train_brain(n_trials=150, save_dir=args.outdir, rev_learn=True)
