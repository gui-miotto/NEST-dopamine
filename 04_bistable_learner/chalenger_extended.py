import sys, os, argparse
import numpy as np
from SpikingBGRL import Experiment

if __name__ == '__main__':
    # Parse command line input parameters
    parser = argparse.ArgumentParser(description='Run one experiment once')
    parser.add_argument('outdir', type=str, help='Output folder for result')
    args = parser.parse_args()
    
    # Build experiment
    #{"target": 184.0, "params": {"aminus": 0.1343804852164956, "aplus": 0.16437680257844103, "aversion": 0.6213721307075251, "wmax": 2.946638360980012},#}
    exp = Experiment()
    exp.brain.vta.DA_pars['A_plus'] = .16437680257844103 * exp.brain.vta.DA_pars['weight']
    exp.brain.vta.DA_pars['A_minus'] = .1343804852164956 * exp.brain.vta.DA_pars['weight']
    exp.brain.vta.DA_pars['Wmax'] = 2.946638360980012 * exp.brain.vta.DA_pars['weight']

    # Run normal conditioning
    success_history = exp.train_brain(n_trials=400, aversion=True, save_dir=args.outdir)
