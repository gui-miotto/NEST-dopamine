import sys, argparse
sys.path.insert(1, '.')  # hate to do that, but...
from SpikingBGRL import Experiment

if __name__ == '__main__':
    # Parse command line input parameters
    parser = argparse.ArgumentParser(description='Run one experiment once')
    parser.add_argument('jobname', type=str, help='Job name')
    parser.add_argument('outdir', type=str, help='Output folder for result')
    
    parser.add_argument('--aplus', default=.1, type=float)
    parser.add_argument('--aminus', default=.15, type=float)
    parser.add_argument('--wmax', default=60., type=float)
    parser.add_argument('--aversion', default=0., type=float)
    args = parser.parse_args()
    
    # Build experiment
    exp = Experiment()
    
    # Tweeks go here:
    exp.brain.vta.DA_pars['A_plus'] = args.aplus * exp.brain.vta.DA_pars['weight']
    exp.brain.vta.DA_pars['A_minus'] = args.aminus * exp.brain.vta.DA_pars['weight']
    exp.brain.vta.DA_pars['Wmax'] = args.wmax * exp.brain.vta.DA_pars['weight']
    av = True if args.aversion > .5 else False

    # Run normal conditioning
    success_history = exp.train_brain(n_trials=150, aversion=av, full_io=False)
    result = np.sum(success_history[-100:])
    # Run reversal learning
    success_history = exp.train_brain(n_trials=150, aversion=av, full_io=False, rev_learn=True)
    result += np.sum(success_history[-100:])

    # Write results to file
    res_path = os.path.join(args.outdir, args.jobname+'.results')
    res_fo = open(res_path, 'w')
    res_fo.write(result)
    res_fo.close()