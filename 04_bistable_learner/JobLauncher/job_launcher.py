import numpy as np
import os, subprocess
from typing import Dict
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.util import load_logs
from bayes_opt.event import Events
from bayes_opt import UtilityFunction
#from Experiment import Experiment


class Job():
    def __init__(self, local_id:int, pars:Dict[str, float]):
        self.local_id = local_id
        self.pars = pars
        self.nemo_id = None
        self.status = None
    
    @property
    def name(self):
        return 'job' + str(self.local_id).rjust(3, '0')

    @property
    def args(self):
        args_str = ''
        for arg_key, arg_val in self.args.items():
            args_str += '--' + arg_key + ' ' + str(arg_val) + ' '
        return args_str





class JobLauncher():
    def __init__(self):
        self.launcher_dir = os.path.dirname(os.path.realpath(__file__))
        self.jobs_dir = os.path.join(self.launcher_dir, 'jobs')
        self.run_job_script_path = os.path.join(self.launcher_dir, 'run_job.py')
        self.jobs = list()
        self.max_running_jobs = 4
        self.optimizer = BayesianOptimization(
            f=None,
            pbounds={'x': (-2, 2), 'y': (-3, 3)},
            verbose=2,
            random_state=1)
        self.utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    @property
    def n_jobs(self):
        return len(self.jobs)

    @property
    def n_jobs_running(self):
        return np.sum([job.status=='Running' or job.status=='Idle' for job in self.jobs])

    def launch_new_job(self):
        new_job = Job(
            local_id=self.n_jobs,
            pars=self.optimizer.suggest(self.utility))
        script_path= self.create_batchjob(new_job)
        new_job.nemo_id = self.run_msub(script_path)
        self.jobs.append(new_job)

    def create_batchjob(self, job):
        job_script_path = os.path.join(self.jobs_dir, job.name+'.sh')
        job_fo = open(job_script_path, 'w')
        job_fo.write('#!/bin/bash\n')
        job_fo.write('\n')
        job_fo.write('#MSUB -l nodes=5:ppn=20\n')
        job_fo.write('#MSUB -l walltime=24:00:00\n')
        job_fo.write('#MSUB -l pmem=6gb\n')
        job_fo.write('#MSUB -m bea -M alessang@tf.uni-freiburg.de\n')
        job_fo.write('#MSUB -v MPIRUN_OPTIONS="--bind-to core --map-by core -report-bindings"\n')
        job_fo.write(f'#MSUB -N {job.name}\n')
        job_fo.write('\n')
        job_fo.write('module load mpi/openmpi/3.1-gnu-8.2\n')
        job_fo.write('module load system/modules/testing\n')
        job_fo.write('module load neuro/nest/2.16.0-python-3.7.0\n')
        job_fo.write('\n')
        job_fo.write(f'mpirun python {self.run_job_script_path} {job.args}\n')
        job_fo.close()
        return job_script_path
    
    def run_msub(self, script_path):
        result = subprocess.run(['msub', script_path], stdout=subprocess.PIPE)
        nemo_id = result.stdout.decode('utf8').strip()
        return nemo_id

    def update_job_states(self):
        for job in self.jobs:
            cmd = f'checkjob {job.nemo_id} | grep ^State'
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
            job.status = result.stdout.decode('utf8').strip().split()[1]

    def run_optimization()



        





    
    
    







# Let's start by definying our function, bounds, and instanciating an optimization object.
def black_box_function(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1






for _ in range(2):
    next_point = optimizer.suggest(utility)
    target = black_box_function(**next_point)
    optimizer.register(params=next_point, target=target)
    print(target, next_point)

print('max', optimizer.max)


for i in range(10):
    next_point_to_probe = optimizer.suggest(utility)
    print("Next point to probe is:", next_point_to_probe)
