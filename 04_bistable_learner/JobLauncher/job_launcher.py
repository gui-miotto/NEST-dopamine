import numpy as np
import os, subprocess, time
from typing import Dict
from glob import glob
from bayes_opt import BayesianOptimization as BayesOptim
from bayes_opt import UtilityFunction
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events


class Job():
    def __init__(self, local_id:int, pars:Dict[str, float]):
        self.local_id = local_id
        self.pars = pars
        self.nemo_id = None
        self.result = None
    
    @property
    def name(self):
        return 'job_' + str(self.local_id).rjust(3, '0')

    @property
    def args(self):
        args_str = ''
        for arg_key, arg_val in self.pars.items():
            args_str += '--' + arg_key + ' ' + str(arg_val) + ' '
        return args_str


class JobLauncher():
    def __init__(self):
        # configurable stuff
        self.user = 'fr_ga52'
        self.max_jobs = 3
        self.max_running_jobs = 2
        self.par_bounds = {
            'aplus': (.005, .5), 
            'aminus': (0., 1.),
            'aversion' : (0., 1.),
            'wmax': (1.5, 3.),
            }
        # stuff that can remain fixed
        self.launcher_dir = os.path.dirname(os.path.realpath(__file__))
        self.jobs_dir = os.path.join(self.launcher_dir, 'jobs')
        self.run_job_script_path = os.path.join(self.launcher_dir, 'run_job.py')
        self.optimizer = BayesOptim(f=None, pbounds=self.par_bounds, verbose=2, random_state=1)
        self.utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        logger = JSONLogger(path=os.path.join(self.launcher_dir, 'logger.json'))
        self.optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
        self.jobs = list()

    @property
    def n_jobs(self):
        return len(self.jobs)

    @property
    def n_jobs_running(self):
        cmd = f'showq -u {self.user} | grep ^Total.job'
        cmd_out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        jobs_running = cmd_out.stdout.decode('utf8').strip().split()[-1]
        return int(jobs_running)

    def run_optimization(self):
        for i in range(self.max_jobs):
            print(f'Starting job number {i}')
            submited = False
            while not submited:
                self.read_results()
                if self.n_jobs_running < self.max_running_jobs:
                    self.launch_new_job()
                    submited = True
                    print('Submited!')
                    sleept = 10
                else:
                    print(f'Too many jobs already running.')
                    sleept = 120
                print(f'Going to sleep for {sleept} seconds')
                time.sleep(sleept)
            print('Best job so far', self.optimizer.max)
            
    def read_results(self):
        res_files_pattern = os.path.join(self.jobs_dir, 'job_*.result')
        for res_file in glob(res_files_pattern):
            # read result file
            job_fo = open(res_file, 'r')
            result = float(job_fo.readline())
            job_fo.close
            #os.remove(res_file)
            # update job list
            job_id = int(res_file.split('.')[0].split('_')[1])
            self.jobs[job_id].result = result
            # register result in the optimizer
            self.optimizer.register(params=self.jobs[job_id].params, target=result)

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
        job_fo.write('#MSUB -l nodes=1:ppn=20\n')
        job_fo.write('#MSUB -l walltime=12:00:00\n')
        job_fo.write('#MSUB -l pmem=6gb\n')
        job_fo.write('#MSUB -m bea -M alessang@tf.uni-freiburg.de\n')
        job_fo.write('#MSUB -v MPIRUN_OPTIONS="--bind-to core --map-by core -report-bindings"\n')
        job_fo.write(f'#MSUB -N {job.name}\n')
        job_fo.write('\n')
        job_fo.write('module load mpi/openmpi/3.1-gnu-8.2\n')
        job_fo.write('module load system/modules/testing\n')
        job_fo.write('module load neuro/nest/2.16.0-python-3.7.0\n')
        job_fo.write('\n')
        job_fo.write(f'mpirun python {self.run_job_script_path} {job.name} {self.jobs_dir} {job.args}\n')
        job_fo.close()
        return job_script_path
    
    def run_msub(self, script_path):
        msub_out = subprocess.run(['msub', script_path], stdout=subprocess.PIPE)
        nemo_id = msub_out.stdout.decode('utf8').strip()
        return nemo_id

if __name__ == '__main__':
    jlauncher = JobLauncher()
    jlauncher.run_optimization()
