import os, subprocess, glob, pickle, time, shutil

import numpy as np
from bayes_opt import BayesianOptimization as BayesOptim
from bayes_opt import UtilityFunction
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events


class Job():
    def __init__(self, pars):
        self.pars = pars
        self.result = None


class LocalSide():
    def __init__(self):
        # Things that can be altered
        par_bounds = {
            'aplus': (.005, .5),
            'aminus': (0., 1.),
            'wmax': (1.5, 3.),
            'degree' : (.25, 2.),
            'memory' : (20, 50),
            }

        self.max_running_jobs = 40
        self.nemo_user = 'fr_ga52@login2.nemo.uni-freiburg.de'
        self.nemo_new_jobs_dir = '/home/fr/fr_fr/fr_ga52/code/new_jobs'
        nemo_finished_jobs_dir = '/home/fr/fr_fr/fr_ga52/code/finished_jobs/'

        # Things that can remain fixed
        my_dir = os.path.dirname(os.path.realpath(__file__))
        self.local_new_jobs_dir = os.path.join(my_dir, 'new_jobs/')
        if os.path.exists(self.local_new_jobs_dir):
            shutil.rmtree(self.local_new_jobs_dir, ignore_errors=False, onerror=None)
        os.mkdir(self.local_new_jobs_dir)

        local_finished_jobs_dir = os.path.join(my_dir, 'finished_jobs/')
        if os.path.exists(local_finished_jobs_dir):
            shutil.rmtree(local_finished_jobs_dir, ignore_errors=False, onerror=None)
        os.mkdir(local_finished_jobs_dir)
        
        self.download_cmd = f'rsync -az {self.nemo_user}:{nemo_finished_jobs_dir} {local_finished_jobs_dir}'
        self.res_files_pattern = os.path.join(local_finished_jobs_dir, 'job_*.res')

        # optimizer stuff
        self.optimizer = BayesOptim(f=None, pbounds=par_bounds, verbose=2, random_state=1)
        self.utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        logger = JSONLogger(path=os.path.join(my_dir, 'logger.json'))
        self.optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

        self.jobs = list()

    @property
    def jobs_running(self):
        return np.sum([job.result is None for job in self.jobs])

    def run_optimization(self, n_jobs):
        # submission/read phase
        for i in range(n_jobs):
            print(f'\nStarting job number {i+1}')
            next_job_pars = self.optimizer.suggest(self.utility)
            submitted = False
            while not submitted:
                results_read = self.read_results()
                print(f'Read {results_read} new results. Best job so far:')
                print(self.optimizer.max)
                if self.jobs_running < self.max_running_jobs:
                    self.submit_new_job(next_job_pars)
                    submitted = True
                    tsleep = 5
                    print(f'Submitted. Sleeping for {tsleep} seconds')
                else:
                    tsleep = 120
                    print(f'Queue is full. Sleeping for {tsleep} seconds')
                time.sleep(tsleep)
        # read only phase
        print('\nAll jobs submitted. Waiting for some of them to finish.\n')
        sleept = 120
        while self.jobs_running > 0:
            results_read = self.read_results()
            print(f'Read {results_read} new results. Best job so far:')
            print(self.optimizer.max)
            print(f'Checking again in {sleept} seconds\n')
            time.sleep(sleept)
        # Finished
        self.read_results()
        print('Optimization complete. Best job:')
        print(self.optimizer.max)


    def submit_new_job(self, pars):
        self.jobs.append(Job(pars))
        
        # Create file
        job_name = 'job_' + str(len(self.jobs)).rjust(4, '0') + '.par'
        local_pars_path = os.path.join(self.local_new_jobs_dir, job_name)
        pickle.dump(pars, open(local_pars_path, 'wb'))

        # Send file        
        nemo_pars_path = os.path.join(self.nemo_new_jobs_dir, job_name)
        upload_cmd = f'rsync -az {local_pars_path} {self.nemo_user}:{nemo_pars_path}'
        subprocess.run(upload_cmd, shell=True).check_returncode()


    def read_results(self):
        # sync results folder
        subprocess.run(self.download_cmd, shell=True).check_returncode()
        
        # check result files
        results_read = 0
        for res_file in glob.glob(self.res_files_pattern):
            # get job id
            fname = os.path.basename(res_file)
            job_id = int(fname.split('.')[0].split('_')[1])
            
            # skip to the next in case we already read this file
            if self.jobs[job_id-1].result is not None:
                continue

            # read result file
            job_fo = open(res_file, 'r')
            result = float(job_fo.readline())
            job_fo.close()

            # register result
            self.jobs[job_id-1].result = result
            self.optimizer.register(params=self.jobs[job_id-1].pars, target=result)

            results_read += 1
        
        return results_read
    
if __name__ == '__main__':
    LS = LocalSide()
    LS.run_optimization(n_jobs=400)


    
    
    
    