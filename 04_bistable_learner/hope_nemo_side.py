import glob, pickle, os, subprocess, time, shutil

class NemoSide():
    def __init__(self):
        my_dir = os.path.dirname(os.path.realpath(__file__))
        par_files_dir = os.path.join(my_dir, 'new_jobs')
        self.par_files_pattern = os.path.join(par_files_dir, 'job_*.par')
        self.res_files_dir = os.path.join(my_dir, 'finished_jobs')
        self.run_job_script_path = os.path.join(my_dir, 'hope_run_job.py')

        if os.path.exists(par_files_dir):
            shutil.rmtree(par_files_dir, ignore_errors=False, onerror=None)
        os.mkdir(par_files_dir)
        
        if os.path.exists(self.res_files_dir):
            shutil.rmtree(self.res_files_dir, ignore_errors=False, onerror=None)
        os.mkdir(self.res_files_dir)

    def create_batchjob(self, job_name, job_args):
        job_script_path = os.path.join(self.res_files_dir, job_name+'.sh')
        job_fo = open(job_script_path, 'w')
        job_fo.write('#!/bin/bash\n')
        job_fo.write('\n')
        job_fo.write('#MSUB -l nodes=1:ppn=20\n')
        job_fo.write('#MSUB -l walltime=3:00:00\n')
        job_fo.write('#MSUB -l pmem=6gb\n')
        #job_fo.write('#MSUB -m bea -M alessang@tf.uni-freiburg.de\n')
        job_fo.write('#MSUB -v MPIRUN_OPTIONS="--bind-to core --map-by core -report-bindings"\n')
        job_fo.write(f'#MSUB -N {job_name}\n')
        job_fo.write('\n')
        job_fo.write('module load mpi/openmpi/3.1-gnu-8.2\n')
        job_fo.write('module load system/modules/testing\n')
        job_fo.write('module load neuro/nest/2.16.0-python-3.7.0\n')
        job_fo.write('\n')
        job_fo.write(f'mpirun python {self.run_job_script_path} {job_name} {self.res_files_dir} {job_args}\n')
        job_fo.close()
        return job_script_path

    def get_job_args(self, pars):
        args_str = ''
        for arg_key, arg_val in pars.items():
            args_str += '--' + arg_key + ' ' + str(arg_val) + ' '
        return args_str

    def run(self):
        while True:
            submitted = False
            for par_file in glob.glob(self.par_files_pattern):
                print('Submiting new job...')
                # get job name
                fname = os.path.basename(par_file)
                job_name = fname.split('.')[0]
                # get job arguments
                params = pickle.load(open(par_file, 'rb'))
                os.remove(par_file)
                job_args = self.get_job_args(params)
                # submit batchjob
                batch_path = self.create_batchjob(job_name, job_args)
                subprocess.run(['msub', batch_path]).check_returncode()
                submitted = True
                print('...done. Sleeping for 1 minute')
                time.sleep(60)
            if not submitted:
                print('No new job to submit. Checking again in 1 minute')
                time.sleep(60)
        
if __name__ == '__main__':
    NS = NemoSide()
    NS.run()








