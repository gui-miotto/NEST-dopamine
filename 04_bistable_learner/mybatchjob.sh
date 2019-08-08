#!/bin/bash

#MSUB -l nodes=5:ppn=20
#MSUB -l walltime=24:00:00
#MSUB -l pmem=6gb
##MSUB -l naccesspolicy=singlejob
#MSUB -m bea -M alessang@tf.uni-freiburg.de
#MSUB -v MPIRUN_OPTIONS="--bind-to core --map-by core -report-bindings"
#MSUB -N run_04

TASKNAME=run_04
SCRIPTPATH=$HOME/code/$TASKNAME.py
OUTDIR=$(ws_find learner)/$TASKNAME

module load system/modules/testing
module load mpi/openmpi/3.1-gnu-7.3
module load neuro/nest/2.16.0-python-3.7.0

mpirun python $SCRIPTPATH $OUTDIR --n_trials 300