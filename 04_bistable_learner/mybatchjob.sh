#!/bin/bash

## FOR THE BIG NETWORK
##MSUB -l nodes=5:ppn=20
##MSUB -l walltime=24:00:00

# FOR THE SMALL NETWORK
#MSUB -l nodes=1:ppn=20
#MSUB -l walltime=3:00:00

#MSUB -l pmem=6gb
##MSUB -l naccesspolicy=singlejob
#MSUB -m bea -M alessang@tf.uni-freiburg.de
#MSUB -v MPIRUN_OPTIONS="--bind-to core --map-by core -report-bindings"
#MSUB -N final_06

TASKNAME=final_06
SCRIPTPATH=$HOME/code/$TASKNAME.py
OUTDIR=$(ws_find learner)/$TASKNAME

module load system/modules/testing
module load mpi/openmpi/3.1-gnu-8.2
module load neuro/nest/2.16.0-python-3.7.0

mpirun python $SCRIPTPATH $OUTDIR 