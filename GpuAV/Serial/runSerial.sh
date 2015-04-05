#!/bin/bash
#PBS -l procs=1,tpn=1,mem=46gb
#PBS -l walltime=20:00
#PBS -N result
#PBS -r n
#PBS -j oe
#PBS -q cpsc424

module load Langs/Intel/14 MPI/OpenMPI/1.6.5
pwd
cd $PBS_O_WORKDIR
pwd
cat $PBS_NODEFILE
./serial
