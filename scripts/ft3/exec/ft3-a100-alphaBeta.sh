#!/bin/bash
#SBATCH -J output/cuda_once_exec_beta_cdf_A100
#SBATCH -o %x-%j.out
#SBATCH -c 32
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 02:30:00

module load cesga/2020 cuda/12.2.0

cd build

## check if the number of parameters is correct
if [ "$#" -ne 2 ]; then
    ALPHA=9.34
    BETA=11.34
else
    ALPHA=$1
    BETA=$2
fi

EXECUTABLE=../bin/bench_base

# run the script for different values of parametrs
# for examples num elements = 10000000, 100000000 and 1000000000

for i in 10000000 100000000 1000000000
do
    for j in cuda cuda_omp omp seq
    do
        for f in betacdf
        do
            for n in {1..7}
            do
                OMP_NUM_THREADS=32 $EXECUTABLE $i 1 $j $f -s $ALPHA $BETA
                OMP_NUM_THREADS=32 $EXECUTABLE $i 1 $j $f -p -s $ALPHA $BETA
            done
        done
    done
done