#!/bin/bash
#SBATCH -J output/cuda_once_exec_beta_cdf_A100
#SBATCH -o %x-%j.out
#SBATCH -c 32
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 30:00

module load cesga/2020 cuda/12.2.0
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
sleep 1
make clean
sleep 1
make VERBOSE=1
sleep 1

EXECUTABLE=../bin/bench_base

# run the script for different values of parametrs
# for examples num elements = 10000000, 100000000 and 1000000000

for i in 10000000 100000000 1000000000
do
    for j in cuda omp cuda_omp
    do
        for f in betacdf
        do
            for n in {1..7}
            do
                OMP_NUM_THREADS=32 $EXECUTABLE $i 1 $j $f -s
                OMP_NUM_THREADS=32 $EXECUTABLE $i 1 $j $f -p -s
            done
        done
    done
done