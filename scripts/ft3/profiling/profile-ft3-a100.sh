#!/bin/bash
#SBATCH -J output/test_beta_profile_A100
#SBATCH -o %x-%j.out
#SBATCH -c 32
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 5:00

module load cesga/2020 cuda/11.5.0
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
sleep 1
make clean
sleep 1
make VERBOSE=1
sleep 1
echo "$@"
ncu -o profile -f "$@"
ncu --section ComputeWorkloadAnalysis -o workload_analysis -f "$@"
