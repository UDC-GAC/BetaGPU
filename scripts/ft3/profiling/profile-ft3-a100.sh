#!/bin/sh
#SBATCH -J test_beta_profile_A100
#SBATCH -o %x-%j.txt
#SBATCH -e %x-%j_err.txt
#SBATCH -c 32
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 5:00

module load cesga/2020 cuda/12.2.0
cd build
cmake ..
make clean
make -j`nproc`
ncu -o a100-profile -f "$@"