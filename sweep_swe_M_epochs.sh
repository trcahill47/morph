#!/bin/bash

# ================================================================

# sweep_swe_M_epochs.sh

# Medium model, SWE dataset, epoch sweep

# ================================================================

#SBATCH --job-name="swe_M_epch"

#SBATCH --output="swe_M_epch.%j.%N.out"

#SBATCH --partition=gpu

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --gpus=1

#SBATCH --mem=32G

#SBATCH --account=ccu108

#SBATCH --export=ALL

#SBATCH -t 01:10:00



module purge

module load cpu/0.15.4

source ~/.bashrc

conda activate pytorch_py38_env



cd /expanse/lustre/scratch/tcahill1/temp_project/morph/morph/



python sweep_script.py \
  --model_size M \
  --dataset swe-data/solution_0.npy \
  --dataset_name swe \
  --sweep_mode epochs \
  --train_pct 15 \
  --max_epochs 25 \
  --checkpoint_every 5 \
  --rollout 20 \
  --patience 10 \
  --dataset_specs 1 1 1 128 128
