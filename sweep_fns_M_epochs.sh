#!/bin/bash

# ================================================================

# sweep_fns_M_epochs.sh

# Medium model, fns-kf dataset, epoch-axis sweep

# Fixed at 15% data, trains to 25 epochs, checkpoints every 5

# Produces 5 .pth files + 5 CSV rows showing learning over time

# Estimated: 25 epochs x ~3 min = ~75 min -> tight, patience=5

# helps it stop early if converged

# ================================================================

#SBATCH --job-name="fns_M_epch"

#SBATCH --output="fns_M_epch.%j.%N.out"

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
  --dataset fns-kf/solution_0.npy \
  --dataset_name fns-kf \
  --sweep_mode epochs \
  --train_pct 15 \
  --max_epochs 25 \
  --checkpoint_every 5 \
  --rollout 20


