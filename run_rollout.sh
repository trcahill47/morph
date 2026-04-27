#!/bin/bash

#SBATCH --job-name="visuals"

#SBATCH --output="visuals.%j.out"

#SBATCH --partition=gpu

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=4

#SBATCH --gpus=1

#SBATCH --mem=32G

#SBATCH --account=ccu108

#SBATCH -t 01:00:00



module purge

module load gpu/0.15.4



source ~/.bashrc

conda activate pytorch_py38_env



cd /expanse/lustre/scratch/tcahill1/temp_project/morph/morph/



echo "=== fns-kf M 15pct ep25 (best model) ==="

python visualize_results.py visuals \
       --pth sweep_results/fns-kf/models/M/epoch_sweep/M_15pct_ep25.pth \
       --dataset datasets/fns-kf/solution_0.npy \
       --dataset_name fns-kf \
       --model_size M \
       --rollout_horizon 20 \
       --out_dir sweep_results/fns-kf/final_visuals/M_15pct_ep25



echo "=== swe M 10pct best ==="

python visualize_results.py visuals \
       --pth sweep_results/swe/models/M/fraction_sweep/best_M_10pct.pth \
       --dataset datasets/swe-data/solution_0.npy \
       --dataset_name swe \
       --model_size M \
       --rollout_horizon 50 \
       --dataset_specs 1 1 1 128 128 \
       --out_dir sweep_results/swe/final_visuals/M_10pct_best



echo "=== All visuals complete ==="
