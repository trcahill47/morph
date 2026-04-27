#!/bin/bash

#SBATCH --job-name="visualize"

#SBATCH --output="visualize.%j.out"

#SBATCH --partition=shared

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=4

#SBATCH --mem=32G

#SBATCH --account=ccu108

#SBATCH -t 00:30:00

 

module purge

module load cpu/0.15.4

 

source ~/.bashrc

conda activate pytorch_py38_env

 

cd /expanse/lustre/scratch/tcahill1/temp_project/morph/morph/

 

echo "=== fns-kf plots ==="

python visualize_results.py plots \
  --csv sweep_results/fns-kf/master_metrics.csv \
  --dataset fns-kf \
  --model_sizes M L \
  --out_dir sweep_results/fns-kf/plots

 

echo "=== swe plots ==="

python visualize_results.py plots \
  --csv sweep_results/swe/master_metrics.csv \
  --dataset swe \
  --model_sizes M L \
  --out_dir sweep_results/swe/plots

 

echo "=== cross-dataset comparison ==="

python visualize_results.py compare \
  --csv1 sweep_results/fns-kf/master_metrics.csv \
  --csv2 sweep_results/swe/master_metrics.csv \
  --name1 fns-kf \
  --name2 swe \
  --out_dir sweep_results/plots/cross_dataset \
  --model_sizes M L
 

echo "=== All visualizations complete ==="
