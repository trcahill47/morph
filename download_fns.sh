#!/bin/bash
#SBATCH --job-name="dl_fns"
#SBATCH --output="dl_fns.%j.out"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -A ccu108
#SBATCH --export=ALL

# Load modules
module purge
module load cpu/0.15.4
module load anaconda3/2020.11
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate your environment
conda activate pytorch_py38_env

# Navigate to your project directory
cd /expanse/lustre/scratch/tcahill1/temp_project/morph/morph

# 1. Download the .nc file using hf download
hf download camlab-ethz/FNS-KF solution_0.nc --repo-type dataset --local-dir "datasets/fns-kf"

# 2. Convert the .nc file to .npy
python src/utils/convert_nc_h5_to_npy.py --data_format nc --source_nc_file datasets/fns-kf/solution_0.nc --target_npy_loc datasets/fns-kf