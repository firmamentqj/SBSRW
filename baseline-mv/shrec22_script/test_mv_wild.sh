#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=mv_test_wild
#SBATCH --output=mv_test_wild.out

module purge
source ~/.bashrc
conda activate /scratch/sy2366/.conda_env/shyuan3D
cd /scratch/sy2366/shrec22/MVTN
python run_shrec22.py \
      --config_file ./shrec22_config/config_mv_wild_01.yaml \
      --run_mode test \
      --mvnetwork mvcnn \
      --nb_views 12 \
      --views_config spherical \
      --pc_rendering \
      --batch_size 1 \
      --task wild \
      --epochs 100