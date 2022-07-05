#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=mv_train_cad
#SBATCH --output=mv_train_cad.out

module purge
source ~/.bashrc
conda activate /scratch/sy2366/.conda_env/shyuan3D
cd /scratch/sy2366/shrec22/MVTN
python run_shrec22.py \
      --config_file ./shrec22_config/config_mv_cad_01.yaml \
      --run_mode train \
      --mvnetwork mvcnn \
      --nb_views 12 \
      --views_config spherical \
      --pc_rendering \
      --batch_size 8 \
      --epochs 80