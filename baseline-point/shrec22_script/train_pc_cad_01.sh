#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=re
#SBATCH --output=re.out

module purge
source ~/.bashrc
conda activate /scratch/sy2366/.conda_env/shyuan3D
#cd /scratch/sy2366/view/semantic
cd /scratch/sy2366/shrec22/semantic
python run_sbsr.py \
      --run_mode train \
      --batch_size 20 \
      --task cad \
      --config_file ./shrec22_config/config_pc_cad_01.yaml \
      --epochs 150