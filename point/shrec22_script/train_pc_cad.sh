#!/bin/bash

python run_sbsr.py \
      --run_mode train \
      --batch_size 20 \
      --task cad \
      --config_file ./shrec22_config/config_pc_cad_01.yaml \
      --epochs 150