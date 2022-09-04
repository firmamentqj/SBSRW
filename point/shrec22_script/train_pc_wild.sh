#!/bin/bash

python run_sbsr.py \
      --run_mode train \
      --batch_size 32 \
      --task wild \
      --config_file ./shrec22_config/config_pc_wild_01.yaml \
      --epochs 100