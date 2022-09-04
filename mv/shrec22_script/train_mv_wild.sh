#!/bin/bash

python run_shrec22.py \
      --config_file ./shrec22_config/config_mv_wild_01.yaml \
      --run_mode train \
      --mvnetwork mvcnn \
      --nb_views 12 \
      --views_config spherical \
      --pc_rendering \
      --batch_size 8 \
      --task wild \
      --epochs 100