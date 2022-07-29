
python run_shrec22.py \
      --config_file ./shrec22_config/config_mv_cad_01.yaml \
      --run_mode test \
      --mvnetwork mvcnn \
      --nb_views 12 \
      --views_config spherical \
      --pc_rendering \
      --batch_size 1 \
      --epochs 80