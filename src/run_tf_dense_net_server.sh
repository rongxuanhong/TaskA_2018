#!/usr/bin/env bash
python3 DensetNet_train.py --batch_size=64 --epochs=30 --nb_layers=5 --n_db=3 --grow_rate=16 \
--data_format='channels_last' --output_dir='/data/TaskA_2018/src/check_point' --lr=0.01 --log_interval=10 --alpha=0.2
