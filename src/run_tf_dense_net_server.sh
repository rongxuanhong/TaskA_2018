#!/usr/bin/env bash

python3 DensetNet_train.py --batch_size=32 --epochs=30 --nb_layers=5 --n_db=5 --grow_rate=24 \
--data_format='channels_last' --output_dir='/data/TaskA_2018/src/check_point' --lr=0.1 --log_interval=50 --alpha=0.2

