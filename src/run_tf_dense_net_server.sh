#!/usr/bin/env bash
python3 DensetNet_train.py --batch_size=64 --epochs=3 --nb_layers=30 --n_db=4 --grow_rate=32 \
--data_format='channels_last' --output_dir='/data/TaskA_2018/src/check_point' --lr=0.00001 --log_interval=10 --alpha=0.2

