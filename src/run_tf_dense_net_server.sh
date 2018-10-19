#!/usr/bin/env bash
python3 DensetNet_train.py --batch_size=64 --epochs=10 --nb_layers=40 --n_db=3 --grow_rate=24 \
--data_format='channels_last' --output_dir='/data/taska_2018/src/check_point' --lr=0.1 --log_interval=15 --alpha=0.2

