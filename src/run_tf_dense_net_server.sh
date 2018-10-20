#!/usr/bin/env bash
python3 DensetNet_train.py --batch_size=100 --epochs=7 --nb_layers=40 --n_db=3 --grow_rate=24 \
--data_format='channels_last' --output_dir='/data/taska_2018/src/check_point' --lr=0.0001 --log_interval=10 --alpha=0.2

