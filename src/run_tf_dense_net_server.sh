#!/usr/bin/env bash
python3 DensetNet_train.py --epochs=20 --nb_layers=5 --n_db=2 --grow_rate=12 \
--data_format='channels_last' --output_dir='/data/TaskA_2018/src/' --lr=0.1 --log_interval=10 --alpha=0.2
