#!/usr/bin/env bash
python3 DensetNet_train.py --batch_size=10 --epochs=1 --nb_layers=4 --n_db=2 --grow_rate=16 \
--data_format='channels_last' --output_dir='/data/Task1_2018/src/' --lr=0.1 --log_interval=10 --alpha=0.2
